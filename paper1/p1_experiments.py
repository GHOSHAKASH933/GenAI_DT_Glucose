"""
Paper 1 experiments: a generative digital twin for personalised, real-time glucose
forecasting and scenario simulation, evaluated on real CGM data
(ShanghaiT1DM / ShanghaiT2DM, Zhao et al., Sci. Data 2023).

P1  point accuracy at 30/60/120 min, Clarke error grid
P2  probabilistic quality: CRPS, interval coverage and width
P3  hypoglycaemia risk in the next 60 min: AUROC, AUPRC, Brier score
P4  scenario realism: temporal coherence and distance of sampled trajectories
P5  real-time personalisation with 1, 2 and 4 days of a new patient's data
P6  federated training across four non-IID clients
P7  edge deployment: distillation and int8 quantisation

All splits are by patient and repeated over several seeds. The twin is never
used to choose treatment here.
"""
from __future__ import annotations
import argparse, copy, io, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "realdata")); sys.path.insert(0, str(Path(__file__).parent))

import genai_dt.data as D
D.CONTEXT, D.HORIZON = 8, 8
D.CTX_DIM = D.CONTEXT * 3 + D.HORIZON * 2
D.I_SCALE, D.C_SCALE = 6.0, 1.0
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from genai_dt.models import (CVAETwin, PointMLP, EdgeStudent, train_cvae, train_point,
                             Persistence, RidgeAR, mlp)
from genai_dt.federated import train_local_only, train_fedavg, personalise
from shanghai import load_all
from windowing import windows, pack, split

torch.set_num_threads(1)
H = D.HORIZON
STEP = {30: 1, 60: 3, 120: 7}          # index of the 30/60/120-min step (15-min grid)


# ------------------------------------------------------------------ extra models

class GaussMLP(nn.Module):
    """Heteroscedastic Gaussian MLP: a probabilistic baseline without a latent."""

    def __init__(self, ctx_dim=None, horizon=H, hidden=128):
        super().__init__()
        self.net = mlp([ctx_dim or D.CTX_DIM, hidden, hidden, 2 * horizon])
        self.horizon = horizon

    def forward(self, x):
        m, ls = self.net(x).chunk(2, -1)
        return m, ls.clamp(-6, 2)

    @torch.no_grad()
    def sample(self, x, n_samples=32, include_obs_noise=True):
        m, ls = self(x)
        if not include_obs_noise:
            return m.unsqueeze(0).expand(n_samples, *m.shape)
        return m.unsqueeze(0) + ls.exp().unsqueeze(0) * torch.randn(n_samples, *m.shape)


def train_gauss(model, X, Y, epochs, seed, bs=256, lr=2e-3):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    X, Y = torch.as_tensor(X), torch.as_tensor(Y)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for b in range(0, len(X), bs):
            i = perm[b:b + bs]
            m, ls = model(X[i])
            loss = (ls + 0.5 * ((Y[i] - m) / ls.exp()) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


class GRUForecaster(nn.Module):
    """Sequence baseline: GRU over the 2-h history, plus the planned inputs."""

    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(3, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + 2 * H, 128), nn.GELU(), nn.Linear(128, H))

    def forward(self, x):
        K = D.CONTEXT
        seq = torch.stack([x[:, :K], x[:, K:2 * K], x[:, 2 * K:3 * K]], -1)
        _, h = self.gru(seq)
        return self.head(torch.cat([h[-1], x[:, 3 * K:]], -1))


# ------------------------------------------------------------------ helpers

def mg(y):
    return D.denorm(np.asarray(y))


def samples_of(model, X, n=64, bs=4096):
    out = []
    with torch.no_grad():
        for b in range(0, len(X), bs):
            out.append(model.sample(torch.as_tensor(X[b:b + bs]), n_samples=n).numpy())
    return mg(np.concatenate(out, 1))                    # (S, N, H) mg/dL


def point(model, X):
    with torch.no_grad():
        if isinstance(model, (CVAETwin, GaussMLP)):
            return samples_of(model, X, 48).mean(0)
        return mg(model(torch.as_tensor(X)).numpy())


def errors(pred, Y):
    y = mg(Y)
    out = {}
    for mins, k in STEP.items():
        e = pred[:, k] - y[:, k]
        out[f"rmse_{mins}"] = float(np.sqrt((e ** 2).mean()))
        out[f"mae_{mins}"] = float(np.abs(e).mean())
    out["rmse_all"] = float(np.sqrt(((pred - y) ** 2).mean()))
    a, ab = clarke(y[:, STEP[60]], pred[:, STEP[60]])
    out["ceg_A_60"], out["ceg_AB_60"] = a, ab
    return out


def clarke(ref, pred):
    """Clarke error grid zones A and A+B (Clarke et al., Diabetes Care 1987)."""
    ref = np.asarray(ref, float); pred = np.asarray(pred, float)
    A = ((np.abs(pred - ref) <= 0.2 * ref) | ((ref < 70) & (pred < 70)))
    E = ((ref <= 70) & (pred >= 180)) | ((ref >= 180) & (pred <= 70))
    C = ((ref >= 70) & (ref <= 290) & (pred >= ref + 110)) | \
        ((ref >= 130) & (ref <= 180) & (pred <= (7 / 5) * ref - 182))
    Dz = ((ref >= 240) & (pred >= 70) & (pred <= 180)) | \
        ((ref <= 175 / 3) & (pred <= 180) & (pred >= 70)) | \
        ((ref >= 175 / 3) & (ref <= 70) & (pred >= (6 / 5) * ref))
    B = ~(A | C | Dz | E)
    return float(A.mean() * 100), float((A | B).mean() * 100)


def crps(samples, y):
    """Sample CRPS: E|X - y| - 0.5 E|X - X'| (per window and step, averaged)."""
    S = samples.shape[0]
    t1 = np.abs(samples - y[None]).mean(0)
    s = np.sort(samples, 0)
    w = (2 * np.arange(1, S + 1) - S - 1)[:, None, None]
    t2 = (w * s).sum(0) / (S * S)              # = 0.5 E|X - X'|
    return t1 - t2


def prob_metrics(samples, Y, alpha=0.10):
    y = mg(Y)
    lo = np.quantile(samples, alpha / 2, 0); hi = np.quantile(samples, 1 - alpha / 2, 0)
    c = crps(samples, y)
    out = {"crps_60": float(c[:, STEP[60]].mean()), "crps_120": float(c[:, STEP[120]].mean()),
           "crps_all": float(c.mean()),
           "cov90_60": float(((y[:, STEP[60]] >= lo[:, STEP[60]]) & (y[:, STEP[60]] <= hi[:, STEP[60]])).mean()),
           "width90_60": float((hi - lo)[:, STEP[60]].mean())}
    return out


def hypo_metrics(score, Y, prob=None):
    """Event: glucose below 70 mg/dL at any point in the next 60 minutes."""
    ev = (mg(Y)[:, :STEP[60] + 1].min(1) < 70).astype(int)
    out = {"prevalence": float(ev.mean())}
    if ev.sum() == 0 or ev.sum() == len(ev):
        return out
    out["auroc"] = float(roc_auc_score(ev, score))
    out["auprc"] = float(average_precision_score(ev, score))
    if prob is not None:
        out["brier"] = float(((prob - ev) ** 2).mean())
    return out


def realism(samples, Y, rng, n=2000):
    """How realistic are sampled 2-h scenarios?
    lag1: lag-1 autocorrelation of 15-min increments (real vs sampled)
    energy: energy distance between sampled and real trajectories (lower is better)"""
    y = mg(Y)
    idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
    real = y[idx]; fake = samples[0, idx]
    def lag1(T):
        d = np.diff(T, axis=1)
        a, b = d[:, :-1].ravel(), d[:, 1:].ravel()
        return float(np.corrcoef(a, b)[0, 1])
    def edist(A, B, m=600):
        A = A[:m]; B = B[:m]
        ab = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1)).mean()
        aa = np.sqrt(((A[:, None] - A[None]) ** 2).sum(-1)).mean()
        bb = np.sqrt(((B[:, None] - B[None]) ** 2).sum(-1)).mean()
        return float(2 * ab - aa - bb)
    return {"lag1_real": lag1(real), "lag1_sampled": lag1(fake), "energy": edist(fake, real)}


def model_bytes(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); return buf.tell()


def latency_ms(fn, x, reps=200):
    fn(x)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(x); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


# ------------------------------------------------------------------ one seed

def run_seed(recs, seed, cfg):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    sp = split(recs, seed)
    Xtr, Ytr, Ttr = pack(sp["train"], cfg["stride"])
    Xva, Yva, _ = pack(sp["calib"], cfg["stride"])
    Xte, Yte, Tte = pack(sp["test"], cfg["stride"])
    res = dict(seed=seed, n_windows=dict(train=len(Xtr), val=len(Xva), test=len(Xte)),
               n_patients={k: len({r["dtype"] + r["pid"] for r in v}) for k, v in sp.items()})
    E = cfg["epochs"]

    # --- models
    torch.manual_seed(seed)
    models = {"persistence": Persistence(), "ridge": RidgeAR(lam=5.0).fit(Xtr, Ytr)}
    torch.manual_seed(seed); models["mlp"] = train_point(PointMLP(), Xtr, Ytr, epochs=E, seed=seed)
    torch.manual_seed(seed); gru = GRUForecaster()
    models["gru"] = train_point(gru, Xtr, Ytr, epochs=E, seed=seed)
    torch.manual_seed(seed); models["gauss_mlp"] = train_gauss(GaussMLP(), Xtr, Ytr, E, seed)

    # twin: KL weight chosen on validation patients (CRPS), never on test
    best, best_crps = None, np.inf
    for beta in cfg["betas"]:
        torch.manual_seed(seed)
        m = train_cvae(CVAETwin(z_dim=cfg["z_dim"]), Xtr, Ytr, epochs=E, seed=seed, beta=beta)
        v = prob_metrics(samples_of(m, Xva, 32), Yva)["crps_all"]
        if v < best_crps:
            best, best_crps, best_beta = m, v, beta
    models["twin"] = best
    res["twin_beta"] = best_beta

    # --- P1 point accuracy
    pt = {}
    for k, m in models.items():
        p = mg(m.predict(Xte)) if k in ("persistence", "ridge") else point(m, Xte)
        pt[k] = errors(p, Yte)
    res["accuracy"] = pt

    # --- P2 probabilistic quality, P3 hypo risk, P4 realism
    prob, hyp, real = {}, {}, {}
    for k in ("gauss_mlp", "twin"):
        s = samples_of(models[k], Xte, cfg["n_samples"])
        prob[k] = prob_metrics(s, Yte)
        pmin = (s[:, :, :STEP[60] + 1].min(2) < 70).mean(0)
        hyp[k] = hypo_metrics(pmin, Yte, prob=pmin)
        real[k] = realism(s, Yte, rng)
    # twin scenarios drawn from the latent only (no per-step observation noise)
    with torch.no_grad():
        sl = mg(torch.cat([models["twin"].sample(torch.as_tensor(Xte[b:b + 4096]), n_samples=8,
                                                 include_obs_noise=False)
                           for b in range(0, len(Xte), 4096)], 1).numpy())
    real["twin_latent"] = realism(sl, Yte, rng)
    real["twin_latent"]["spread_60"] = float(sl[:, :, STEP[60]].std(0).mean())
    prob["twin_latent"] = prob_metrics(sl, Yte)
    if seed == 0:   # a few test windows for the scenario figure
        yte = mg(Yte); pick = []
        for dt in ("T1DM", "T2DM"):
            idx = np.where(Tte == dt)[0]
            var = yte[idx].std(1)
            pick += [int(i) for i in idx[np.argsort(var)[-int(0.25 * len(idx))::max(1, int(0.25 * len(idx)) // 2)][:2]]]
        Xp = torch.as_tensor(Xte[pick])
        with torch.no_grad():
            tw = mg(models["twin"].sample(Xp, n_samples=20, include_obs_noise=False).numpy())
            ga = mg(models["gauss_mlp"].sample(Xp, n_samples=20).numpy())
        res["examples"] = dict(dtype=[str(Tte[i]) for i in pick],
                               history=[(Xte[i, :D.CONTEXT] * D.G_SCALE + D.G_MEAN).tolist() for i in pick],
                               truth=yte[pick].tolist(), twin=tw.transpose(1, 0, 2).tolist(),
                               gauss=ga.transpose(1, 0, 2).tolist())
    k60 = STEP[60]; yv = mg(Yva)[:, k60]; yt = mg(Yte)[:, k60]
    for k in ("gauss_mlp", "twin"):
        sv = samples_of(models[k], Xva, cfg["n_samples"])[:, :, k60]
        st = samples_of(models[k], Xte, cfg["n_samples"])[:, :, k60]
        lv, hv = np.quantile(sv, 0.05, 0), np.quantile(sv, 0.95, 0)
        sc = np.maximum(lv - yv, yv - hv); n = len(sc)
        Q = float(np.sort(sc)[min(n - 1, int(np.ceil((n + 1) * 0.9)) - 1)])
        lt, ht = np.quantile(st, 0.05, 0) - Q, np.quantile(st, 0.95, 0) + Q
        prob[k]["conf_cov90_60"] = float(((yt >= lt) & (yt <= ht)).mean())
        prob[k]["conf_width90_60"] = float((ht - lt).mean())
        prob[k]["conf_Q"] = Q
        for dt in ("T1DM", "T2DM"):
            mk = Tte == dt
            prob[k][f"conf_cov90_60_{dt}"] = float(((yt[mk] >= lt[mk]) & (yt[mk] <= ht[mk])).mean())
            prob[k][f"cov90_60_{dt}"] = float(((yt[mk] >= np.quantile(st, 0.05, 0)[mk]) & (yt[mk] <= np.quantile(st, 0.95, 0)[mk])).mean())
    for k in ("persistence", "ridge", "mlp", "gru"):
        p = mg(models[k].predict(Xte)) if k in ("persistence", "ridge") else point(models[k], Xte)
        hyp[k] = hypo_metrics(-p[:, :STEP[60] + 1].min(1), Yte)
    res["probabilistic"], res["hypo"], res["realism"] = prob, hyp, real

    # --- P5 real-time personalisation of the twin on a new patient
    pers = {f"{d}d": [] for d in cfg["adapt_days"]}; pers["population"] = []
    pers.update({f"scratch_{d}d": [] for d in cfg["adapt_days"]})
    pers.update({f"dec_{d}d": [] for d in cfg["adapt_days"]})
    for r in sp["test"]:
        n = len(r["glucose"]); cut_eval = int(0.5 * n)
        ev = dict(r, glucose=r["glucose"][cut_eval:], insulin=r["insulin"][cut_eval:], carbs=r["carbs"][cut_eval:])
        Xe, Ye = windows(ev, cfg["stride"])
        if len(Xe) < 20:
            continue
        pers["population"].append(errors(point(models["twin"], Xe), Ye)["rmse_60"])
        for d in cfg["adapt_days"]:
            m_end = min(d * 96, cut_eval)
            ad = dict(r, glucose=r["glucose"][:m_end], insulin=r["insulin"][:m_end], carbs=r["carbs"][:m_end])
            Xa, Ya = windows(ad, 1)
            if len(Xa) < 10:
                for pre in ("", "scratch_", "dec_"):
                    pers[f"{pre}{d}d"].append(np.nan)
                continue
            m = copy.deepcopy(models["twin"])
            for prm in m.enc.parameters():
                prm.requires_grad_(False)
            for layer in list(m.dec)[:-1]:
                for prm in layer.parameters():
                    prm.requires_grad_(False)
            train_cvae(m, Xa, Ya, epochs=cfg["adapt_epochs"], seed=seed, lr=5e-4, beta=best_beta)
            pers[f"{d}d"].append(errors(point(m, Xe), Ye)["rmse_60"])
            m2 = copy.deepcopy(models["twin"])
            for prm in m2.enc.parameters():
                prm.requires_grad_(False)
            train_cvae(m2, Xa, Ya, epochs=cfg["adapt_epochs"], seed=seed, lr=1e-3, beta=best_beta)
            pers[f"dec_{d}d"].append(errors(point(m2, Xe), Ye)["rmse_60"])
            torch.manual_seed(seed)
            sc = train_cvae(CVAETwin(z_dim=cfg["z_dim"]), Xa, Ya, epochs=cfg["adapt_epochs"] * 3,
                            seed=seed, beta=best_beta)
            pers[f"scratch_{d}d"].append(errors(point(sc, Xe), Ye)["rmse_60"])
    res["personalisation"] = {k: float(np.nanmean(v)) for k, v in pers.items()}
    res["personalisation_n_patients"] = len(pers["population"])
    res["personalisation_per_record"] = {k: [float(x) for x in v] for k, v in pers.items()}

    # --- P6 federated: one T1DM client + three T2DM clients, split by patient
    t2 = sorted({r["pid"] for r in sp["train"] if r["dtype"] == "T2DM"})
    rr = np.random.default_rng(seed + 7); rr.shuffle(t2)
    groups = {"T1DM": [r for r in sp["train"] if r["dtype"] == "T1DM"]}
    for j in range(3):
        ids = set(t2[j::3]); groups[f"T2DM-{j+1}"] = [r for r in sp["train"] if r["dtype"] == "T2DM" and r["pid"] in ids]
    site_data = {k: pack(v, cfg["stride"])[:2] for k, v in groups.items()}
    test_site = {"T1DM": [r for r in sp["test"] if r["dtype"] == "T1DM"],
                 **{f"T2DM-{j+1}": [] for j in range(3)}}
    t2te = sorted({r["pid"] for r in sp["test"] if r["dtype"] == "T2DM"})
    for j, pid in enumerate(t2te):
        test_site[f"T2DM-{j % 3 + 1}"] += [r for r in sp["test"] if r["dtype"] == "T2DM" and r["pid"] == pid]
    test_xy = {k: pack(v, cfg["stride"])[:2] for k, v in test_site.items() if v}
    kw = dict(beta=best_beta)
    loc = train_local_only(site_data, seed=seed, epochs=E, cls=lambda: CVAETwin(z_dim=cfg["z_dim"]), **kw)
    glob = train_fedavg(site_data, seed=seed, rounds=cfg["rounds"], local_epochs=cfg["local_epochs"],
                        cls=lambda: CVAETwin(z_dim=cfg["z_dim"]), **kw)
    per = personalise(glob, site_data, seed=seed, epochs=cfg["pers_epochs"],
                      cls=lambda: CVAETwin(z_dim=cfg["z_dim"]), **kw)
    fed = {"local_only": {}, "fedavg": {}, "fedavg_personalised": {}, "centralised": {}}
    for site, (Xs, Ys) in test_xy.items():
        fed["local_only"][site] = errors(point(loc[site], Xs), Ys)["rmse_60"]
        fed["fedavg"][site] = errors(point(glob, Xs), Ys)["rmse_60"]
        fed["fedavg_personalised"][site] = errors(point(per[site], Xs), Ys)["rmse_60"]
        fed["centralised"][site] = errors(point(models["twin"], Xs), Ys)["rmse_60"]
    res["federated"] = fed
    res["federated_sizes"] = {k: int(len(v[0])) for k, v in site_data.items()}

    # --- P7 edge: distillation and int8 dynamic quantisation
    twin = models["twin"]
    S = []
    with torch.no_grad():
        for b in range(0, len(Xtr), 4096):
            S.append(twin.sample(torch.as_tensor(Xtr[b:b + 4096]), n_samples=32))
    S = torch.cat(S, 1)
    tq = [torch.quantile(S, q, 0) for q in (0.05, 0.5, 0.95)]
    torch.manual_seed(seed); stu = EdgeStudent(ctx_dim=D.CTX_DIM, horizon=H)
    opt = torch.optim.AdamW(stu.parameters(), lr=3e-3); Xt = torch.as_tensor(Xtr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(E):
        perm = torch.randperm(len(Xt), generator=g)
        for b in range(0, len(Xt), 256):
            i = perm[b:b + 256]; lo, md, hi = stu(Xt[i])
            loss = sum(((a - t[i]) ** 2).mean() for a, t in zip((lo, md, hi), tq))
            opt.zero_grad(); loss.backward(); opt.step()
    stu.eval()
    q8 = torch.ao.quantization.quantize_dynamic(copy.deepcopy(twin), {nn.Linear}, dtype=torch.qint8)
    x1 = torch.as_tensor(Xte[:1])
    with torch.no_grad():
        stu_pred = mg(stu(torch.as_tensor(Xte))[1].numpy())
        q8_pred = mg(q8.sample(torch.as_tensor(Xte), n_samples=32, include_obs_noise=False).mean(0).numpy())
        edge = {
            "twin": dict(kb=model_bytes(twin) / 1024, ms=latency_ms(lambda x: twin.sample(x, n_samples=32), x1),
                         rmse_60=pt["twin"]["rmse_60"]),
            "twin_int8": dict(kb=model_bytes(q8) / 1024, ms=latency_ms(lambda x: q8.sample(x, n_samples=32), x1),
                              rmse_60=errors(q8_pred, Yte)["rmse_60"]),
            "student": dict(kb=model_bytes(stu) / 1024, ms=latency_ms(lambda x: stu(x), x1),
                            rmse_60=errors(stu_pred, Yte)["rmse_60"]),
        }
    res["edge"] = edge
    res["seconds"] = round(time.time() - t0)
    print(f"  seed {seed} done in {res['seconds']}s (twin beta={best_beta})", flush=True)
    return res


CFG = dict(epochs=25, stride=2, betas=(0.1, 0.5), z_dim=16, n_samples=64,
           adapt_days=(1, 2, 4), adapt_epochs=10, rounds=8, local_epochs=3, pers_epochs=5)
QUICK = dict(epochs=2, stride=6, betas=(0.1,), z_dim=8, n_samples=16,
             adapt_days=(1,), adapt_epochs=2, rounds=1, local_epochs=1, pers_epochs=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "paper1_results.json"))
    a = ap.parse_args()
    cfg = QUICK if a.quick else CFG
    recs = load_all(a.data)
    print(f"loaded {len(recs)} records; config {cfg}", flush=True)
    runs = []
    for s in range(a.seeds):
        runs.append(run_seed(recs, s, cfg))
        json.dump(dict(config={k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
                       dataset="ShanghaiT1DM+ShanghaiT2DM", runs=runs), open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
