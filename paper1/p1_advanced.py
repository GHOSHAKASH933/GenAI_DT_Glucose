"""
Paper 1, extended benchmark and analyses. Same patient-level splits, seeds and
metrics as p1_experiments.py, on the real ShanghaiT1DM/T2DM records.

A1  stronger baselines
      Transformer forecaster with a Gaussian head
      conditional diffusion model (DDIM sampler), on glucose and on ridge residuals
      quantile network with split-conformal correction (CQR)
A2  subgroup (T1DM / T2DM), glycaemic-range, calibration and hypoglycaemia
    operating-point analyses for every model; pooled and per-type (Mondrian)
    split-conformal intervals
A3  input ablations of the hybrid twin (insulin, meals, planned inputs)
A4  training-set size (25 % and 50 % of the training patients; ridge and hybrid twin)

Nothing here is tuned on test patients. Hyper-parameters of the earlier models
are taken from the main run (KL weight chosen on validation patients); the new
models use fixed settings.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import p1_experiments as P
from p1_experiments import (D, torch, nn, CVAETwin, RidgeAR, GRUForecaster, GaussMLP, train_gauss,
                            train_cvae, train_point, PointMLP, mlp, mg, errors, prob_metrics, hypo_metrics,
                            crps, realism, pack, split, load_all, STEP, ROOT, H, samples_of)
from p1_hybrid import HybridTwin
from sklearn.metrics import roc_curve

K = D.CONTEXT
K60 = STEP[60]
QL = np.linspace(0.05, 0.95, 19)


# ------------------------------------------------------------------ models

def fit(model, X, Y, loss_fn, epochs, seed, lr=2e-3, bs=256, clip=1.0):
    g = torch.Generator().manual_seed(seed)
    X, Y = torch.as_tensor(X), torch.as_tensor(Y)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(X) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                  pct_start=min(0.3, max(0.1, 3.0 / steps)))  # 0.1 unless the run is tiny
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for b in range(0, len(X), bs):
            i = perm[b:b + bs]
            loss = loss_fn(model, X[i], Y[i])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); sched.step()
    model.eval()
    return model


class TransformerGauss(nn.Module):
    """Small Transformer encoder over the 2-h history (glucose, insulin, meals per step),
    plus the planned inputs; Gaussian head, independent across the horizon."""

    def __init__(self, d=48, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(3, d)
        self.pos = nn.Parameter(torch.randn(1, K, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = mlp([2 * d + 2 * H, 128, 2 * H])

    def forward(self, x):
        seq = torch.stack([x[:, :K], x[:, K:2 * K], x[:, 2 * K:3 * K]], -1)
        h = self.enc(self.inp(seq) + self.pos)
        m, ls = self.head(torch.cat([h[:, -1], h.mean(1), x[:, 3 * K:]], -1)).chunk(2, -1)
        return m, ls.clamp(-6, 2)

    @torch.no_grad()
    def sample(self, x, n_samples=32, include_obs_noise=True):
        m, ls = self(x)
        if not include_obs_noise:
            return m.unsqueeze(0).expand(n_samples, *m.shape)
        return m.unsqueeze(0) + ls.exp().unsqueeze(0) * torch.randn(n_samples, *m.shape)


class MeanBase:
    """Wraps the mean head of a trained network so it can serve as the base of a hybrid twin."""

    def __init__(self, net):
        self.net = net

    def predict(self, X):
        with torch.no_grad():
            return self.net(torch.as_tensor(np.asarray(X)))[0].numpy().astype(np.float32)


def gauss_loss(model, x, y):
    m, ls = model(x)
    return (ls + 0.5 * ((y - m) / ls.exp()) ** 2).sum(-1).mean()


class QuantileNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = mlp([D.CTX_DIM, hidden, hidden, H * len(QL)])

    def forward(self, x):
        return self.net(x).view(-1, len(QL), H)


def pinball(model, x, y):
    e = y[:, None, :] - model(x)
    tau = torch.as_tensor(QL, dtype=torch.float32)[None, :, None]
    return torch.maximum(tau * e, (tau - 1) * e).mean()


class CondDiffusion(nn.Module):
    """Conditional DDPM over the 2-h future (8 values), MLP noise predictor, cosine schedule,
    DDIM sampler with a few steps. `scale` rescales the target to unit variance."""

    def __init__(self, scale, T=50, hidden=128):
        super().__init__()
        self.T = T
        t = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        ab = f / f[0]
        betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
        self.register_buffer("ab", torch.cumprod(1 - betas, 0).float())
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("freq", torch.exp(-np.log(1000.0) * torch.arange(8) / 8).float())
        self.net = mlp([H + D.CTX_DIM + 16, hidden, hidden, hidden, H])

    def temb(self, t):
        a = t.float()[:, None] / self.T * 1000 * self.freq[None]
        return torch.cat([a.sin(), a.cos()], -1)

    def eps(self, yt, t, x):
        return self.net(torch.cat([yt, x, self.temb(t)], -1))

    def loss(self, x, y):
        y0 = y / self.scale
        t = torch.randint(0, self.T, (len(y0),))
        ab = self.ab[t][:, None]
        e = torch.randn_like(y0)
        return ((self.eps(ab.sqrt() * y0 + (1 - ab).sqrt() * e, t, x) - e) ** 2).mean()

    @torch.no_grad()
    def sample(self, x, n_samples=32, include_obs_noise=True, steps=15, eta=1.0):
        B = len(x)
        xr = x.unsqueeze(0).expand(n_samples, *x.shape).reshape(-1, x.shape[-1])
        y = torch.randn(len(xr), H)
        ts = torch.linspace(self.T - 1, 0, steps).round().long()
        for j, t in enumerate(ts):
            ab_t = self.ab[t]
            ab_p = self.ab[ts[j + 1]] if j + 1 < steps else torch.tensor(1.0)
            e = self.eps(y, torch.full((len(xr),), int(t)), xr)
            y0 = ((y - (1 - ab_t).sqrt() * e) / ab_t.sqrt()).clamp(-6, 6)
            sig = eta * ((1 - ab_p) / (1 - ab_t) * (1 - ab_t / ab_p)).clamp(min=0).sqrt()
            y = ab_p.sqrt() * y0 + (1 - ab_p - sig ** 2).clamp(min=0).sqrt() * e
            if j + 1 < steps:
                y = y + sig * torch.randn_like(y)
        return (y * self.scale).reshape(n_samples, B, H)


# ------------------------------------------------------------------ metrics

def interval_stats(lo, hi, y, T, alpha=0.10):
    cov = (y >= lo) & (y <= hi)
    score = (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
    out = {"coverage": float(cov.mean()), "width": float((hi - lo).mean()), "interval_score": float(score.mean())}
    for dt in ("T1DM", "T2DM"):
        mk = T == dt
        out[f"coverage_{dt}"] = float(cov[mk].mean())
        out[f"interval_score_{dt}"] = float(score[mk].mean())
    return out


def conformal_Q(lo_v, hi_v, y_v, level=0.9):
    sc = np.maximum(lo_v - y_v, y_v - hi_v)
    n = len(sc)
    return float(np.sort(sc)[min(n - 1, int(np.ceil((n + 1) * level)) - 1)])


def mondrian_Q(lo_v, hi_v, y_v, T_v, level=0.9):
    """Group-conditional (Mondrian) split conformal: one correction per diabetes type,
    computed from that type's validation windows only. Pooled conformal guarantees
    coverage on average over the test mix; Mondrian conformal guarantees it within each
    type, which matters when one type (here T1DM) is a small minority."""
    return {dt: conformal_Q(lo_v[T_v == dt], hi_v[T_v == dt], y_v[T_v == dt], level)
            for dt in ("T1DM", "T2DM") if (T_v == dt).any()}


def mondrian_stats(lo, hi, y, T, Qs, lo_v, hi_v, y_v, T_v):
    """Interval statistics after a per-type correction; falls back to the pooled
    correction for a type with no validation windows."""
    pooled = conformal_Q(lo_v, hi_v, y_v)
    q = np.array([Qs.get(t, pooled) for t in T])
    out = interval_stats(lo - q, hi + q, y, T)
    out.update({f"Q_{dt}": v for dt, v in Qs.items()})
    out["n_val_windows"] = {dt: int((T_v == dt).sum()) for dt in ("T1DM", "T2DM")}
    return out


def ece(prob, ev, bins=10):
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        mk = idx == b
        if mk.any():
            tot += mk.mean() * abs(prob[mk].mean() - ev[mk].mean())
    return float(tot)


def hypo_full(score, Y, prob=None):
    out = hypo_metrics(score, Y, prob)
    ev = (mg(Y)[:, :K60 + 1].min(1) < 70).astype(int)
    if "auroc" in out:
        fpr, tpr, _ = roc_curve(ev, score)
        out["sens_at_spec90"] = float(tpr[np.searchsorted(fpr, 0.10, side="right") - 1])
        if prob is not None:
            out["ece"] = ece(prob, ev)
            idx = np.minimum((prob * 10).astype(int), 9)
            out["reliability"] = [[float(prob[idx == b].mean()), float(ev[idx == b].mean()), int((idx == b).sum())]
                                  if (idx == b).any() else [None, None, 0] for b in range(10)]
    return out


def common(pt, Yte, Tte):
    y = mg(Yte)
    out = {"accuracy": errors(pt, Yte)}
    e = pt[:, K60] - y[:, K60]
    for dt in ("T1DM", "T2DM"):
        mk = Tte == dt
        out[f"rmse_60_{dt}"] = float(np.sqrt((e[mk] ** 2).mean()))
    for nm, mk in (("hypo", y[:, K60] < 70), ("eu", (y[:, K60] >= 70) & (y[:, K60] <= 180)), ("hyper", y[:, K60] > 180)):
        out[f"rmse_60_{nm}"] = float(np.sqrt((e[mk] ** 2).mean())) if mk.sum() else None
    return out


def summ_point(pt, Yte, Tte):
    out = common(pt, Yte, Tte)
    out["hypo"] = hypo_full(-pt[:, :K60 + 1].min(1), Yte)
    return out


def latent_samples(model, X, n=8):
    """Scenarios drawn from the latent variable alone (no per-step observation noise)."""
    with torch.no_grad():
        return mg(np.concatenate([model.sample(torch.as_tensor(X[b:b + 4096]), n_samples=n,
                                               include_obs_noise=False).numpy() for b in range(0, len(X), 4096)], 1))


def summ_samples(model, Xva, Yva, Xte, Yte, Tte, n, seed, latent=False, Tva=None):
    """`seed` fixes the random subset of test windows used by the realism metrics, so every model is scored on the same windows."""
    st = samples_of(model, Xte, n)
    sv = samples_of(model, Xva, 32)
    y = mg(Yte)
    out = common(st.mean(0), Yte, Tte)
    out["prob"] = prob_metrics(st, Yte)
    c = crps(st, y)[:, K60]
    for dt in ("T1DM", "T2DM"):
        out["prob"][f"crps_60_{dt}"] = float(c[Tte == dt].mean())
    lo, hi = np.quantile(st[:, :, K60], 0.05, 0), np.quantile(st[:, :, K60], 0.95, 0)
    out["interval_raw"] = interval_stats(lo, hi, y[:, K60], Tte)
    lo_v, hi_v = np.quantile(sv[:, :, K60], 0.05, 0), np.quantile(sv[:, :, K60], 0.95, 0)
    Q = conformal_Q(lo_v, hi_v, mg(Yva)[:, K60])
    out["interval_conf"] = interval_stats(lo - Q, hi + Q, y[:, K60], Tte)
    out["interval_conf"]["Q"] = Q
    if Tva is not None:
        out["interval_mondrian"] = mondrian_stats(lo, hi, y[:, K60], Tte,
                                                  mondrian_Q(lo_v, hi_v, mg(Yva)[:, K60], Tva),
                                                  lo_v, hi_v, mg(Yva)[:, K60], Tva)
    out["calibration"] = {str(a): float(((y[:, K60] >= np.quantile(st[:, :, K60], (1 - a) / 2, 0)) &
                                         (y[:, K60] <= np.quantile(st[:, :, K60], 1 - (1 - a) / 2, 0))).mean())
                          for a in (0.5, 0.8, 0.9, 0.95)}
    pmin = (st[:, :, :K60 + 1].min(2) < 70).mean(0)
    out["hypo"] = hypo_full(pmin, Yte, prob=pmin)
    out["realism"] = realism(st, Yte, np.random.default_rng(seed))
    if latent:
        sl = latent_samples(model, Xte)
        out["realism_latent"] = realism(sl, Yte, np.random.default_rng(seed))
        out["prob_latent"] = prob_metrics(sl, Yte)
    return out


def summ_quantile(net, Xva, Yva, Xte, Yte, Tte, Tva=None):
    def q(X):
        with torch.no_grad():
            return mg(np.sort(net(torch.as_tensor(X)).numpy(), 1))  # (N,Q,H)
    qt, qv = q(Xte), q(Xva)
    y = mg(Yte)
    out = common(qt[:, 9, :], Yte, Tte)
    lo, hi = qt[:, 0, K60], qt[:, -1, K60]
    out["interval_raw"] = interval_stats(lo, hi, y[:, K60], Tte)
    Q = conformal_Q(qv[:, 0, K60], qv[:, -1, K60], mg(Yva)[:, K60])
    out["interval_conf"] = interval_stats(lo - Q, hi + Q, y[:, K60], Tte)
    out["interval_conf"]["Q"] = Q
    if Tva is not None:
        out["interval_mondrian"] = mondrian_stats(lo, hi, y[:, K60], Tte,
                                                  mondrian_Q(qv[:, 0, K60], qv[:, -1, K60], mg(Yva)[:, K60], Tva),
                                                  qv[:, 0, K60], qv[:, -1, K60], mg(Yva)[:, K60], Tva)
    out["calibration"] = {str(a): float(((y[:, K60] >= qt[:, int(round((((1 - a) / 2) - 0.05) / 0.05)), K60]) &
                                         (y[:, K60] <= qt[:, int(round(((1 - (1 - a) / 2) - 0.05) / 0.05)), K60])).mean())
                          for a in (0.5, 0.8, 0.9)}
    # marginal P(glucose < 70) at each step from the quantile function, worst step
    p = np.zeros(len(qt))
    for k in range(K60 + 1):
        pk = np.array([np.interp(70.0, qt[i, :, k], QL, left=0.0, right=1.0) for i in range(len(qt))])
        p = np.maximum(p, pk)
    out["hypo"] = hypo_full(p, Yte)
    return out


# ------------------------------------------------------------------ ablations

MASKS = {"no_insulin": list(range(K, 2 * K)) + list(range(3 * K, 3 * K + H)),
         "no_meals": list(range(2 * K, 3 * K)) + list(range(3 * K + H, 3 * K + 2 * H)),
         "no_planned": list(range(3 * K, 3 * K + 2 * H)),
         "glucose_only": list(range(K, 3 * K + 2 * H))}


def masked(X, name):
    X = X.copy(); X[:, MASKS[name]] = 0.0; return X


def hybrid_fit(Xtr, Ytr, beta, E, seed, ridge_lam=5.0):
    ridge = RidgeAR(lam=ridge_lam).fit(Xtr, Ytr)
    torch.manual_seed(seed)
    cv = train_cvae(CVAETwin(z_dim=16), Xtr, (Ytr - ridge.predict(Xtr)).astype(np.float32), epochs=E, seed=seed, beta=beta)
    return HybridTwin(ridge, cv)


def quick_eval(model, Xte, Yte, n=32):
    s = samples_of(model, Xte, n)
    p = prob_metrics(s, Yte)
    pmin = (s[:, :, :K60 + 1].min(2) < 70).mean(0)
    h = hypo_metrics(pmin, Yte, prob=pmin)
    return dict(rmse_60=errors(s.mean(0), Yte)["rmse_60"], crps_60=p["crps_60"], auroc=h.get("auroc"))


# ------------------------------------------------------------------ one seed

def load_betas(seed, strict=True):
    """KL weights chosen on validation patients in the core run. Looked up by seed, not by list position.
    With strict=False (quick runs, tests) a missing result file falls back to 0.5, the value chosen in every split."""
    try:
        runs = json.load(open(ROOT / "results" / "paper1_results.json"))["runs"]
        hyb = json.load(open(ROOT / "results" / "paper1_hybrid.json"))["runs"]
        return {"twin": next(r["twin_beta"] for r in runs if r["seed"] == seed),
                "hybrid": next(r["beta"] for r in hyb if r["seed"] == seed)}
    except (OSError, KeyError, StopIteration):
        if strict:
            raise SystemExit("results/paper1_results.json and paper1_hybrid.json are needed for the KL weights "
                             "(run p1_experiments.py and p1_hybrid.py first, or pass --quick)")
        return {"twin": 0.5, "hybrid": 0.5}


def run_seed(recs, seed, cfg):
    t0 = time.time(); rng = np.random.default_rng(seed)
    sp = split(recs, seed)
    Xtr, Ytr, _ = pack(sp["train"], cfg["stride"]); Xva, Yva, Tva = pack(sp["calib"], cfg["stride"])
    Xte, Yte, Tte = pack(sp["test"], cfg["stride"])
    E, beta = cfg["epochs"], load_betas(seed, strict=not cfg.get("quick", False))
    res = dict(seed=seed, betas=beta, models={}, time={},
               n_patients={k: len({r["dtype"] + r["pid"] for r in v}) for k, v in sp.items()},
               n_windows=dict(train=len(Xtr), val=len(Xva), test=len(Xte)))

    def done(name, t):
        res["time"][name] = round(time.time() - t, 1)
        print(f"  seed {seed} {name:18s} {res['time'][name]:6.1f}s", flush=True)

    t = time.time(); ridge = RidgeAR(lam=5.0).fit(Xtr, Ytr)
    res["models"]["ridge"] = summ_point(mg(ridge.predict(Xte)), Yte, Tte); done("ridge", t)

    t = time.time(); torch.manual_seed(seed)
    gru = train_point(GRUForecaster(), Xtr, Ytr, epochs=E, seed=seed)
    with torch.no_grad():
        res["models"]["gru"] = summ_point(mg(gru(torch.as_tensor(Xte)).numpy()), Yte, Tte)
    done("gru", t)

    t = time.time(); torch.manual_seed(seed)
    gm = train_gauss(GaussMLP(), Xtr, Ytr, E, seed)
    res["models"]["gauss_mlp"] = summ_samples(gm, Xva, Yva, Xte, Yte, Tte, cfg["n_samples"], seed, Tva=Tva); done("gauss_mlp", t)

    t = time.time(); torch.manual_seed(seed)
    tf = fit(TransformerGauss(), Xtr, Ytr, gauss_loss, E, seed, lr=1e-3)
    res["models"]["transformer"] = summ_samples(tf, Xva, Yva, Xte, Yte, Tte, cfg["n_samples"], seed, Tva=Tva); done("transformer", t)

    t = time.time(); torch.manual_seed(seed)
    qn = fit(QuantileNet(), Xtr, Ytr, pinball, E, seed)
    res["models"]["cqr"] = summ_quantile(qn, Xva, Yva, Xte, Yte, Tte, Tva=Tva); done("cqr", t)

    t = time.time(); torch.manual_seed(seed)
    tw = P.train_cvae(CVAETwin(z_dim=16), Xtr, Ytr, epochs=E, seed=seed, beta=beta["twin"])
    res["models"]["twin"] = summ_samples(tw, Xva, Yva, Xte, Yte, Tte, cfg["n_samples"], seed, latent=True, Tva=Tva); done("twin", t)

    t = time.time()
    hy = hybrid_fit(Xtr, Ytr, beta["hybrid"], E, seed)
    res["models"]["hybrid"] = summ_samples(hy, Xva, Yva, Xte, Yte, Tte, cfg["n_samples"], seed, latent=True, Tva=Tva); done("hybrid", t)

    t = time.time(); torch.manual_seed(seed)
    df = fit(CondDiffusion(Ytr.std(0)), Xtr, Ytr, lambda m, x, y: m.loss(x, y), cfg["diff_epochs"], seed)
    res["models"]["diffusion"] = summ_samples(df, Xva, Yva, Xte, Yte, Tte, cfg["n_samples"], seed, Tva=Tva); done("diffusion", t)

    t = time.time(); torch.manual_seed(seed)
    R = (Ytr - hy.ridge.predict(Xtr)).astype(np.float32)
    dh = fit(CondDiffusion(R.std(0)), Xtr, R, lambda m, x, y: m.loss(x, y), cfg["diff_epochs"], seed)
    res["models"]["hybrid_diffusion"] = summ_samples(HybridTwin(hy.ridge, dh), Xva, Yva, Xte, Yte, Tte,
                                                     cfg["n_samples"], seed, Tva=Tva); done("hybrid_diffusion", t)

    # base x generator factorial: Transformer mean with a residual generator
    base = MeanBase(tf)
    R2 = (Ytr - base.predict(Xtr)).astype(np.float32)
    t = time.time(); torch.manual_seed(seed)
    tc = train_cvae(CVAETwin(z_dim=16), Xtr, R2, epochs=E, seed=seed, beta=beta["hybrid"])
    res["models"]["tf_cvae"] = summ_samples(HybridTwin(base, tc), Xva, Yva, Xte, Yte, Tte, cfg["n_samples"], seed, latent=True, Tva=Tva); done("tf_cvae", t)
    t = time.time(); torch.manual_seed(seed)
    td = fit(CondDiffusion(R2.std(0)), Xtr, R2, lambda m, x, y: m.loss(x, y), cfg["diff_epochs"], seed)
    res["models"]["tf_diffusion"] = summ_samples(HybridTwin(base, td), Xva, Yva, Xte, Yte, Tte, cfg["n_samples"], seed, Tva=Tva); done("tf_diffusion", t)

    if seed == 0:   # a few test windows for the scenario figure (same windows as the main study)
        yte, pick = mg(Yte), []
        for dt in ("T1DM", "T2DM"):
            idx = np.where(Tte == dt)[0]
            var = yte[idx].std(1)
            pick += [int(i) for i in idx[np.argsort(var)[-int(0.25 * len(idx))::max(1, int(0.25 * len(idx)) // 2)][:2]]]
        Xp = torch.as_tensor(Xte[pick])
        ex = dict(dtype=[str(Tte[i]) for i in pick], truth=yte[pick].tolist(),
                  history=[(Xte[i, :K] * D.G_SCALE + D.G_MEAN).tolist() for i in pick])
        for name, mdl, kw in (("transformer", tf, {}), ("hybrid", hy, dict(include_obs_noise=False)), ("diffusion", df, {}),
                              ("hybrid_diffusion", HybridTwin(hy.ridge, dh), {}), ("tf_diffusion", HybridTwin(base, td), {})):
            with torch.no_grad():
                ex[name] = mg(mdl.sample(Xp, n_samples=20, **kw).numpy()).transpose(1, 0, 2).tolist()
        res["examples"] = ex

    # ---- A3 input ablations of the hybrid twin
    t = time.time(); abl = {"full": quick_eval(hy, Xte, Yte)}
    abl["full"]["ridge_rmse_60"] = res["models"]["ridge"]["accuracy"]["rmse_60"]
    for name in MASKS:
        m = hybrid_fit(masked(Xtr, name), Ytr, beta["hybrid"], E, seed)
        abl[name] = quick_eval(m, masked(Xte, name), Yte)
        r2 = RidgeAR(lam=5.0).fit(masked(Xtr, name), Ytr)
        abl[name]["ridge_rmse_60"] = errors(mg(r2.predict(masked(Xte, name))), Yte)["rmse_60"]
    res["ablation"] = abl; done("ablation", t)

    # ---- A4 training-set size (patients)
    t = time.time(); size = {}
    rr = np.random.default_rng(seed + 99)
    pats = {dt: sorted({r["pid"] for r in sp["train"] if r["dtype"] == dt}) for dt in ("T1DM", "T2DM")}
    for dt in pats:
        rr.shuffle(pats[dt])
    for frac in cfg["fractions"]:
        keep = {dt: set(pats[dt][:max(1, int(round(frac * len(pats[dt]))))]) for dt in pats}
        sub = [r for r in sp["train"] if r["pid"] in keep[r["dtype"]]]
        Xs, Ys, _ = pack(sub, cfg["stride"])
        rg = RidgeAR(lam=5.0).fit(Xs, Ys)
        h2 = hybrid_fit(Xs, Ys, beta["hybrid"], E, seed)
        with torch.no_grad():
            size[str(frac)] = dict(n_patients=sum(len(v) for v in keep.values()), n_windows=len(Xs),
                                   ridge_rmse_60=errors(mg(rg.predict(Xte)), Yte)["rmse_60"],
                                   hybrid=quick_eval(h2, Xte, Yte))
    res["train_size"] = size; done("train_size", t)
    res["seconds"] = round(time.time() - t0)
    return res


CFG = dict(epochs=25, diff_epochs=40, stride=2, n_samples=64, fractions=(0.25, 0.5))
QUICK = dict(epochs=2, diff_epochs=2, stride=6, n_samples=16, fractions=(0.5,), quick=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-7", help="e.g. 0-3 or 4-7")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "paper1_advanced.json"))
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    cfg = QUICK if a.quick else CFG
    recs = load_all(str(ROOT / "data"))
    runs = []
    for s in range(lo, hi + 1):
        runs.append(run_seed(recs, s, cfg))
        json.dump(dict(config=dict((k, list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()), runs=runs),
                  open(a.out, "w"), indent=1)
        print(f"seed {s} total {runs[-1]['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
