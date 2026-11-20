"""
Hybrid residual twin for Paper 1: ridge regression supplies the mean trajectory and a
CVAE models the residual around it, so scenarios = ridge forecast + CVAE residual draws.
Same splits, seeds and metrics as p1_experiments.py (P1-P4 and per-patient adaptation).
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import p1_experiments as P
from p1_experiments import (D, CVAETwin, RidgeAR, train_cvae, mg, errors, prob_metrics, hypo_metrics,
                            realism, pack, split, windows, load_all, STEP, ROOT)


class HybridTwin:
    def __init__(self, ridge, cvae):
        self.ridge, self.cvae = ridge, cvae

    def base(self, x):
        return torch.as_tensor(self.ridge.predict(np.asarray(x)), dtype=torch.float32)

    @torch.no_grad()
    def sample(self, x, n_samples=32, include_obs_noise=True):
        x = torch.as_tensor(x)
        return self.base(x).unsqueeze(0) + self.cvae.sample(x, n_samples=n_samples, include_obs_noise=include_obs_noise)


def samples(model, X, n):
    return mg(np.concatenate([model.sample(X[b:b + 4096], n_samples=n).numpy() for b in range(0, len(X), 4096)], 1))


def run_seed(recs, seed, cfg):
    t0 = time.time(); rng = np.random.default_rng(seed)
    sp = split(recs, seed)
    Xtr, Ytr, _ = pack(sp["train"], cfg["stride"]); Xva, Yva, _ = pack(sp["calib"], cfg["stride"])
    Xte, Yte, Tte = pack(sp["test"], cfg["stride"])
    ridge = RidgeAR(lam=5.0).fit(Xtr, Ytr)
    Rtr = (Ytr - ridge.predict(Xtr)).astype(np.float32)
    best, bc = None, np.inf
    for beta in cfg["betas"]:
        torch.manual_seed(seed)
        cv = train_cvae(CVAETwin(z_dim=16), Xtr, Rtr, epochs=cfg["epochs"], seed=seed, beta=beta)
        h = HybridTwin(ridge, cv)
        v = prob_metrics(samples(h, Xva, 32), Yva)["crps_all"]
        if v < bc:
            best, bc, bb = h, v, beta
    res = dict(seed=seed, beta=bb)
    s = samples(best, Xte, cfg["n_samples"])
    res["accuracy"] = errors(s.mean(0), Yte)
    pr = prob_metrics(s, Yte)
    k60 = STEP[60]; yv = mg(Yva)[:, k60]; yt = mg(Yte)[:, k60]
    sv = samples(best, Xva, cfg["n_samples"])[:, :, k60]
    lv, hv = np.quantile(sv, 0.05, 0), np.quantile(sv, 0.95, 0)
    sc = np.maximum(lv - yv, yv - hv); n = len(sc); Q = float(np.sort(sc)[min(n - 1, int(np.ceil((n + 1) * 0.9)) - 1)])
    lt, ht = np.quantile(s[:, :, k60], 0.05, 0) - Q, np.quantile(s[:, :, k60], 0.95, 0) + Q
    pr.update(conf_cov90_60=float(((yt >= lt) & (yt <= ht)).mean()), conf_width90_60=float((ht - lt).mean()), conf_Q=Q)
    for dt in ("T1DM", "T2DM"):
        mk = Tte == dt
        pr[f"conf_cov90_60_{dt}"] = float(((yt[mk] >= lt[mk]) & (yt[mk] <= ht[mk])).mean())
    res["probabilistic"] = pr
    pmin = (s[:, :, :k60 + 1].min(2) < 70).mean(0)
    res["hypo"] = hypo_metrics(pmin, Yte, prob=pmin)
    res["realism"] = realism(s, Yte, rng)
    with torch.no_grad():
        sl = mg(np.concatenate([best.sample(Xte[b:b + 4096], n_samples=8, include_obs_noise=False).numpy()
                                for b in range(0, len(Xte), 4096)], 1))
    res["realism_latent"] = realism(sl, Yte, rng)
    res["prob_latent"] = prob_metrics(sl, Yte)
    # per-patient adaptation of the residual model (decoder output layer), as in P5
    pers = {"population": [], **{f"{d}d": [] for d in cfg["adapt_days"]}}
    for r in sp["test"]:
        n_ = len(r["glucose"]); cut = int(0.5 * n_)
        ev = dict(r, glucose=r["glucose"][cut:], insulin=r["insulin"][cut:], carbs=r["carbs"][cut:])
        Xe, Ye = windows(ev, cfg["stride"])
        if len(Xe) < 20:
            continue
        pers["population"].append(errors(samples(best, Xe, 48).mean(0), Ye)["rmse_60"])
        for d in cfg["adapt_days"]:
            me = min(d * 96, cut)
            ad = dict(r, glucose=r["glucose"][:me], insulin=r["insulin"][:me], carbs=r["carbs"][:me])
            Xa, Ya = windows(ad, 1)
            if len(Xa) < 10:
                pers[f"{d}d"].append(np.nan); continue
            cv = copy.deepcopy(best.cvae)
            for prm in cv.enc.parameters(): prm.requires_grad_(False)
            for layer in list(cv.dec)[:-1]:
                for prm in layer.parameters(): prm.requires_grad_(False)
            train_cvae(cv, Xa, (Ya - ridge.predict(Xa)).astype(np.float32), epochs=cfg["adapt_epochs"], seed=seed, lr=5e-4, beta=bb)
            pers[f"{d}d"].append(errors(samples(HybridTwin(ridge, cv), Xe, 48).mean(0), Ye)["rmse_60"])
    res["personalisation"] = {k: float(np.nanmean(v)) for k, v in pers.items()}
    res["personalisation_per_record"] = {k: [float(x) for x in v] for k, v in pers.items()}
    res["seconds"] = round(time.time() - t0)
    print(f"  hybrid seed {seed} done in {res['seconds']}s (beta={bb})", flush=True)
    return res


CFG = dict(epochs=25, stride=2, betas=(0.1, 0.5), n_samples=64, adapt_days=(1, 2, 4), adapt_epochs=10)
QUICK = dict(epochs=2, stride=6, betas=(0.1,), n_samples=16, adapt_days=(1,), adapt_epochs=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "paper1_hybrid.json"))
    a = ap.parse_args(); cfg = QUICK if a.quick else CFG
    recs = load_all(str(ROOT / "data"))
    runs = []
    for s in range(a.seeds):
        runs.append(run_seed(recs, s, cfg))
        json.dump(dict(config={k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}, runs=runs),
                  open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
