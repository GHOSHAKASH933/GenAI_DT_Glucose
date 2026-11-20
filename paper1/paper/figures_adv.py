"""Figures for the extended benchmark, drawn from results/paper1_advanced.json."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESDIR = ROOT / "results"
BUILD = HERE / "build"
sys.path.insert(0, str(HERE))
from common import boot_ci
from adv_tables import LABEL, ORDER, SAMPLERS, ABL

SRC = sys.argv[1] if len(sys.argv) > 1 else str(RESDIR / "paper1_advanced.json")
A = json.load(open(SRC))["runs"]
OUT = BUILD / "figures"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "savefig.dpi": 300})
COL = {"ridge": "#4c72b0", "gru": "#8172b2", "gauss_mlp": "#dd8452", "transformer": "#937860", "cqr": "#8c8c8c",
       "twin": "#c44e52", "diffusion": "#2a7ab0", "hybrid": "#7a1f3d", "hybrid_diffusion": "#1b9e77",
       "tf_cvae": "#e377c2", "tf_diffusion": "#0b5d40"}
SHORT = {"gauss_mlp": "Gaussian MLP", "transformer": "Transformer", "twin": "CVAE", "diffusion": "Diffusion",
         "hybrid": "Ridge+CVAE", "hybrid_diffusion": "Ridge+diff.", "tf_cvae": "Transf.+CVAE", "tf_diffusion": "Transf.+diff."}


def vals(model, *path, src=None):
    out = []
    for r in (src or A):
        v = r["models"][model]
        for p in path:
            v = v[p]
        out.append(v)
    return np.array(out, float)


def ci(v):
    lo, hi = boot_ci(v); mu = np.mean(v); return mu, mu - lo, hi - mu


def tradeoff():
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for k in SAMPLERS:
        x = vals(k, "accuracy", "rmse_60"); y = vals(k, "realism", "energy")
        mx, ex1, ex2 = ci(x); my, ey1, ey2 = ci(y)
        ax.errorbar(mx, my, xerr=[[ex1], [ex2]], yerr=[[ey1], [ey2]], fmt="o", color=COL[k], ms=5.5, lw=0.8, capsize=1.5, label=SHORT[k])
        if "realism_latent" in A[0]["models"][k]:
            yl = vals(k, "realism_latent", "energy"); ml = ci(yl)[0]
            ax.plot([mx, mx], [my, ml], color=COL[k], lw=0.7, ls=":")
            ax.plot(mx, ml, marker="s", mfc="white", color=COL[k], ms=5.5, ls="none")
    ax.plot([], [], "s", mfc="white", color="k", ms=5.5, ls="none", label="latent-only draw\n(CVAE models)")
    ax.set_xlabel("60-minute RMSE (mg/dL)"); ax.set_ylabel("Energy distance to observed trajectories")
    h, l = ax.get_legend_handles_labels()
    order = [i for i, x in enumerate(l) if not x.startswith("latent")] + [i for i, x in enumerate(l) if x.startswith("latent")]
    ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=7, loc="center left",
              bbox_to_anchor=(1.01, 0.5), title="circles: full predictive draw", title_fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig6_tradeoff.png", bbox_inches="tight"); plt.close(fig)


def calibration():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    lv = ["0.5", "0.8", "0.9", "0.95"]
    ax[0].plot([0.45, 1], [0.45, 1], color="#bbb", lw=0.8)
    for k in SAMPLERS:
        y = [np.mean([r["models"][k]["calibration"][l] for r in A]) for l in lv]
        ax[0].plot([float(l) for l in lv], y, "o-", color=COL[k], ms=3, lw=1.1, label=SHORT[k])
    y = [np.mean([r["models"]["cqr"]["calibration"][l] for r in A]) for l in lv[:3]]
    ax[0].plot([0.5, 0.8, 0.9], y, "o--", color=COL["cqr"], ms=3, lw=1.1, label="Quantile net")
    ax[0].set_xlabel("Nominal level of central interval"); ax[0].set_ylabel("Empirical coverage")
    ax[0].set_title("(a) Intervals at 60 min, before correction", loc="left", fontsize=8.5)
    ax[0].legend(frameon=False, fontsize=6.3, ncol=2)
    ax[1].plot([0, 1], [0, 1], color="#bbb", lw=0.8)
    for k in SAMPLERS:
        tot = np.zeros((10, 3))
        for r in A:
            for b, (p, e, n) in enumerate(r["models"][k]["hypo"]["reliability"]):
                if n:
                    tot[b] += [p * n, e * n, n]
        ok = tot[:, 2] > 30
        ax[1].plot(tot[ok, 0] / tot[ok, 2], tot[ok, 1] / tot[ok, 2], "o-", color=COL[k], ms=3, lw=1.1)
    ax[1].set_xlabel("Predicted probability of glucose < 70 mg/dL"); ax[1].set_ylabel("Observed frequency")
    ax[1].set_title("(b) Hypoglycaemia probability within 60 min", loc="left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "fig7_calibration.png", bbox_inches="tight"); plt.close(fig)


def scenarios():
    ex = next(r["examples"] for r in A if "examples" in r)
    keys = ["transformer", "hybrid", "diffusion", "hybrid_diffusion", "tf_diffusion"]
    names = {"transformer": "Transformer,\nGaussian head", "hybrid": "Ridge + CVAE\n(latent only)", "diffusion": "Diffusion",
             "hybrid_diffusion": "Ridge + diffusion", "tf_diffusion": "Transformer\n+ diffusion"}
    n = len(ex["truth"]); fig, ax = plt.subplots(len(keys), n, figsize=(7.2, 7.2), sharey="col")
    th = np.arange(-7, 1) * 15; tf = np.arange(1, 9) * 15
    for i, k in enumerate(keys):
        for j in range(n):
            a = ax[i, j]
            for s in ex[k][j]:
                a.plot(tf, s, color=COL[k], alpha=0.35, lw=0.8)
            a.plot(th, ex["history"][j], color="k", lw=1.2); a.plot(tf, ex["truth"][j], color="k", lw=1.2, ls="--")
            a.axhline(70, color="#999", lw=0.6, ls=":"); a.axvline(0, color="#ccc", lw=0.6)
            if i == 0: a.set_title(ex["dtype"][j], fontsize=8)
            if j == 0: a.set_ylabel(f"{names[k]}\nglucose (mg/dL)", fontsize=7)
            if i == len(keys) - 1: a.set_xlabel("min", fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "fig8_scenarios_adv.png", bbox_inches="tight"); plt.close(fig)


def ablation_size():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    labs = [l for _, l in ABL]; keys = [k for k, _ in ABL]
    short = ["All inputs", "No insulin", "No meals", "No planned\ninputs", "Glucose\nonly"]
    for j, (f, col, lab) in enumerate((("rmse_60", COL["hybrid"], "Hybrid twin"), ("ridge_rmse_60", COL["ridge"], "Ridge"))):
        for i, k in enumerate(keys):
            mu, a, b = ci(np.array([r["ablation"][k][f] for r in A]))
            ax[0].bar(i + (j - 0.5) * 0.38, mu, 0.36, yerr=[[a], [b]], color=col, capsize=1.5, label=lab if i == 0 else None)
    ax[0].set_xticks(range(len(keys))); ax[0].set_xticklabels(short, fontsize=7)
    lo = min(np.mean([r["ablation"][k]["rmse_60"] for r in A]) for k in keys)
    ax[0].set_ylim(lo - 4, None); ax[0].set_ylabel("60-minute RMSE (mg/dL)"); ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title("(a) Input ablation", loc="left", fontsize=8.5)
    fr = [float(f) for f in A[0]["train_size"]]
    npat = np.mean([r["n_patients"]["train"] for r in A]); xs = [f * npat for f in fr] + [npat]
    for lab, col, get, full in (("Hybrid twin", COL["hybrid"], lambda r, f: r["train_size"][str(f)]["hybrid"]["rmse_60"],
                                  lambda r: r["ablation"]["full"]["rmse_60"]),
                                ("Ridge", COL["ridge"], lambda r, f: r["train_size"][str(f)]["ridge_rmse_60"],
                                 lambda r: r["models"]["ridge"]["accuracy"]["rmse_60"])):
        pts = [np.array([get(r, f) for r in A]) for f in fr] + [np.array([full(r) for r in A])]
        mu, lo, hi = zip(*[ci(p) for p in pts])
        ax[1].errorbar(xs, mu, yerr=[lo, hi], fmt="o-", color=col, label=lab, capsize=2, ms=4, lw=1.2)
    ax[1].set_xlabel("Training patients"); ax[1].set_ylabel("60-minute RMSE (mg/dL)"); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title("(b) Training-set size", loc="left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "fig9_ablation_size.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    tradeoff(); calibration(); scenarios(); ablation_size()
    print("advanced figures ->", OUT)
