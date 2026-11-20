"""Figures for Paper 1, drawn from results/paper1_results.json."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESDIR = ROOT / "results"
BUILD = HERE / "build"
sys.path.insert(0, str(HERE))
from common import boot_ci

HY = json.load(open(str(RESDIR / "paper1_hybrid.json")))["runs"]
RES = json.load(open(sys.argv[1] if len(sys.argv) > 1 else str(RESDIR / "paper1_results.json")))["runs"]
OUT = BUILD / "figures"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "savefig.dpi": 300})
COL = {"persistence": "#9e9e9e", "ridge": "#4c72b0", "mlp": "#55a868", "gru": "#8172b2",
       "gauss_mlp": "#dd8452", "twin": "#c44e52"}
NAME = {"persistence": "Persistence", "ridge": "Ridge", "mlp": "MLP", "gru": "GRU",
        "gauss_mlp": "Gaussian MLP", "twin": "CVAE twin"}


def ci(v):
    lo, hi = boot_ci(v); mu = np.mean(v); return mu, mu - lo, hi - mu


# ---------------------------------------------------------- architecture
def arch():
    fig, ax = plt.subplots(figsize=(7.2, 2.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)
    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=c, ec="#333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)
    def arr(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9, color="#333"))
    box(0.1, 2.2, 1.9, 0.9, "CGM, insulin\nand meal records", "#eef3fb")
    box(0.1, 0.4, 1.9, 0.9, "Federated clients\n(T1DM, T2DM-1..3)", "#eef3fb")
    box(2.6, 1.3, 2.2, 1.0, "Generative twin\nCVAE  $p_\\theta(y\\mid x,z)$", "#fbeaea")
    box(5.4, 2.3, 2.0, 0.85, "Scenarios and\nforecast intervals", "#fff5e6")
    box(5.4, 1.25, 2.0, 0.85, "Hypoglycaemia\nrisk score", "#fff5e6")
    box(5.4, 0.2, 2.0, 0.85, "Per-patient\nadaptation", "#fff5e6")
    box(8.0, 1.25, 1.9, 0.85, "Edge model\n(int8 / distilled)", "#eaf6ec")
    arr(2.0, 2.65, 2.6, 2.0); arr(2.0, 0.85, 2.6, 1.6)
    for y in (2.72, 1.67, 0.62):
        arr(4.8, 1.8, 5.4, y)
    arr(7.4, 2.72, 8.0, 1.9); arr(7.4, 1.67, 8.0, 1.67)
    ax.text(5.0, 3.32, "Evaluated on held-out patients: accuracy, CRPS, coverage, AUROC, realism, latency",
            ha="center", fontsize=7.5, style="italic")
    fig.savefig(OUT / "fig1_pipeline.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------- horizon accuracy
def horizon():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    hs = [30, 60, 120]
    for k in COL:
        mu = [np.mean([r["accuracy"][k][f"rmse_{h}"] for r in RES]) for h in hs]
        ax[0].plot(hs, mu, "o-", color=COL[k], label=NAME[k], lw=1.4, ms=4)
    ax[0].plot(hs, [np.mean([r["accuracy"][f"rmse_{x}"] for r in HY]) for x in hs], "s-", color="#7a1f3d", label="Hybrid twin", lw=1.6, ms=4)
    ax[0].set_xticks(hs); ax[0].set_xlabel("Forecast horizon (min)"); ax[0].set_ylabel("RMSE (mg/dL)")
    ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("(a) Error by horizon", loc="left")
    ks = list(COL)
    for j, k in enumerate(ks):
        mu, a, b = ci([r["accuracy"][k]["ceg_A_60"] for r in RES])
        ax[1].bar(j, mu, yerr=[[a], [b]], color=COL[k], capsize=2)
    mu, a, b = ci([r["accuracy"]["ceg_A_60"] for r in HY]); ax[1].bar(len(ks), mu, yerr=[[a], [b]], color="#7a1f3d", capsize=2)
    ax[1].set_xticks(range(len(ks) + 1)); ax[1].set_xticklabels([NAME[k] for k in ks] + ["Hybrid twin"], rotation=35, ha="right")
    ax[1].set_ylabel("Clarke zone A at 60 min (%)"); lo = min(np.mean([r["accuracy"][k]["ceg_A_60"] for r in RES]) for k in ks)
    ax[1].set_ylim(max(0, lo - 10), 100); ax[1].set_title("(b) Clinical accuracy", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig2_accuracy.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------- scenarios
def scenarios():
    ex = json.load(open(str(RESDIR / "paper1_examples.json")))
    n = len(ex["truth"]); fig, ax = plt.subplots(3, n, figsize=(7.2, 5.0), sharey="col")
    th = np.arange(-7, 1) * 15; tf = np.arange(1, 9) * 15
    spec = ((0, "hybrid", "Hybrid twin", "#7a1f3d"), (1, "twin", "CVAE twin", COL["twin"]),
            (2, "gauss", "Gaussian MLP", COL["gauss_mlp"]))
    for j in range(n):
        for row, key, lab, col in spec:
            a = ax[row, j]
            for smp in ex[key][j]:
                a.plot(tf, smp, color=col, alpha=0.35, lw=0.8)
            a.plot(th, ex["history"][j], color="k", lw=1.2)
            a.plot(tf, ex["truth"][j], color="k", lw=1.2, ls="--")
            a.axhline(70, color="#999", lw=0.6, ls=":"); a.axvline(0, color="#ccc", lw=0.6)
            if row == 0: a.set_title(ex["dtype"][j], fontsize=8)
            if j == 0: a.set_ylabel(f"{lab}\nglucose (mg/dL)", fontsize=7.5)
            if row == 2: a.set_xlabel("min", fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "fig3_scenarios.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------- personalisation
def personalisation():
    days = [1, 2, 4]; fig, ax = plt.subplots(figsize=(4.2, 2.8))
    pop = [r["personalisation"]["population"] for r in RES]
    ax.axhline(np.mean(pop), color="k", ls="--", lw=1, label="Population twin")
    for pre, lab, c in (("", "Fine-tune output layer", COL["twin"]), ("dec_", "Fine-tune decoder", "#8c2d04"),
                        ("scratch_", "Train from scratch", COL["persistence"])):
        mu, lo, hi = zip(*[ci([r["personalisation"][f"{pre}{d}d"] for r in RES]) for d in days])
        ax.errorbar(days, mu, yerr=[lo, hi], fmt="o-", color=c, label=lab, capsize=2, ms=4, lw=1.3)
    mu, lo, hi = zip(*[ci([r["personalisation"][f"{d}d"] for r in HY]) for d in days])
    ax.errorbar(days, mu, yerr=[lo, hi], fmt="s-", color="#7a1f3d", label="Hybrid twin, fine-tuned", capsize=2, ms=4, lw=1.3)
    ax.axhline(np.mean([r["personalisation"]["population"] for r in HY]), color="#7a1f3d", ls=":", lw=1, label="Hybrid population twin")
    ax.set_xticks(days); ax.set_xlabel("Days of the new patient's data"); ax.set_ylabel("60-min RMSE (mg/dL)")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig4_personalisation.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------- federated
def federated():
    regs = [("local_only", "Local only", "#9e9e9e"), ("fedavg", "FedAvg", "#4c72b0"),
            ("fedavg_personalised", "FedAvg + fine-tune", "#55a868"), ("centralised", "Centralised", "#c44e52")]
    sites = ["T1DM", "T2DM-1", "T2DM-2", "T2DM-3"]
    fig, ax = plt.subplots(figsize=(6.0, 2.8)); w = 0.2
    for i, (k, lab, c) in enumerate(regs):
        for j, s in enumerate(sites):
            v = [r["federated"][k][s] for r in RES if s in r["federated"][k]]
            mu, a, b = ci(v)
            ax.bar(j + (i - 1.5) * w, mu, w, yerr=[[a], [b]], color=c, capsize=1.5, label=lab if j == 0 else None)
    ax.set_xticks(range(4)); ax.set_xticklabels(["T1DM client", "T2DM client 1", "T2DM client 2", "T2DM client 3"])
    ax.set_ylabel("60-min RMSE (mg/dL)"); ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout(); fig.savefig(OUT / "fig5_federated.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    arch(); horizon(); scenarios(); personalisation(); federated()
    print("figures ->", OUT)
