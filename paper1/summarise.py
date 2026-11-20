"""Print the headline results of the extended benchmark as a Markdown table (mean over seeds).
Usage: python paper1/summarise.py > results/summary.md"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runs = json.load(open(ROOT / "results" / "paper1_advanced.json"))["runs"]
lat = json.load(open(ROOT / "results" / "paper1_latency.json")) if (ROOT / "results" / "paper1_latency.json").exists() else {}
NAMES = [("ridge", "Ridge regression"), ("gru", "GRU"), ("gauss_mlp", "Gaussian MLP"), ("transformer", "Transformer, Gaussian head"),
         ("cqr", "Quantile network + CQR"), ("twin", "CVAE twin"), ("diffusion", "Diffusion twin"),
         ("hybrid", "Ridge + CVAE"), ("hybrid_diffusion", "Ridge + diffusion"), ("tf_cvae", "Transformer + CVAE"),
         ("tf_diffusion", "Transformer + diffusion")]


def mean(model, *path):
    v = []
    for r in runs:
        x = r["models"][model]
        for p in path:
            x = x.get(p) if isinstance(x, dict) else None
            if x is None:
                break
        if x is not None:
            v.append(x)
    return float(np.mean(v)) if v else None


def f(x, p=1):
    return "–" if x is None else f"{x:.{p}f}"


print(f"Mean over {len(runs)} patient-level splits (seeds {runs[0]['seed']}-{runs[-1]['seed']}). RMSE, CRPS and interval score in mg/dL.\n")
print("| Model | RMSE 60 | CRPS 60 | Interval score (conformal) | Coverage | AUROC hypo | Brier hypo | ECE hypo | Energy dist. | Latency (ms) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k, name in NAMES:
    print(f"| {name} | {f(mean(k, 'accuracy', 'rmse_60'))} | {f(mean(k, 'prob', 'crps_60'))} | "
          f"{f(mean(k, 'interval_conf', 'interval_score'))} | {f(mean(k, 'interval_conf', 'coverage'), 3)} | "
          f"{f(mean(k, 'hypo', 'auroc'), 3)} | {f(mean(k, 'hypo', 'brier'), 3)} | {f(mean(k, 'hypo', 'ece'), 3)} | "
          f"{f(mean(k, 'realism', 'energy'), 2)} | {f(lat.get(k, {}).get('latency_ms'), 2) if lat else '–'} |")

# Per-type coverage, pooled vs Mondrian conformal. The first run did not store per-type intervals; the rerun on the
# same splits (results/paper1_advanced_rerun.json) does, so the table is read from whichever file has them.
if mean("twin", "interval_mondrian", "coverage") is None and (ROOT / "results" / "paper1_advanced_rerun.json").exists():
    runs = json.load(open(ROOT / "results" / "paper1_advanced_rerun.json"))["runs"]
    print(f"\n(Per-type table from the rerun of the extended benchmark on the same {len(runs)} splits.)")
rows = [(k, name) for k, name in NAMES if mean(k, "interval_mondrian", "coverage") is not None]
if rows:
    print("\nCoverage of 90% intervals at 60 min by diabetes type: pooled split-conformal vs per-type (Mondrian) correction.\n")
    print("| Model | Pooled T1DM | Pooled T2DM | Mondrian T1DM | Mondrian T2DM | Interval score pooled | Interval score Mondrian |")
    print("|---|---|---|---|---|---|---|")
    for k, name in rows:
        print(f"| {name} | {f(mean(k, 'interval_conf', 'coverage_T1DM'), 3)} | {f(mean(k, 'interval_conf', 'coverage_T2DM'), 3)} | "
              f"{f(mean(k, 'interval_mondrian', 'coverage_T1DM'), 3)} | {f(mean(k, 'interval_mondrian', 'coverage_T2DM'), 3)} | "
              f"{f(mean(k, 'interval_conf', 'interval_score'))} | {f(mean(k, 'interval_mondrian', 'interval_score'))} |")
