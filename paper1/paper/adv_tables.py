"""Tables for the extended benchmark (p1_advanced.py results)."""
import numpy as np
from common import c, m, table

LABEL = {
    "ridge": "Ridge regression", "gru": "GRU", "gauss_mlp": "Gaussian MLP",
    "transformer": "Transformer, Gaussian head", "cqr": "Quantile network with CQR",
    "twin": "CVAE twin", "diffusion": "Diffusion twin",
    "hybrid": "Ridge + CVAE (hybrid twin)", "hybrid_diffusion": "Ridge + diffusion",
    "tf_cvae": "Transformer + CVAE", "tf_diffusion": "Transformer + diffusion",
}
ORDER = ["ridge", "gru", "gauss_mlp", "transformer", "cqr", "twin", "diffusion", "hybrid", "hybrid_diffusion",
         "tf_cvae", "tf_diffusion"]
SAMPLERS = ("gauss_mlp", "transformer", "twin", "diffusion", "hybrid", "hybrid_diffusion", "tf_cvae", "tf_diffusion")


def getter(A):
    def a(model, *path):
        out = []
        for r in A:
            v = r["models"][model]
            for p in path:
                v = v.get(p) if isinstance(v, dict) else None
                if v is None:
                    break
            out.append(np.nan if v is None else v)
        return out
    return a


def has(A, model, *path):
    return np.isfinite(np.nanmean(getter(A)(model, *path)))


def cell(A, model, path, p=1, plain=False):
    if not has(A, model, *path):
        return "--"
    v = getter(A)(model, *path)
    return m(v, p) if plain else c(v, p)


def wide(t):
    return t.replace("\\begin{table}[H]", "\\begin{table}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}", 1).replace(
        "\\end{table}", "\\end{adjustwidth}\n\\end{table}", 1)


def acc_table(A):
    rows = []
    for k in ORDER:
        rows.append(f"{LABEL[k]} & {cell(A, k, ('accuracy', 'rmse_60'))} & {cell(A, k, ('accuracy', 'rmse_120'))} & "
                    f"{cell(A, k, ('prob', 'crps_60'))} & {cell(A, k, ('interval_conf', 'interval_score'))} & "
                    f"{cell(A, k, ('interval_conf', 'coverage'), 3)}")
    return wide(table(
        "Extended benchmark on held-out patients. RMSE in mg/dL, CRPS in mg/dL, interval score of the split-conformal "
        "90\\% interval at 60 minutes (mg/dL, lower is better) and its empirical coverage. Mean over seeds with "
        "bootstrap 95\\% interval; a dash marks a quantity the model does not provide.",
        "tab:adv_acc", "lccccc",
        r"\textbf{Model} & \textbf{RMSE 60} & \textbf{RMSE 120} & \textbf{CRPS 60} & \textbf{Interval score} & \textbf{Coverage}",
        rows, size=r"\footnotesize"))


def risk_table(A):
    rows = []
    for k in ORDER:
        rows.append(f"{LABEL[k]} & {cell(A, k, ('hypo', 'auroc'), 3)} & {cell(A, k, ('hypo', 'sens_at_spec90'), 3)} & "
                    f"{cell(A, k, ('hypo', 'brier'), 3)} & {cell(A, k, ('hypo', 'ece'), 3)} & "
                    f"{cell(A, k, ('realism', 'lag1_sampled'), 2)} & {cell(A, k, ('realism', 'energy'), 2)}")
    return wide(table(
        "Hypoglycaemia risk and scenario realism of the extended benchmark. AUROC, sensitivity at 90\\% specificity, "
        "Brier score and expected calibration error (ECE) refer to glucose below 70 mg/dL within 60 minutes; the last two "
        "columns refer to one sampled scenario per window (observed lag-one autocorrelation "
        f"{m(getter(A)('gauss_mlp', 'realism', 'lag1_real'), 2)}).",
        "tab:adv_risk", "lcccccc",
        r"\textbf{Model} & \textbf{AUROC} & \textbf{Sens.@90\% spec.} & \textbf{Brier} & \textbf{ECE} & "
        r"\textbf{Lag-1 autocorr.} & \textbf{Energy dist.}",
        rows, size=r"\scriptsize"))


def sub_table(A):
    rows = []
    for k in ORDER:
        rows.append(f"{LABEL[k]} & {cell(A, k, ('rmse_60_T1DM',))} & {cell(A, k, ('rmse_60_T2DM',))} & "
                    f"{cell(A, k, ('rmse_60_hypo',))} & {cell(A, k, ('rmse_60_eu',))} & {cell(A, k, ('rmse_60_hyper',))} & "
                    f"{cell(A, k, ('interval_conf', 'coverage_T1DM'), 3)} & {cell(A, k, ('interval_conf', 'coverage_T2DM'), 3)}")
    return wide(table(
        "Accuracy by diabetes type and by glycaemic range of the observed value at 60 minutes (below 70, 70 to 180 and "
        "above 180 mg/dL), and coverage of the pooled split-conformal 90\\% interval within each type.",
        "tab:sub", "lccccccc",
        r"& \multicolumn{5}{c}{\textbf{RMSE at 60 min (mg/dL)}} & \multicolumn{2}{c}{\textbf{Conformal coverage}}\\ "
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}\textbf{Model} & T1DM & T2DM & $<70$ & 70--180 & $>180$ & T1DM & T2DM",
        rows, size=r"\scriptsize"))


ABL = [("full", "All inputs"), ("no_insulin", "Without insulin (past and planned)"),
       ("no_meals", "Without meals (past and planned)"), ("no_planned", "Without planned insulin and meals"),
       ("glucose_only", "Glucose history only")]


def abl_table(R):
    rows = []
    for k, lab in ABL:
        g = lambda f: [r["ablation"][k][f] for r in R]
        rows.append(f"{lab} & {c(g('ridge_rmse_60'))} & {c(g('rmse_60'))} & {c(g('crps_60'))} & {c(g('auroc'), 3)}")
    return table(
        "Input ablation at the 60-minute horizon (RMSE and CRPS in mg/dL). Each row retrains ridge regression and the hybrid twin with the listed inputs set to zero. The hybrid "
        "twin's forecast is the mean of 32 draws.",
        "tab:abl", "lcccc",
        r"\textbf{Inputs} & \textbf{Ridge RMSE} & \textbf{Hybrid RMSE} & \textbf{Hybrid CRPS} & \textbf{Hybrid AUROC}",
        rows, size=r"\footnotesize")


def size_table(R, fractions, n_full=67):
    rows = []
    for f in fractions:
        g = lambda *p: [_dig(r["train_size"][str(f)], p) for r in R]
        n = int(np.mean([r["train_size"][str(f)]["n_patients"] for r in R]))
        rows.append(f"{int(round(100 * f))}\\% (about {n} patients) & {c(g('ridge_rmse_60'))} & {c(g('hybrid', 'rmse_60'))} & "
                    f"{c(g('hybrid', 'crps_60'))} & {c(g('hybrid', 'auroc'), 3)}")
    full = lambda *p: [_dig(r["models"], p) for r in R]
    abl = lambda f: [r["ablation"]["full"][f] for r in R]       # same trained hybrid, scored with the same 32 draws as the subsets
    rows.append(f"100\\% (about {n_full} patients) & {c(full('ridge', 'accuracy', 'rmse_60'))} & "
                f"{c(abl('rmse_60'))} & {c(abl('crps_60'))} & {c(abl('auroc'), 3)}")
    return table(
        "Effect of the number of training patients at the 60-minute horizon (RMSE and CRPS in mg/dL). Ridge regression and the hybrid twin were retrained on random subsets "
        "of the training patients (stratified by diabetes type) and evaluated on the same test patients. The hybrid twin's forecast "
        "is the mean of 32 draws in every row.",
        "tab:size", "lcccc",
        r"\textbf{Training patients} & \textbf{Ridge RMSE} & \textbf{Hybrid RMSE} & \textbf{Hybrid CRPS} & \textbf{Hybrid AUROC}",
        rows, size=r"\footnotesize")


def _dig(d, path):
    for p in path:
        d = d[p]
    return d


def lat_table(LAT):
    rows = []
    for k in ORDER:
        if k in LAT:
            rows.append(f"{LABEL[k]} & {LAT[k]['params']:,} & {LAT[k]['latency_ms']:.2f} & {LAT[k]['latency_p90_ms']:.2f}".replace(",", "{,}"))
    return table(
        "Parameters and single-window latency on one CPU thread (median and 90th percentile of 1500 calls in five interleaved rounds). Sampling models "
        "draw 32 scenarios per window; the diffusion sampler uses 15 steps. The Transformer-based hybrids need both networks. "
        "The quantile network and the point models return one forecast.",
        "tab:lat", "lccc", r"\textbf{Model} & \textbf{Parameters} & \textbf{Median ms} & \textbf{90th pct. ms}", rows)


MOND = [k for k in ORDER if k not in ("ridge", "gru")]      # models with a conformal interval


def mond_table(A):
    """Pooled against per-type (Mondrian) split-conformal 90% intervals at 60 minutes, from the rerun that stores both."""
    rows = []
    for k in MOND:
        rows.append(f"{LABEL[k]} & {cell(A, k, ('interval_conf', 'coverage_T1DM'), 3)} & "
                    f"{cell(A, k, ('interval_mondrian', 'coverage_T1DM'), 3)} & "
                    f"{cell(A, k, ('interval_conf', 'coverage_T2DM'), 3)} & "
                    f"{cell(A, k, ('interval_mondrian', 'coverage_T2DM'), 3)} & "
                    f"{cell(A, k, ('interval_conf', 'interval_score'), 1, plain=True)} & "
                    f"{cell(A, k, ('interval_mondrian', 'interval_score'), 1, plain=True)}")
    return wide(table(
        "Coverage of 90\\% intervals at 60 minutes for each diabetes type after a pooled split-conformal correction and after "
        "a separate (Mondrian) correction for each type, with the interval score (mg/dL, lower is better) over all test "
        "windows. Mean over seeds with bootstrap 95\\% interval; interval scores are means.",
        "tab:mond", "lcccccc",
        r"& \multicolumn{2}{c}{\textbf{T1DM coverage}} & \multicolumn{2}{c}{\textbf{T2DM coverage}} & "
        r"\multicolumn{2}{c}{\textbf{Interval score}}\\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
        r"\textbf{Model} & pooled & per type & pooled & per type & pooled & per type",
        rows, size=r"\scriptsize"))
