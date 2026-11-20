"""Shared formatting and statistics helpers for both manuscripts."""
from __future__ import annotations
import numpy as np
from scipy import stats


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def c(v, p=1):
    """mean (lo--hi)"""
    v = np.asarray(v, float)
    m = np.nanmean(v); lo, hi = boot_ci(v)
    sep = " to " if (lo < 0 or hi < 0) else "--"
    return f"{m:.{p}f} ({lo:.{p}f}{sep}{hi:.{p}f})"


def m(v, p=1):
    return f"{np.nanmean(np.asarray(v, float)):.{p}f}"


def wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = np.isfinite(a) & np.isfinite(b); a, b = a[k], b[k]
    if len(a) < 3 or np.allclose(a, b):
        return float("nan")
    return float(stats.wilcoxon(a, b).pvalue)


def pv(p):
    if not np.isfinite(p):
        return "n/a"
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"


def dz(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); n = len(ps)
    adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (n - r) * ps[i])); adj[i] = run
    return adj


def table(cap, label, spec, header, rows, env="table", size=r"\small"):
    out = [rf"\begin{{{env}}}[H]" if env == "table" else rf"\begin{{{env}}}[t]", r"\centering", size,
           r"\setlength{\tabcolsep}{4pt}", f"\\caption{{{cap}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule"]
    out += [r + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}", ""]
    return "\n".join(out)


def figure(name, cap, label, width=r"\linewidth", pos="H"):
    return (f"\\begin{{figure}}[{pos}]\n\\centering\n\\includegraphics[width={width}]{{{name}}}\n"
            f"\\caption{{{cap}}}\n\\label{{{label}}}\n\\end{{figure}}\n")
