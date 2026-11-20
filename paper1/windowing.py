"""Windowing and patient-level splitting for the ShanghaiT1DM/T2DM records.

A window has `D.CONTEXT` steps of history (glucose, insulin, meal) and `D.HORIZON`
steps of future. The model input also contains the insulin and meal entries recorded
in the forecast window, so forecasts are conditioned on what the patient did next.
Windows overlapping a gap longer than the interpolation limit are dropped.
"""
from __future__ import annotations
import numpy as np
import genai_dt.data as D


def windows(rec, stride=2):
    g, i, c = rec["glucose"], rec["insulin"], rec["carbs"]
    K, H = D.CONTEXT, D.HORIZON
    X, Y = [], []
    for s in range(0, len(g) - K - H + 1, stride):
        gw = g[s:s + K + H]
        if np.isnan(gw).any():
            continue
        X.append(D.encode_context(g[s:s + K], i[s:s + K], c[s:s + K],
                                  i[s + K:s + K + H], c[s + K:s + K + H]))
        Y.append((g[s + K:s + K + H] - D.G_MEAN) / D.G_SCALE)
    if not X:
        return np.zeros((0, D.CTX_DIM), np.float32), np.zeros((0, H), np.float32)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def pack(recs, stride=2):
    """Stack the windows of several records; also return each window's diabetes type."""
    xs, ys, ts = [], [], []
    for r in recs:
        x, y = windows(r, stride)
        if len(x):
            xs.append(x); ys.append(y); ts += [r["dtype"]] * len(x)
    return np.concatenate(xs), np.concatenate(ys), np.asarray(ts)


def split(recs, seed, frac=(0.6, 0.2)):
    """Patient-level split stratified by diabetes type. All records of one patient
    go to the same group ('train', 'calib' for validation, 'test')."""
    rng = np.random.default_rng(1000 + seed)
    out = {"train": [], "calib": [], "test": []}
    for dt in ("T1DM", "T2DM"):
        pids = sorted({r["pid"] for r in recs if r["dtype"] == dt})
        rng.shuffle(pids)
        n = len(pids); a = int(round(frac[0] * n)); b = a + int(round(frac[1] * n))
        groups = {p: ("train" if k < a else "calib" if k < b else "test") for k, p in enumerate(pids)}
        for r in recs:
            if r["dtype"] == dt:
                out[groups[r["pid"]]].append(r)
    return out
