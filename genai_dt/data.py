"""
Windowing and normalisation. Splits are by PATIENT, never by window.

The twin is ACTION-CONDITIONED: it predicts p(future glucose | past, planned
future insulin, announced future carbs). This matters. A twin trained only on
past context learns the observational distribution, yet a controller queries it
counterfactually ("what if I dose now?"). Conflating the two is a live error in
the digital-twin literature; conditioning on the action is the minimum fix.
"""
from __future__ import annotations

import numpy as np

CONTEXT = 24      # 2 h of history at 5-min sampling
HORIZON = 24      # 120 min — must cover the subcutaneous insulin action peak

G_MEAN, G_SCALE = 140.0, 50.0     # fixed constants; no statistics from test data
I_SCALE = 2.0
C_SCALE = 40.0

CTX_DIM = CONTEXT * 3 + HORIZON * 2   # past (g,i,c) + planned (insulin, carbs)


def encode_context(g, i, c, fut_i, fut_c):
    gn = (g - G_MEAN) / G_SCALE
    return np.concatenate([gn, i / I_SCALE, c / C_SCALE,
                           fut_i / I_SCALE, fut_c / C_SCALE], axis=-1)


DECISION_PERIOD = 6   # 30 min between correction decisions


def windows_from_record(rec, stride: int = DECISION_PERIOD):
    """
    Window starts are aligned to decision epochs, so that horizon step 0 is
    always a point at which a correction dose may be given. Without this
    alignment the action channel is almost always zero at the step the
    controller actually queries, and the twin sees no dose variation there.
    """
    g, i, c = rec["glucose"], rec["insulin"], rec["carbs"]
    n = len(g)
    assert CONTEXT % DECISION_PERIOD == 0
    starts = np.arange(0, n - CONTEXT - HORIZON, stride)
    if len(starts) == 0:
        return (np.zeros((0, CTX_DIM), np.float32),
                np.zeros((0, HORIZON), np.float32))
    ctx = np.stack([
        encode_context(g[s:s + CONTEXT], i[s:s + CONTEXT], c[s:s + CONTEXT],
                       i[s + CONTEXT:s + CONTEXT + HORIZON],
                       c[s + CONTEXT:s + CONTEXT + HORIZON])
        for s in starts])
    fut = np.stack([g[s + CONTEXT:s + CONTEXT + HORIZON] for s in starts])
    return ctx.astype(np.float32), ((fut - G_MEAN) / G_SCALE).astype(np.float32)


def denorm(y):
    return y * G_SCALE + G_MEAN


def split_by_patient(cohort, seed: int, frac=(0.60, 0.20, 0.20)):
    """Patient-level train/calibration/test split, stratified by site."""
    rng = np.random.default_rng(seed + 9871)
    train, calib, test = [], [], []
    for site in sorted({r["site"] for r in cohort}):
        members = [r for r in cohort if r["site"] == site]
        order = rng.permutation(len(members))
        n = len(members)
        n_tr = int(round(frac[0] * n))
        n_ca = int(round(frac[1] * n))
        for j, idx in enumerate(order):
            (train if j < n_tr else calib if j < n_tr + n_ca else test).append(members[idx])
    return train, calib, test


def pack(records, stride: int = DECISION_PERIOD):
    xs, ys, pids = [], [], []
    for r in records:
        x, y = windows_from_record(r, stride)
        if len(x):
            xs.append(x); ys.append(y)
            pids.append(np.full(len(x), r["pid"], dtype=object))
    if not xs:
        return (np.zeros((0, CTX_DIM), np.float32),
                np.zeros((0, HORIZON), np.float32),
                np.zeros((0,), dtype=object))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(pids)


def by_site(records):
    out = {}
    for r in records:
        out.setdefault(r["site"], []).append(r)
    return out
