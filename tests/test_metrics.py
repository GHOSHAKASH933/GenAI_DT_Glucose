"""Closed-form and property checks for the evaluation metrics."""
import numpy as np
from math import erf, exp, pi, sqrt

import p1_experiments as P
import p1_advanced as A
import genai_dt.data as D


def to_norm(mg):
    return ((np.asarray(mg, float) - D.G_MEAN) / D.G_SCALE).astype(np.float32)


def test_crps_matches_gaussian_closed_form():
    rng = np.random.default_rng(0)
    y = 0.5
    s = rng.standard_normal((4000, 1, 1))
    phi = exp(-y * y / 2) / sqrt(2 * pi)
    Phi = 0.5 * (1 + erf(y / sqrt(2)))
    exact = y * (2 * Phi - 1) + 2 * phi - 1 / sqrt(pi)
    est = P.crps(s, np.array([[y]]))[0, 0]
    assert abs(est - exact) < 0.02


def test_crps_zero_for_perfect_point_forecast():
    s = np.full((16, 5, 3), 7.0)
    assert np.allclose(P.crps(s, np.full((5, 3), 7.0)), 0.0)


def test_clarke_zones():
    a, ab = P.clarke([100, 150, 60], [100, 160, 50])
    assert a == 100.0 and ab == 100.0
    a, _ = P.clarke([60], [400])
    assert a == 0.0


def test_interval_score_known_values():
    T = np.array(["T1DM", "T2DM"])
    out = A.interval_stats(np.array([0.0, 0.0]), np.array([10.0, 10.0]), np.array([5.0, 15.0]), T)
    assert out["coverage"] == 0.5
    assert np.isclose(out["interval_score"], (10 + (10 + 20 * 5)) / 2)


def test_conformal_correction_restores_coverage():
    rng = np.random.default_rng(1)
    yv, yt = rng.normal(0, 1, 8000), rng.normal(0, 1, 8000)
    lo, hi = -0.5, 0.5                       # an interval that is far too narrow
    Q = A.conformal_Q(np.full(8000, lo), np.full(8000, hi), yv, level=0.9)
    cov = ((yt >= lo - Q) & (yt <= hi + Q)).mean()
    assert abs(cov - 0.9) < 0.02


def test_ece():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=50000)
    ev = (rng.uniform(size=p.size) < p).astype(int)
    assert A.ece(p, ev) < 0.02
    assert np.isclose(A.ece(np.full(1000, 0.5), np.r_[np.ones(200), np.zeros(800)].astype(int)), 0.3)


def test_hypo_metrics_perfect_ranking():
    g = np.tile(np.array([120.0, 110, 100, 95, 100, 105, 110, 115]), (200, 1))
    g[:50, 2] = 60.0                         # 50 windows contain a value below 70 mg/dL
    Y = to_norm(g)
    score = (g[:, :P.STEP[60] + 1].min(1) < 70).astype(float)
    out = A.hypo_full(score, Y, prob=score)
    assert out["auroc"] == 1.0 and out["brier"] == 0.0 and out["sens_at_spec90"] == 1.0


def test_mondrian_restores_coverage_in_minority_group():
    """A group with wider errors is under-covered by a pooled correction and covered by
    its own (Mondrian) correction; the majority group keeps its coverage."""
    rng = np.random.default_rng(1)

    def draw(n1, n2):
        T = np.array(["T1DM"] * n1 + ["T2DM"] * n2)
        y = np.where(T == "T1DM", rng.normal(0, 3.0, len(T)), rng.normal(0, 1.0, len(T)))
        return np.full(len(T), -1.0), np.full(len(T), 1.0), y, T

    lo_v, hi_v, y_v, T_v = draw(400, 1100)
    lo, hi, y, T = draw(2000, 2000)
    pooled = A.conformal_Q(lo_v, hi_v, y_v)
    raw = A.interval_stats(lo - pooled, hi + pooled, y, T)
    mond = A.mondrian_stats(lo, hi, y, T, A.mondrian_Q(lo_v, hi_v, y_v, T_v), lo_v, hi_v, y_v, T_v)
    # expected coverage is 0.90 in both groups (sd about 0.016 over draws); pooled gives T1DM about 0.64
    assert raw["coverage_T1DM"] < 0.80
    assert mond["coverage_T1DM"] >= 0.85 and mond["coverage_T2DM"] >= 0.85
    assert mond["coverage_T1DM"] - raw["coverage_T1DM"] > 0.15
    assert mond["Q_T1DM"] > mond["Q_T2DM"]
    assert mond["n_val_windows"] == {"T1DM": 400, "T2DM": 1100}


def test_mondrian_falls_back_to_pooled_for_missing_group():
    lo_v, hi_v = np.full(50, -1.0), np.full(50, 1.0)
    y_v, T_v = np.linspace(-2, 2, 50), np.array(["T2DM"] * 50)
    Qs = A.mondrian_Q(lo_v, hi_v, y_v, T_v)
    assert set(Qs) == {"T2DM"}
    out = A.mondrian_stats(np.zeros(2) - 1, np.zeros(2) + 1, np.zeros(2), np.array(["T1DM", "T2DM"]),
                           Qs, lo_v, hi_v, y_v, T_v)
    assert out["coverage"] == 1.0
