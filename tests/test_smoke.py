"""End-to-end smoke test of the extended benchmark on small synthetic records (about half a minute)."""
import numpy as np
import pytest

import p1_advanced as A


def synthetic(n_pat=18, n=260, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n_pat):
        t = np.arange(n)
        meals = (rng.uniform(size=n) < 0.06).astype(float)
        ins = meals * rng.integers(2, 8, n) * (rng.uniform(size=n) < 0.8)
        g = 130 + 35 * np.sin(t / 9 + i) + np.convolve(meals, np.exp(-np.arange(12) / 4), "same") * 45 \
            - np.convolve(ins, np.exp(-np.arange(12) / 5), "same") * 6 + rng.normal(0, 4, n)
        recs.append(dict(glucose=np.clip(g, 40, 400), insulin=ins, carbs=meals,
                         dtype="T1DM" if i < 6 else "T2DM", pid=str(i)))
    return recs


@pytest.mark.slow
def test_run_seed_quick_produces_all_outputs():
    res = A.run_seed(synthetic(), 0, dict(A.QUICK, stride=3))
    for m in ("ridge", "gru", "gauss_mlp", "transformer", "cqr", "twin", "hybrid", "diffusion",
              "hybrid_diffusion", "tf_cvae", "tf_diffusion"):
        acc = res["models"][m]["accuracy"]
        assert np.isfinite(acc["rmse_60"]) and acc["rmse_30"] <= acc["rmse_120"] * 1.5
    for m in ("gauss_mlp", "transformer", "twin", "hybrid", "diffusion"):
        assert res["models"][m]["prob"]["crps_60"] > 0
        assert 0 <= res["models"][m]["interval_conf"]["coverage"] <= 1
        mond = res["models"][m]["interval_mondrian"]
        assert 0 <= mond["coverage_T1DM"] <= 1 and {"Q_T1DM", "Q_T2DM"} <= set(mond)
    assert "interval_mondrian" in res["models"]["cqr"]
    assert set(res["ablation"]) == {"full", "no_insulin", "no_meals", "no_planned", "glucose_only"}
