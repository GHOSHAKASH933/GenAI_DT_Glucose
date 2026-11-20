import numpy as np
import genai_dt.data as D
import p1_experiments as P        # patches the shared constants (8-step context and horizon)
from windowing import windows, pack, split


def record(n=100, pid="1", dtype="T2DM", nan_at=None):
    g = 100.0 + np.arange(n, dtype=float)
    if nan_at is not None:
        g[nan_at] = np.nan
    return dict(glucose=g, insulin=np.zeros(n), carbs=np.zeros(n), dtype=dtype, pid=pid)


def test_window_shapes_and_alignment():
    X, Y = windows(record(), stride=2)
    K, H = D.CONTEXT, D.HORIZON
    assert X.shape == (len(range(0, 100 - K - H + 1, 2)), D.CTX_DIM) and Y.shape[1] == H
    g = 100.0 + np.arange(100)
    assert np.allclose(Y[0], (g[K:K + H] - D.G_MEAN) / D.G_SCALE, atol=1e-5)
    assert np.allclose(X[0, :K], (g[:K] - D.G_MEAN) / D.G_SCALE, atol=1e-5)


def test_windows_with_missing_values_are_dropped():
    full, _ = windows(record(), 1)
    gap, _ = windows(record(nan_at=50), 1)
    assert len(gap) < len(full)
    X, _ = windows(record(nan_at=50), 1)
    assert not np.isnan(X).any()


def test_split_is_by_patient_and_stratified():
    recs = [record(pid=str(i), dtype="T1DM" if i < 10 else "T2DM") for i in range(50)]
    sp = split(recs, seed=0)
    groups = {k: {r["pid"] for r in v} for k, v in sp.items()}
    assert not (groups["train"] & groups["calib"]) and not (groups["train"] & groups["test"]) \
        and not (groups["calib"] & groups["test"])
    assert sum(len(v) for v in groups.values()) == 50
    for dt in ("T1DM", "T2DM"):
        assert all(any(r["dtype"] == dt for r in sp[k]) for k in sp)
    assert [r["pid"] for r in split(recs, 0)["test"]] == [r["pid"] for r in split(recs, 0)["test"]]
    assert [r["pid"] for r in split(recs, 0)["test"]] != [r["pid"] for r in split(recs, 1)["test"]]


def test_two_records_of_one_patient_stay_together():
    recs = [record(pid="7"), record(pid="7"), *[record(pid=str(i)) for i in range(20)]]
    sp = split(recs, seed=3)
    where = {k for k, v in sp.items() if any(r["pid"] == "7" for r in v)}
    assert len(where) == 1


def test_pack_returns_one_type_label_per_window():
    X, Y, T = pack([record(dtype="T1DM"), record(dtype="T2DM")], 4)
    assert len(X) == len(Y) == len(T) and set(T) == {"T1DM", "T2DM"}
