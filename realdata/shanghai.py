"""
Loader for the ShanghaiT1DM / ShanghaiT2DM datasets
(Zhao et al., Sci. Data 2023, doi:10.1038/s41597-023-01940-7;
 data: figshare doi:10.6084/m9.figshare.c.6310860).

Each Excel file is one monitoring period of one patient. We resample to a
regular 15-minute grid and build three channels per step:
  glucose  CGM in mg/dL (short gaps of <= 30 min linearly interpolated)
  insulin  IU of prandial/short-acting/premixed insulin delivered in the step
           (s.c. injections, CSII bolus, and insulin given i.v.)
  meal     1 if a meal was recorded in the step, else 0
Long-acting basal injections (degludec, detemir, glargine) and the CSII basal
rate are not part of the action channel; meals have no carbohydrate amounts in
the source, so they enter as events.
"""
from __future__ import annotations
import glob, os, re
import numpy as np
import pandas as pd

STEP_MIN = 15
LONG_ACTING = re.compile(r"degludec|detemir|glargine|glarigine", re.I)
DOSE = re.compile(r"(\d+(?:\.\d+)?)\s*IU", re.I)


def _col(df, prefix):
    for c in df.columns:
        if str(c).strip().lower().startswith(prefix):
            return c
    return None


def parse_sc(text):
    """-> (prandial IU, long-acting IU) from a free-text s.c. insulin entry."""
    if not isinstance(text, str):
        return 0.0, 0.0
    pran = longa = 0.0
    last = 0
    for m in DOSE.finditer(text):
        name = text[last:m.start()]
        dose = float(m.group(1))
        if LONG_ACTING.search(name):
            longa += dose
        else:
            pran += dose
        last = m.end()
    return pran, longa


def parse_iv(text):
    """IU of insulin in an i.v. infusion entry ('..., 4 IU Novolin R, ...')."""
    if not isinstance(text, str):
        return 0.0
    return float(sum(float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*IU", text)))


def load_file(path):
    df = pd.read_excel(path)
    t = pd.to_datetime(df[_col(df, "date")], errors="coerce")
    keep = t.notna().to_numpy()
    df, t = df[keep].reset_index(drop=True), t[keep].reset_index(drop=True)
    cgm_c = _col(df, "cgm")
    g = pd.to_numeric(df[cgm_c], errors="coerce") if cgm_c else pd.Series(np.nan, index=df.index)
    sc_c, iv_c = _col(df, "insulin dose - s.c"), _col(df, "insulin dose - i.v")
    bol_c, diet_c = _col(df, "csii - bolus"), _col(df, "dietary intake")

    pran = np.zeros(len(df)); longa = np.zeros(len(df))
    if sc_c:
        for i, v in enumerate(df[sc_c]):
            pran[i], longa[i] = parse_sc(v)
    if iv_c:
        pran += np.array([parse_iv(v) for v in df[iv_c]])
    if bol_c:
        pran += pd.to_numeric(df[bol_c], errors="coerce").fillna(0).to_numpy()
    meal = np.zeros(len(df))
    if diet_c:
        d = df[diet_c].astype(str).str.strip().str.lower()
        meal = (df[diet_c].notna() & ~d.isin(["data not available", "nan", ""])).to_numpy().astype(float)

    # regular 15-min grid anchored at the first CGM reading
    ok = g.notna()
    if ok.sum() < 20:
        return None
    t0 = t[ok].min(); t1 = t[ok].max()
    n = int((t1 - t0).total_seconds() // (STEP_MIN * 60)) + 1
    grid_g = np.full(n, np.nan); ins = np.zeros(n); mealv = np.zeros(n); longv = np.zeros(n)
    idx = np.floor(((t - t0).dt.total_seconds() + STEP_MIN * 30) / (STEP_MIN * 60)).astype(int).to_numpy()
    for k, (j, gv) in enumerate(zip(idx, g.to_numpy())):
        if 0 <= j < n:
            if not np.isnan(gv):
                grid_g[j] = gv
            ins[j] += pran[k]; mealv[j] = max(mealv[j], meal[k]); longv[j] += longa[k]
    gs = pd.Series(grid_g).interpolate(limit=2, limit_area="inside").to_numpy()
    base = os.path.basename(path)
    pid = base.split("_")[0]
    return dict(glucose=gs.astype(np.float32), insulin=ins.astype(np.float32),
                carbs=mealv.astype(np.float32), long_acting=longv.astype(np.float32),
                pid=pid, file=base,
                dtype="T1DM" if "T1DM" in path else "T2DM")


def load_all(root):
    recs = []
    for f in sorted(glob.glob(os.path.join(root, "Shanghai_T*DM", "*.xls*"))):
        r = load_file(f)
        if r is not None:
            recs.append(r)
    return recs


if __name__ == "__main__":
    import sys
    recs = load_all(sys.argv[1])
    pats = {(r["dtype"], r["pid"]) for r in recs}
    print("records", len(recs), "patients", len(pats),
          "T1DM", sum(1 for p in pats if p[0] == "T1DM"), "T2DM", sum(1 for p in pats if p[0] == "T2DM"))
    steps = sum(len(r["glucose"]) for r in recs)
    print("15-min steps", steps, "~days", round(steps / 96, 1))
    g = np.concatenate([r["glucose"] for r in recs]); g = g[~np.isnan(g)]
    print(f"CGM mean {g.mean():.1f} sd {g.std():.1f}; TIR {((g>=70)&(g<=180)).mean()*100:.1f}%  TBR {(g<70).mean()*100:.2f}%")
    ins = np.concatenate([r["insulin"] for r in recs]); meal = np.concatenate([r["carbs"] for r in recs])
    print(f"steps with prandial insulin {(ins>0).mean()*100:.2f}%  median dose {np.median(ins[ins>0]):.1f} IU; meal steps {(meal>0).mean()*100:.2f}%")
    both = ((ins > 0) & (meal > 0)).sum()
    print(f"insulin steps that coincide with a meal (same step): {both/(ins>0).sum()*100:.1f}%")
