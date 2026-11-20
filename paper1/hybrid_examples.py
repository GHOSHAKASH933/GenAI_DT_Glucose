"""Scenario examples from the hybrid twin, seed 0, same windows as p1_experiments (for Figure 3)."""
import json, sys, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import p1_hybrid as H
from p1_experiments import (D, CVAETwin, RidgeAR, train_cvae, mg, pack, split, load_all, ROOT, GaussMLP, train_gauss)
recs = load_all(str(ROOT / "data")); seed = 0; cfg = H.CFG
sp = split(recs, seed)
Xtr, Ytr, _ = pack(sp["train"], 2); Xte, Yte, Tte = pack(sp["test"], 2)
ridge = RidgeAR(lam=5.0).fit(Xtr, Ytr)
torch.manual_seed(seed)
cv = train_cvae(CVAETwin(z_dim=16), Xtr, (Ytr - ridge.predict(Xtr)).astype(np.float32), epochs=25, seed=seed, beta=0.5)
hy = H.HybridTwin(ridge, cv)
ex = json.load(open(ROOT / "results" / "paper1_results.json"))["runs"][0]["examples"]
yte = mg(Yte); pick = []
for dt in ("T1DM", "T2DM"):
    idx = np.where(Tte == dt)[0]; var = yte[idx].std(1)
    pick += [int(i) for i in idx[np.argsort(var)[-int(0.25 * len(idx))::max(1, int(0.25 * len(idx)) // 2)][:2]]]
assert np.allclose(yte[pick], np.array(ex["truth"]), atol=1e-3)
with torch.no_grad():
    s = mg(hy.sample(Xte[pick], n_samples=20, include_obs_noise=False).numpy())
ex["hybrid"] = s.transpose(1, 0, 2).tolist()
json.dump(ex, open(ROOT / "results" / "paper1_examples.json", "w"))
print("ok")
