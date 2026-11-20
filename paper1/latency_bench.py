"""Single-window latency, parameter count and serialised size of every model in the extended benchmark.
Latency does not depend on the weights, so untrained networks are timed. One CPU thread, median of 1500 calls, five interleaved rounds.
Run it on an otherwise idle machine."""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import p1_experiments as P
import p1_advanced as A
from p1_experiments import torch, D, H, ROOT, RidgeAR, CVAETwin, GRUForecaster, GaussMLP, PointMLP, model_bytes
from p1_hybrid import HybridTwin

torch.manual_seed(0); torch.set_num_threads(1)
x = torch.randn(1, D.CTX_DIM); xn = x.numpy()
ridge = RidgeAR(lam=5.0).fit(np.random.randn(500, D.CTX_DIM).astype(np.float32), np.random.randn(500, H).astype(np.float32))
tf, gm, gru, qn = A.TransformerGauss().eval(), GaussMLP().eval(), GRUForecaster().eval(), A.QuantileNet().eval()
cv, df = CVAETwin(z_dim=16).eval(), A.CondDiffusion(np.ones(H)).eval()


class MB:                                   # transformer mean as the base of a hybrid
    def __init__(s, n): s.n = n
    def predict(s, X):
        with torch.no_grad(): return s.n(torch.as_tensor(np.asarray(X)))[0].numpy()


def sampler(m, n=32):
    return lambda: m.sample(x, n_samples=n)


def timed(fn, reps=300):
    for _ in range(20): fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def nparams(*ms): return int(sum(p.numel() for m in ms for p in m.parameters()))


bench = {
    "ridge": (lambda: ridge.predict(xn), (H + D.CTX_DIM * H + H), None),
    "gru": (lambda: gru(x), nparams(gru), gru),
    "gauss_mlp": (sampler(gm), nparams(gm), gm),
    "transformer": (sampler(tf), nparams(tf), tf),
    "cqr": (lambda: qn(x), nparams(qn), qn),
    "twin": (sampler(cv), nparams(cv), cv),
    "diffusion": (sampler(df), nparams(df), df),
    "hybrid": (sampler(HybridTwin(ridge, cv)), nparams(cv) + D.CTX_DIM * H, cv),
    "hybrid_diffusion": (sampler(HybridTwin(ridge, df)), nparams(df) + D.CTX_DIM * H, df),
    "tf_cvae": (sampler(HybridTwin(MB(tf), cv)), nparams(tf, cv), None),
    "tf_diffusion": (sampler(HybridTwin(MB(tf), df)), nparams(tf, df), None),
}
out = {}
REPS = int(os.environ.get("LAT_REPS", 300))
# Round-robin over the models so that slow drift of a shared machine affects all of them alike; 5 rounds of REPS calls each.
ts = {k: [] for k in bench}
with torch.no_grad():
    for k, (fn, n, mod) in bench.items():
        for _ in range(20): fn()
    for rnd in range(5):
        for _ in range(REPS):
            for k, (fn, n, mod) in bench.items():
                t = time.perf_counter(); fn(); ts[k].append((time.perf_counter() - t) * 1000)
for k, (fn, n, mod) in bench.items():
    med, p90 = float(np.median(ts[k])), float(np.percentile(ts[k], 90))
    out[k] = dict(latency_ms=med, latency_p90_ms=p90, params=n)
    print(f"{k:18s} {med:8.3f} ms  p90 {p90:8.3f}  {n:8d} params", flush=True)
out["note"] = "untrained networks, 32 samples per window for sampling models, DDIM with 15 steps, one CPU thread"
json.dump(out, open(ROOT / "results" / os.environ.get("LAT_OUT", "paper1_latency.json"), "w"), indent=1)
