import numpy as np
import torch

import p1_experiments as P
import p1_advanced as A
import genai_dt.data as D
from genai_dt.models import CVAETwin, RidgeAR

B, H = 6, D.HORIZON


def x_batch(n=B):
    torch.manual_seed(0)
    return torch.randn(n, D.CTX_DIM)


def test_sample_shapes():
    x = x_batch()
    for m in (CVAETwin(z_dim=4), P.GaussMLP(), A.TransformerGauss(d=16, heads=2, layers=1),
              A.CondDiffusion(np.ones(H), T=10)):
        m.eval()
        assert tuple(m.sample(x, n_samples=5).shape) == (5, B, H)


def test_quantile_net_shape_and_loss():
    net = A.QuantileNet(hidden=16)
    x, y = x_batch(), torch.randn(B, H)
    assert tuple(net(x).shape) == (B, len(A.QL), H)
    assert A.pinball(net, x, y).item() > 0


def test_diffusion_sampling_is_seeded_and_finite():
    m = A.CondDiffusion(np.ones(H), T=10); m.eval()
    x = x_batch()
    torch.manual_seed(5); a = m.sample(x, 4, steps=5)
    torch.manual_seed(5); b = m.sample(x, 4, steps=5)
    assert torch.allclose(a, b) and torch.isfinite(a).all()


def test_hybrid_twin_is_base_plus_residual():
    class Zero:
        def predict(self, X):
            return np.zeros((len(X), H), np.float32)

    class Const:
        def predict(self, X):
            return np.full((len(X), H), 2.0, np.float32)

    cv = CVAETwin(z_dim=4); cv.eval(); x = x_batch()
    torch.manual_seed(1); s0 = A.HybridTwin(Zero(), cv).sample(x, 3)
    torch.manual_seed(1); s1 = A.HybridTwin(Const(), cv).sample(x, 3)
    assert torch.allclose(s1 - s0, torch.full_like(s0, 2.0), atol=1e-5)


def test_fit_reduces_loss():
    torch.manual_seed(0)
    X = torch.randn(512, D.CTX_DIM); Y = X[:, :H] * 0.5
    m = P.GaussMLP(hidden=32)
    before = float(A.gauss_loss(m, X, Y))
    A.fit(m, X.numpy(), Y.numpy(), A.gauss_loss, epochs=5, seed=0)
    assert float(A.gauss_loss(m, X, Y)) < before


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 5)).astype(np.float32); W = rng.standard_normal((5, 3)).astype(np.float32)
    r = RidgeAR(lam=1e-3).fit(X, X @ W)
    assert np.allclose(r.predict(X), X @ W, atol=1e-2)
