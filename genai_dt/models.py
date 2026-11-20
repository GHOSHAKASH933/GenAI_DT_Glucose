"""Generative twin (conditional VAE) and baselines."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import data as D

torch.set_num_threads(2)


def mlp(sizes, out_act=None):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:-1]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
    layers += [nn.Linear(sizes[-2], sizes[-1])]
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class CVAETwin(nn.Module):
    """
    Conditional VAE over future glucose trajectories.

        q(z | ctx, fut)   encoder
        p(fut | ctx, z)   decoder, heteroscedastic Gaussian

    Sampling z gives a distribution over plausible patient futures: the
    "scenario simulation" capability, made concrete and measurable.
    """

    def __init__(self, ctx_dim=D.CTX_DIM, horizon=D.HORIZON, z_dim=16, hidden=128):
        super().__init__()
        self.z_dim = z_dim
        self.horizon = horizon
        self.enc = mlp([ctx_dim + horizon, hidden, hidden, 2 * z_dim])
        self.dec = mlp([ctx_dim + z_dim, hidden, hidden, 2 * horizon])

    def encode(self, x, y):
        h = self.enc(torch.cat([x, y], -1))
        mu, logvar = h.chunk(2, -1)
        return mu, logvar.clamp(-8, 4)

    def decode(self, x, z):
        h = self.dec(torch.cat([x, z], -1))
        mean, logsig = h.chunk(2, -1)
        return mean, logsig.clamp(-6, 2)

    def forward(self, x, y, beta=1.0):
        mu, logvar = self.encode(x, y)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        mean, logsig = self.decode(x, z)
        nll = (logsig + 0.5 * ((y - mean) / logsig.exp()) ** 2).sum(-1).mean()
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1)).mean()
        return nll + beta * kl, nll.detach(), kl.detach()

    @torch.no_grad()
    def sample(self, x, n_samples=64, include_obs_noise=True):
        """-> (n_samples, B, horizon) in normalised units. Vectorised over samples."""
        B = x.shape[0]
        xr = x.unsqueeze(0).expand(n_samples, B, x.shape[-1]).reshape(-1, x.shape[-1])
        z = torch.randn(n_samples * B, self.z_dim)
        mean, logsig = self.decode(xr, z)
        if include_obs_noise:
            mean = mean + logsig.exp() * torch.randn_like(mean)
        return mean.view(n_samples, B, self.horizon)


class ActionResidualTwin(nn.Module):
    """
    Action-residual conditional VAE (AR-Twin).

    A plain CVAE attenuates the effect of its own action channel: the latent z is
    trained with the future in view, so it can explain outcome variation on its
    own and the decoder learns to underuse the conditioning. Measured here at
    roughly a 3-4x shrinkage of the true dose-response.

    This model makes the action structurally un-ignorable:

        mu(ctx, a, z) = baseline(ctx_passive, z) - softplus(effect(ctx, z)) * a_hat

    where `a_hat` is the queried dose and the effect term is forced non-negative,
    so (i) a = 0 recovers the passive forecast exactly, (ii) the dose cannot be
    routed around, and (iii) the sign of the insulin effect is constrained by
    construction rather than hoped for. The effect magnitude remains
    context- and latent-dependent, so heterogeneity is preserved.

    The free-bits floor on the KL keeps the latent from being squeezed so hard
    that it stops representing physiological variability.
    """

    def __init__(self, horizon=D.HORIZON, z_dim=16, hidden=128, free_bits=0.03):
        super().__init__()
        self.z_dim = z_dim
        self.horizon = horizon
        self.free_bits = free_bits
        self.passive_dim = D.CONTEXT * 3 + D.HORIZON      # history + future carbs
        self.enc = mlp([D.CTX_DIM + horizon, hidden, hidden, 2 * z_dim])
        self.base = mlp([self.passive_dim + z_dim, hidden, hidden, 2 * horizon])
        self.eff = mlp([self.passive_dim + z_dim, hidden, horizon])

    @staticmethod
    def split_action(x):
        """-> (passive context, first-step dose in clinical units)"""
        lo = D.CONTEXT * 3
        passive = torch.cat([x[:, :lo], x[:, lo + D.HORIZON:]], dim=1)
        dose = x[:, lo:lo + 1] * D.I_SCALE
        return passive, dose

    def encode(self, x, y):
        h = self.enc(torch.cat([x, y], -1))
        mu, logvar = h.chunk(2, -1)
        return mu, logvar.clamp(-8, 4)

    def decode(self, x, z):
        passive, dose = self.split_action(x)
        h = self.base(torch.cat([passive, z], -1))
        mean, logsig = h.chunk(2, -1)
        # non-negative, context-dependent per-step glucose-lowering per unit
        k = nn.functional.softplus(self.eff(torch.cat([passive, z], -1)))
        mean = mean - (k * dose) / D.G_SCALE
        return mean, logsig.clamp(-6, 2)

    def forward(self, x, y, beta=1.0):
        mu, logvar = self.encode(x, y)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        mean, logsig = self.decode(x, z)
        nll = (logsig + 0.5 * ((y - mean) / logsig.exp()) ** 2).sum(-1).mean()
        kl_dim = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
        kl = kl_dim.clamp(min=self.free_bits).sum(-1).mean()
        return nll + beta * kl, nll.detach(), kl.detach()

    @torch.no_grad()
    def sample(self, x, n_samples=64, include_obs_noise=True):
        B = x.shape[0]
        xr = x.unsqueeze(0).expand(n_samples, B, x.shape[-1]).reshape(-1, x.shape[-1])
        z = torch.randn(n_samples * B, self.z_dim)
        mean, logsig = self.decode(xr, z)
        if include_obs_noise:
            mean = mean + logsig.exp() * torch.randn_like(mean)
        return mean.view(n_samples, B, self.horizon)


class PointMLP(nn.Module):
    """Deterministic predictor: same capacity, no uncertainty. Baseline."""

    def __init__(self, ctx_dim=D.CTX_DIM, horizon=D.HORIZON, hidden=128):
        super().__init__()
        self.net = mlp([ctx_dim, hidden, hidden, horizon])

    def forward(self, x):
        return self.net(x)


class EdgeStudent(nn.Module):
    """Small distilled model for edge deployment. Predicts quantiles directly."""

    def __init__(self, ctx_dim=D.CTX_DIM, horizon=D.HORIZON, hidden=32):
        super().__init__()
        self.net = mlp([ctx_dim, hidden, 3 * horizon])
        self.horizon = horizon

    def forward(self, x):
        h = self.net(x)
        lo, mid, hi = h.chunk(3, -1)
        # enforce monotone quantiles by construction
        return mid - nn.functional.softplus(lo), mid, mid + nn.functional.softplus(hi)


def train_cvae(model, X, Y, epochs=30, bs=256, lr=2e-3, beta=0.5, seed=0, verbose=False):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    X, Y = torch.as_tensor(X), torch.as_tensor(Y)
    n = len(X)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        # KL warm-up avoids posterior collapse
        b = beta * min(1.0, (ep + 1) / max(1, epochs * 0.3))
        tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            loss, nll, kl = model(X[idx], Y[idx], beta=b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    ep{ep:3d} loss={tot / n:8.4f}")
    model.eval()
    return model


def train_point(model, X, Y, epochs=30, bs=256, lr=2e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    X, Y = torch.as_tensor(X), torch.as_tensor(Y)
    n = len(X)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------- baselines

class Persistence:
    """Future = last observed value. The honest floor."""

    def predict(self, X):
        last_g = X[:, D.CONTEXT - 1]
        return np.repeat(last_g[:, None], D.HORIZON, axis=1)


class RidgeAR:
    """Ridge regression on the raw context. Classical linear baseline."""

    def __init__(self, lam=1.0):
        self.lam = lam
        self.W = None

    def fit(self, X, Y):
        Xb = np.hstack([X, np.ones((len(X), 1), np.float32)])
        A = Xb.T @ Xb + self.lam * np.eye(Xb.shape[1], dtype=np.float32)
        self.W = np.linalg.solve(A, Xb.T @ Y)
        return self

    def predict(self, X):
        Xb = np.hstack([X, np.ones((len(X), 1), np.float32)])
        return Xb @ self.W
