"""Federated training: FedAvg, and FedAvg + local personalisation."""
from __future__ import annotations

import copy
import numpy as np
import torch

from .models import CVAETwin, ActionResidualTwin, train_cvae


def _avg_state(states, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = copy.deepcopy(states[0])
    for k in out:
        if out[k].dtype.is_floating_point:
            acc = torch.zeros_like(out[k], dtype=torch.float64)
            for s, wi in zip(states, w):
                acc += s[k].double() * wi
            out[k] = acc.to(out[k].dtype)
        else:
            out[k] = states[0][k].clone()
    return out


def train_local_only(site_data, seed=0, epochs=30, cls=CVAETwin, **kw):
    """One model per site, no communication. Baseline."""
    models = {}
    for site, (X, Y) in site_data.items():
        torch.manual_seed(seed)
        m = cls()
        train_cvae(m, X, Y, epochs=epochs, seed=seed, **kw)
        models[site] = m
    return models


def train_fedavg(site_data, seed=0, rounds=10, local_epochs=3, cls=CVAETwin, **kw):
    """
    Standard FedAvg. Returns the single global model.

    Raw data never leaves a site; only parameter updates are exchanged. We do not
    claim formal privacy here — FedAvg alone is not differentially private, and
    we say so in the paper rather than implying it is.
    """
    torch.manual_seed(seed)
    gm = cls()
    sizes = {s: len(X) for s, (X, _) in site_data.items()}
    for r in range(rounds):
        states, weights = [], []
        for site, (X, Y) in site_data.items():
            local = cls()
            local.load_state_dict(copy.deepcopy(gm.state_dict()))
            train_cvae(local, X, Y, epochs=local_epochs, seed=seed * 100 + r, **kw)
            states.append(local.state_dict())
            weights.append(sizes[site])
        gm.load_state_dict(_avg_state(states, weights))
    gm.eval()
    return gm


def personalise(global_model, site_data, seed=0, epochs=5, lr=5e-4, cls=CVAETwin, **kw):
    """
    Per-site adaptation of the FedAvg model: decoder head fine-tuned locally,
    encoder/shared trunk frozen. Cheap, and the standard remedy for non-IID drift.
    """
    out = {}
    for site, (X, Y) in site_data.items():
        m = cls()
        m.load_state_dict(copy.deepcopy(global_model.state_dict()))
        for p in m.enc.parameters():
            p.requires_grad_(False)
        trunk = m.dec if hasattr(m, "dec") else m.base
        for layer in list(trunk)[:-1]:
            for p in layer.parameters():
                p.requires_grad_(False)
        train_cvae(m, X, Y, epochs=epochs, lr=lr, seed=seed, **kw)
        out[site] = m
    return out
