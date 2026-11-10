"""Training and evaluation primitives shared by every approach.

Centralized training, the Flower client, the reference FedAvg loop and local fine-tuning all
call these functions, so the optimizer, loss, batching and shuffling are identical.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fedheart.model import HeartMLP, build_model, clone_state


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle_seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    gen = torch.Generator().manual_seed(shuffle_seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen, drop_last=False)


def train_epoch(model: HeartMLP, loader: DataLoader, optimizer: torch.optim.Optimizer) -> tuple:
    """One pass over the loader. Returns (sample-weighted mean training loss, optimizer steps)."""
    model.train()
    total, n, steps = 0.0, 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb)
        n += len(yb)
        steps += 1
    return total / n, steps


@torch.no_grad()
def mean_bce(model: HeartMLP, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    logits = model(torch.from_numpy(X))
    target = torch.from_numpy(y.astype(np.float32))
    return F.binary_cross_entropy_with_logits(logits, target).item()


@torch.no_grad()
def predict_proba(model: HeartMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X))).numpy().astype(np.float64)


def steps_per_epoch(n: int, batch_size: int) -> int:
    return math.ceil(n / batch_size)


def local_train(
    state: dict[str, torch.Tensor],
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float,
    batch_size: int,
    shuffle_seeds: list[int],
    hidden: int,
) -> tuple[dict[str, torch.Tensor], float, int]:
    """FedAvg client update: fresh Adam, len(shuffle_seeds) local epochs from `state`."""
    model = build_model(state, X.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss, steps = float("nan"), 0
    for s in shuffle_seeds:
        loss, k = train_epoch(model, make_loader(X, y, batch_size, s), optimizer)
        steps += k
    return clone_state(model.state_dict()), loss, steps
