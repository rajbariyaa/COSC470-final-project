"""Centralized baseline: one model trained on the union of the hospitals' training partitions.

Training runs for the full epoch budget; the reported checkpoint is the epoch with the
lowest pooled validation loss (ties -> earliest epoch).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from fedheart.data import HOSPITALS
from fedheart.engine import make_loader, mean_bce, train_epoch
from fedheart.model import build_model, clone_state
from fedheart.prepare import HospitalArrays
from fedheart.utils import derive_seed


def train_centralized(
    arrays: dict[str, HospitalArrays],
    init_state: dict[str, torch.Tensor],
    *,
    lr: float,
    max_epochs: int,
    batch_size: int,
    seed: int,
    hidden: int,
) -> dict:
    X = np.concatenate([arrays[h].X_train for h in HOSPITALS])
    y = np.concatenate([arrays[h].y_train for h in HOSPITALS])
    Xv = np.concatenate([arrays[h].X_val for h in HOSPITALS])
    yv = np.concatenate([arrays[h].y_val for h in HOSPITALS])

    model = build_model(init_state, X.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def val_row(epoch: int, train_loss: float, steps: int) -> dict:
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": mean_bce(model, Xv, yv)}
        for h in HOSPITALS:
            row[f"val_loss_{h}"] = mean_bce(model, arrays[h].X_val, arrays[h].y_val)
        row["cumulative_steps"] = steps
        return row

    history = [val_row(0, np.nan, 0)]
    best = {"epoch": None, "val_loss": np.inf, "state": None}
    total_steps = 0
    for epoch in range(1, max_epochs + 1):
        loader = make_loader(X, y, batch_size, derive_seed(seed, "centralized", epoch))
        train_loss, steps = train_epoch(model, loader, optimizer)
        total_steps += steps
        row = val_row(epoch, train_loss, total_steps)
        history.append(row)
        if row["val_loss"] < best["val_loss"]:
            state = clone_state(model.state_dict())
            best = {"epoch": epoch, "val_loss": row["val_loss"], "state": state}

    return {
        "history": pd.DataFrame(history),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_state": best["state"],
        "final_state": clone_state(model.state_dict()),
        "budget": {
            "epochs_trained": max_epochs,
            "optimizer_steps": total_steps,
            "examples_processed": int(len(y) * max_epochs),
            "n_train": int(len(y)),
            "n_val": int(len(yv)),
        },
    }
