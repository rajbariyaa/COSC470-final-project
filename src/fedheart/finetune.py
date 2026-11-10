"""FedAvg with local fine-tuning.

After FedAvg, each hospital independently fine-tunes its own *copy* of the selected global
checkpoint on its training partition (fresh Adam state) and keeps the epoch in 0..max_epochs
with the lowest loss on its own validation partition. Epoch 0 is the unmodified global model,
so a hospital whose validation data do not favour fine-tuning keeps the FedAvg model.

This is plain post-hoc fine-tuning. It is not Ditto, FedPer, or any personalization-layer
method: no regularization toward the global model and no layer splitting is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from fedheart.engine import make_loader, mean_bce, train_epoch
from fedheart.model import build_model, clone_state
from fedheart.prepare import HospitalArrays
from fedheart.utils import derive_seed


def finetune_hospital(
    global_state: dict[str, torch.Tensor],
    data: HospitalArrays,
    *,
    lr: float,
    max_epochs: int,
    batch_size: int,
    seed: int,
    hidden: int,
) -> dict:
    model = build_model(clone_state(global_state), data.X_train.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    v0 = mean_bce(model, data.X_val, data.y_val)
    history = [{"epoch": 0, "train_loss": np.nan, "val_loss": v0, "cumulative_steps": 0}]
    best = {"epoch": 0, "val_loss": v0, "state": clone_state(model.state_dict())}
    total_steps = 0
    for epoch in range(1, max_epochs + 1):
        seed_e = derive_seed(seed, "finetune", data.hospital, epoch)
        train_loss, steps = train_epoch(
            model, make_loader(data.X_train, data.y_train, batch_size, seed_e), optimizer
        )
        total_steps += steps
        v = mean_bce(model, data.X_val, data.y_val)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": v,
            "cumulative_steps": total_steps,
        })  # fmt: skip
        if v < best["val_loss"]:
            best = {"epoch": epoch, "val_loss": v, "state": clone_state(model.state_dict())}

    return {
        "history": pd.DataFrame(history),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_state": best["state"],
        "budget": {
            "epochs_trained": max_epochs,
            "optimizer_steps": total_steps,
            "examples_processed": int(data.n_train * max_epochs),
            "n_train": data.n_train,
            "n_val": data.n_val,
        },
    }
