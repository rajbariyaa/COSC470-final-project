"""Plain-PyTorch FedAvg used only to cross-check the Flower implementation in tests.

It applies the same client update (`engine.local_train`, same shuffle seeds) and the same
sample-count weighted average, computed in float32 like Flower's `aggregate_arrayrecords`.
Reported results always come from the Flower run, never from this module.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from fedheart.data import HOSPITALS
from fedheart.engine import local_train, mean_bce
from fedheart.model import build_model, clone_state
from fedheart.prepare import HospitalArrays
from fedheart.utils import derive_seed


def weighted_average(updates: list[tuple[dict[str, torch.Tensor], int]]) -> OrderedDict:
    total = sum(n for _, n in updates)
    acc: OrderedDict[str, np.ndarray] = OrderedDict()
    for state, n in updates:
        w = n / total
        for k, v in state.items():
            contrib = v.numpy() * w  # float32 * Python float stays float32 (NEP 50)
            acc[k] = contrib if k not in acc else acc[k] + contrib
    return OrderedDict((k, torch.from_numpy(np.ascontiguousarray(v))) for k, v in acc.items())


def run_fedavg_reference(
    arrays: dict[str, HospitalArrays],
    init_state: dict[str, torch.Tensor],
    *,
    lr: float,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    seed: int,
    hidden: int,
) -> dict:
    global_state = clone_state(init_state)
    states = {0: global_state}
    rows = []
    for r in range(1, num_rounds + 1):
        updates = []
        for h in HOSPITALS:
            d = arrays[h]
            seeds = [derive_seed(seed, "fedavg", h, r, e) for e in range(local_epochs)]
            state, _, _ = local_train(
                global_state, d.X_train, d.y_train,
                lr=lr, batch_size=batch_size, shuffle_seeds=seeds, hidden=hidden,
            )  # fmt: skip
            updates.append((state, d.n_train))
        global_state = weighted_average(updates)
        states[r] = global_state
        model = build_model(global_state, arrays[HOSPITALS[0]].X_val.shape[1], hidden)
        losses = {h: mean_bce(model, arrays[h].X_val, arrays[h].y_val) for h in HOSPITALS}
        n_val = {h: arrays[h].n_val for h in HOSPITALS}
        pooled = sum(losses[h] * n_val[h] for h in HOSPITALS) / sum(n_val.values())
        per_hospital = {f"val_loss_{h}": losses[h] for h in HOSPITALS}
        rows.append({"round": r, "val_loss": pooled, **per_hospital})
    return {"history": pd.DataFrame(rows), "states": states}
