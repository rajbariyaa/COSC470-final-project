import numpy as np
import torch

from fedheart.engine import mean_bce
from fedheart.finetune import finetune_hospital
from fedheart.model import build_model, initial_state
from fedheart.prepare import HospitalArrays


def _data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 41)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    return HospitalArrays("va", np.arange(40).astype(str), X[:40], y[:40],
                          np.arange(20).astype(str), X[40:], y[40:])  # fmt: skip


def test_epoch_zero_is_the_global_model_and_global_is_not_mutated():
    data = _data()
    g = initial_state(41, 16, 0)
    before = {k: v.clone() for k, v in g.items()}
    res = finetune_hospital(g, data, lr=1e-3, max_epochs=5, batch_size=16, seed=0, hidden=16)
    assert all(torch.equal(before[k], g[k]) for k in g)
    expected = mean_bce(build_model(g, 41, 16), data.X_val, data.y_val)
    assert res["history"].loc[0, "val_loss"] == expected
    assert len(res["history"]) == 6


def test_selected_epoch_has_lowest_validation_loss():
    data = _data()
    res = finetune_hospital(initial_state(41, 16, 0), data, lr=1e-2, max_epochs=10,
                            batch_size=16, seed=0, hidden=16)  # fmt: skip
    h = res["history"]
    assert res["best_epoch"] == int(h.loc[h["val_loss"].idxmin(), "epoch"])
    model = build_model(res["best_state"], 41, 16)
    assert mean_bce(model, data.X_val, data.y_val) == res["best_val_loss"]
