import pandas as pd
import pytest

from fedheart.experiment import select_lr, validation_by_lr


def _runs(values):
    return pd.DataFrame(
        [{"lr": lr, "seed": s, "best_val_loss": v} for lr, vs in values.items()
         for s, v in enumerate(vs)]
    )  # fmt: skip


def test_selects_lowest_mean_validation_loss():
    runs = _runs({1e-3: [0.5, 0.5], 3e-3: [0.4, 0.45]})
    lr, table = select_lr(runs, "fedavg", [1e-3, 3e-3], [0, 1])
    assert lr == 3e-3
    assert table.set_index("lr").loc[3e-3, "selected"]


def test_ties_go_to_the_earlier_grid_entry():
    runs = _runs({3e-3: [0.4, 0.4], 1e-3: [0.4, 0.4]})
    lr, _ = select_lr(runs, "centralized", [1e-3, 3e-3], [0, 1])
    assert lr == 1e-3


def test_missing_runs_are_refused():
    with pytest.raises(RuntimeError, match="missing"):
        select_lr(_runs({1e-3: [0.5], 3e-3: [0.4, 0.4]}), "fedavg", [1e-3, 3e-3], [0, 1])


def test_finetune_loss_is_pooled_by_validation_size():
    runs = pd.DataFrame([
        {"lr": 1e-4, "seed": 0, "hospital": "a", "best_val_loss": 1.0, "budget_n_val": 30},
        {"lr": 1e-4, "seed": 0, "hospital": "b", "best_val_loss": 0.0, "budget_n_val": 10},
    ])  # fmt: skip
    table = validation_by_lr(runs, "fedavg_ft")
    assert table.loc[0, "val_loss_mean"] == pytest.approx(0.75)
