import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from fedheart.data import HOSPITALS
from fedheart.fedavg_flower import (
    FederatedRunError,
    _collect,
    aggregate_evaluate_metrics,
    run_fedavg,
)
from fedheart.fedavg_reference import run_fedavg_reference, weighted_average
from fedheart.model import initial_state
from fedheart.prepare import load_seed, prepare_seed, seed_dir


def _states():
    return [initial_state(41, 16, s) for s in range(4)], [181, 175, 73, 120]


def test_weighted_average_uses_sample_counts():
    a = {"w": torch.tensor([1.0, 0.0])}
    b = {"w": torch.tensor([3.0, 4.0])}
    out = weighted_average([(a, 1), (b, 3)])
    np.testing.assert_allclose(out["w"].numpy(), [2.5, 3.0])


def test_flower_aggregation_weights_by_num_examples():
    """Flower's own FedAvg aggregation equals the sample-count weighted average."""
    states, counts = _states()
    records = [
        RecordDict({"arrays": ArrayRecord(s), "metrics": MetricRecord({"num-examples": n})})
        for s, n in zip(states, counts, strict=True)
    ]
    flower = aggregate_arrayrecords(records, "num-examples").to_torch_state_dict()
    ours = weighted_average(list(zip(states, counts, strict=True)))
    for k in ours:
        np.testing.assert_allclose(flower[k].numpy(), ours[k].numpy(), rtol=0, atol=1e-7)


def test_evaluation_aggregate_is_pooled_loss_and_counts_replies():
    records = [
        RecordDict({"metrics": MetricRecord(
            {"val_loss": float(i + 1), "num-examples": n, "partition-id": i})})
        for i, n in enumerate([10, 30, 0 + 5, 5])
    ]  # fmt: skip
    m = aggregate_evaluate_metrics(records, "num-examples")
    assert m["val_n_replies"] == 4
    assert m["val_loss"] == pytest.approx((1 * 10 + 2 * 30 + 3 * 5 + 4 * 5) / 50)
    assert m["val_loss_switzerland"] == 3.0


def test_round_with_missing_client_reply_raises():
    """Flower aggregates whatever replies arrive; a partial round must not pass silently."""
    result = Result()
    result.train_metrics_clientapp[1] = MetricRecord({"train_n_replies": 3})
    result.evaluate_metrics_clientapp[1] = MetricRecord({"val_n_replies": 4})
    init = initial_state(41, 16, 0)
    with pytest.raises(FederatedRunError, match="3 training"):
        _collect({"result": result, "snapshots": {1: init}}, 1, 4, 1, init)
    with pytest.raises(FederatedRunError, match="did not finish"):
        _collect({}, 1, 4, 1, init)


@pytest.mark.slow
def test_flower_run_matches_reference_fedavg(tmp_cfg, dataset):
    """End-to-end: Flower simulation reproduces an independent FedAvg loop."""
    prepare_seed(tmp_cfg, dataset, 0)
    seed = load_seed(tmp_cfg, 0)
    kw = {"lr": 1e-3, "num_rounds": 3, "local_epochs": 1, "batch_size": 16, "seed": 0, "hidden": 16}
    data_dir = seed_dir(tmp_cfg, 0) / "client_data"
    flower = run_fedavg(data_dir=data_dir, init_state=seed["init"], **kw)
    ref = run_fedavg_reference(seed["arrays"], seed["init"], **kw)
    for k, v in ref["states"][3].items():
        np.testing.assert_allclose(flower["final_state"][k].numpy(), v.numpy(), atol=1e-5)
    np.testing.assert_allclose(
        flower["history"]["val_loss"].to_numpy(), ref["history"]["val_loss"].to_numpy(), atol=1e-5
    )
    hist = flower["history"]
    for h in HOSPITALS:
        assert (hist[f"train_n_{h}"] == seed["arrays"][h].n_train).all()  # aggregation weights
        assert (hist[f"val_n_{h}"] == seed["arrays"][h].n_val).all()
    assert (hist["train_n_replies"] == 4).all() and (hist["val_n_replies"] == 4).all()
    assert flower["budget"]["n_train"] == sum(seed["arrays"][h].n_train for h in HOSPITALS)
