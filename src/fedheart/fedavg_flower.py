"""FedAvg with Flower 1.43 (Message API), run as a local simulation.

* One SuperNode per hospital; `context.node_config["partition-id"]` (0-3) selects the
  hospital in `data.HOSPITALS` order. Each ClientApp reads only its own hospital's
  train/validation arrays.
* Every round all four hospitals train for `local_epochs` epochs from the current global
  model with a fresh Adam optimizer, and report their training-set size as
  "num-examples"; Flower's `FedAvg` uses that key to weight the average (sample-count
  weighting).
* After aggregation, every hospital evaluates the new global model on its validation
  partition; the sample-weighted mean of these losses equals the pooled validation loss and
  is used to select the checkpoint round (ties -> earliest round).
* The server-side `evaluate_fn` hook only snapshots the aggregated weights of each round so
  the selected round can be restored; it never touches patient data.
* Flower drops failed client replies and aggregates the rest. A partial round would silently
  change the method, so `_collect` raises unless every round has all four replies.

Launch: `flwr.simulation.run_simulation` is used so the experiment loop stays in one Python
process. Flower 1.43 marks it as deprecated in favour of `flwr run`; the ServerApp and
ClientApp below are unchanged under either launcher.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Imported at module level on purpose: ArrayRecord.to_torch_state_dict() looks torch up in
# sys.modules, which must already contain it inside the Ray worker processes.
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.simulation import run_simulation

from fedheart.data import HOSPITALS
from fedheart.engine import local_train, mean_bce
from fedheart.model import build_model, clone_state
from fedheart.prepare import HospitalArrays, load_client_arrays
from fedheart.utils import derive_seed

os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

DEFAULT_BACKEND_CONFIG = {
    "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
    "init_args": {"logging_level": "ERROR", "log_to_driver": False},
}
BYTES_PER_PARAMETER = 4  # float32


class FederatedRunError(RuntimeError):
    """Raised when a Flower run does not complete every round with every hospital."""


# --------------------------------------------------------------------------------------
# ClientApp (runs once per hospital)
# --------------------------------------------------------------------------------------

client_app = ClientApp()
_CACHE: dict[str, HospitalArrays] = {}


def _client_data(context: Context, cfg: ConfigRecord) -> tuple[int, HospitalArrays]:
    pid = int(context.node_config["partition-id"])
    path = os.path.join(str(cfg["data-dir"]), f"{HOSPITALS[pid]}.npz")
    if path not in _CACHE:
        _CACHE[path] = load_client_arrays(path)
    return pid, _CACHE[path]


@client_app.train()
def client_train(msg: Message, context: Context) -> Message:
    cfg = msg.content["config"]
    pid, data = _client_data(context, cfg)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed, server_round = int(cfg["seed"]), int(cfg["server-round"])
    shuffle_seeds = [
        derive_seed(seed, "fedavg", data.hospital, server_round, e)
        for e in range(int(cfg["local-epochs"]))
    ]
    state, loss, steps = local_train(
        msg.content["arrays"].to_torch_state_dict(),
        data.X_train,
        data.y_train,
        lr=float(cfg["lr"]),
        batch_size=int(cfg["batch-size"]),
        shuffle_seeds=shuffle_seeds,
        hidden=int(cfg["hidden"]),
    )
    metrics = MetricRecord(
        {"train_loss": loss, "num-examples": data.n_train, "partition-id": pid, "steps": steps}
    )
    content = RecordDict({"arrays": ArrayRecord(state), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@client_app.evaluate()
def client_evaluate(msg: Message, context: Context) -> Message:
    cfg = msg.content["config"]
    pid, data = _client_data(context, cfg)
    model = build_model(
        msg.content["arrays"].to_torch_state_dict(), data.X_val.shape[1], int(cfg["hidden"])
    )
    metrics = MetricRecord(
        {"val_loss": mean_bce(model, data.X_val, data.y_val), "num-examples": data.n_val,
         "partition-id": pid}
    )  # fmt: skip
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)


# --------------------------------------------------------------------------------------
# Metric aggregation (keeps per-hospital values and counts replies)
# --------------------------------------------------------------------------------------


def _reply_metrics(records: list[RecordDict]) -> list[dict]:
    return [dict(next(iter(r.metric_records.values()))) for r in records]


def _aggregate(records: list[RecordDict], weight_key: str, prefix: str) -> MetricRecord:
    """Sample-weighted mean loss plus per-hospital values; keys start with `prefix`."""
    rows = _reply_metrics(records)
    loss_key = f"{prefix}_loss"
    w = np.array([float(r[weight_key]) for r in rows])
    losses = np.array([float(r[loss_key]) for r in rows])
    agg: dict[str, float | int] = {
        loss_key: float(np.dot(w, losses) / w.sum()),
        f"{prefix}_n_replies": len(rows),
        f"{prefix}_n_total": int(w.sum()),
    }
    if "steps" in rows[0]:
        agg["steps_total"] = int(sum(int(r["steps"]) for r in rows))
    for r in rows:
        h = HOSPITALS[int(r["partition-id"])]
        agg[f"{loss_key}_{h}"] = float(r[loss_key])
        agg[f"{prefix}_n_{h}"] = int(r[weight_key])
    return MetricRecord(agg)


def aggregate_train_metrics(records: list[RecordDict], weight_key: str) -> MetricRecord:
    return _aggregate(records, weight_key, "train")


def aggregate_evaluate_metrics(records: list[RecordDict], weight_key: str) -> MetricRecord:
    return _aggregate(records, weight_key, "val")


# --------------------------------------------------------------------------------------
# ServerApp + launcher
# --------------------------------------------------------------------------------------


def run_fedavg(
    *,
    data_dir: str | Path,
    init_state: dict[str, torch.Tensor],
    lr: float,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    seed: int,
    hidden: int,
    backend_config: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Run one FedAvg experiment in Flower simulation and return history + selected weights."""
    num_clients = len(HOSPITALS)
    captured: dict = {}
    server_app = ServerApp()

    @server_app.main()
    def main(grid: Grid, context: Context) -> None:
        snapshots: dict[int, dict[str, torch.Tensor]] = {}

        def snapshot_global(server_round: int, arrays: ArrayRecord) -> None:
            snapshots[server_round] = clone_state(arrays.to_torch_state_dict())
            return None

        strategy = FedAvg(
            fraction_train=1.0,
            fraction_evaluate=1.0,
            min_train_nodes=num_clients,
            min_evaluate_nodes=num_clients,
            min_available_nodes=num_clients,
            weighted_by_key="num-examples",
            train_metrics_aggr_fn=aggregate_train_metrics,
            evaluate_metrics_aggr_fn=aggregate_evaluate_metrics,
        )
        config = ConfigRecord({
            "data-dir": str(data_dir), "seed": int(seed), "lr": float(lr),
            "batch-size": int(batch_size), "local-epochs": int(local_epochs),
            "hidden": int(hidden),
        })  # fmt: skip
        captured["result"] = strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(clone_state(init_state)),
            num_rounds=num_rounds,
            train_config=config,
            evaluate_config=config,
            evaluate_fn=snapshot_global,
        )
        captured["snapshots"] = snapshots

    flwr_logger = logging.getLogger("flwr")
    previous_level = flwr_logger.level
    if not verbose:
        flwr_logger.setLevel(logging.WARNING)
    try:
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=num_clients,
            backend_config=backend_config or DEFAULT_BACKEND_CONFIG,
        )
    finally:
        flwr_logger.setLevel(previous_level)
    return _collect(captured, num_rounds, num_clients, local_epochs, init_state)


def _collect(captured: dict, num_rounds: int, num_clients: int, local_epochs: int, init_state):
    if "result" not in captured:
        raise FederatedRunError("the Flower ServerApp did not finish; see the log above")
    result, snapshots = captured["result"], captured["snapshots"]
    rows = []
    for r in range(1, num_rounds + 1):
        tm = result.train_metrics_clientapp.get(r)
        em = result.evaluate_metrics_clientapp.get(r)
        n_train = int(tm["train_n_replies"]) if tm is not None else 0
        n_eval = int(em["val_n_replies"]) if em is not None else 0
        if n_train != num_clients or n_eval != num_clients or r not in snapshots:
            raise FederatedRunError(
                f"round {r}: {n_train} training and {n_eval} evaluation replies, "
                f"expected {num_clients} of each"
            )
        rows.append({"round": r, **dict(tm), **dict(em)})
    history = pd.DataFrame(rows)
    best_round = int(history.loc[history["val_loss"].idxmin(), "round"])

    n_params = sum(v.numel() for v in init_state.values())
    per_round_bytes = n_params * BYTES_PER_PARAMETER * num_clients
    budget = {
        "rounds": num_rounds,
        "local_epochs": local_epochs,
        "clients_per_round": num_clients,
        "optimizer_steps": int(history["steps_total"].sum()),
        "examples_processed": int(history["train_n_total"].sum()),
        "n_train": int(history["train_n_total"].iloc[0]),
        "n_val": int(history["val_n_total"].iloc[0]),
        "n_parameters": n_params,
        # Training: global model down + update up per client; evaluation: model down.
        "bytes_training": int(2 * per_round_bytes * num_rounds),
        "bytes_evaluation": int(per_round_bytes * num_rounds),
    }
    return {
        "history": history,
        "best_round": best_round,
        "best_val_loss": float(history["val_loss"].min()),
        "best_state": snapshots[best_round],
        "final_state": snapshots[num_rounds],
        "budget": budget,
    }
