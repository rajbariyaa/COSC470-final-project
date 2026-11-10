"""Run bookkeeping and validation-only model selection shared by the pipeline scripts.

Selection rules (applied identically to every method, using validation data only):
* Centralized: for each learning rate, the best pooled validation loss of each seed's run is
  averaged over seeds; the learning rate with the lowest mean wins.
* FedAvg: same, using the sample-weighted validation loss reported by the four hospitals
  (numerically the pooled validation loss).
* FedAvg with local fine-tuning: for each learning rate and seed, the hospitals' selected
  validation losses are combined into a pooled loss (weighted by validation size) and then
  averaged over seeds.
Ties go to the earlier entry of the grid in the config. The epoch/round (or fine-tuning
epoch) of the checkpoint is chosen per seed (and per hospital) inside each run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from fedheart.utils import lr_tag, read_json, results_dir, seed_tag, write_json

RUN_METHODS = ("centralized", "fedavg", "fedavg_ft")


def runs_root(cfg: dict[str, Any]) -> Path:
    return results_dir(cfg) / "runs"


def run_dir(cfg: dict[str, Any], method: str, lr: float, seed: int, hospital: str | None = None):
    d = runs_root(cfg) / method / lr_tag(lr) / seed_tag(seed)
    return d / hospital if hospital else d


def is_done(out: Path) -> bool:
    return (out / "summary.json").exists()


def save_run(out: Path, result: dict, meta: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    result["history"].to_csv(out / "history.csv", index=False)
    torch.save(result["best_state"], out / "best.pt")
    keep = ("best_epoch", "best_round", "best_val_loss", "budget")
    summary = {**meta, **{k: result[k] for k in keep if k in result}}
    write_json(summary, out / "summary.json")
    return summary


def collect_summaries(cfg: dict[str, Any], method: str) -> pd.DataFrame:
    pattern = "*/seed_*/*/summary.json" if method == "fedavg_ft" else "*/seed_*/summary.json"
    rows = []
    for path in sorted((runs_root(cfg) / method).glob(pattern)):
        s = read_json(path)
        budget = s.pop("budget", {})
        rows.append({**s, **{f"budget_{k}": v for k, v in budget.items()}, "dir": str(path.parent)})
    return pd.DataFrame(rows)


def validation_by_lr(summaries: pd.DataFrame, method: str) -> pd.DataFrame:
    """Per learning rate: mean and SD over seeds of the (pooled) selected validation loss."""
    s = summaries
    if method == "fedavg_ft":
        s = s.assign(weighted=s["best_val_loss"] * s["budget_n_val"])
        per_seed = s.groupby(["lr", "seed"]).agg(w=("weighted", "sum"), n=("budget_n_val", "sum"))
        per_seed = (per_seed["w"] / per_seed["n"]).rename("val_loss").reset_index()
    else:
        per_seed = s[["lr", "seed", "best_val_loss"]].rename(columns={"best_val_loss": "val_loss"})
    out = per_seed.groupby("lr")["val_loss"].agg(["mean", "std", "count"]).reset_index()
    return out.rename(columns={"mean": "val_loss_mean", "std": "val_loss_sd", "count": "n_seeds"})


def select_lr(summaries: pd.DataFrame, method: str, grid: list[float], seeds: list[int]):
    """Learning rate with the lowest mean validation loss; requires every run to exist."""
    table = validation_by_lr(summaries, method)
    complete = table["n_seeds"] == len(seeds)
    if set(table["lr"]) != set(grid) or not complete.all():
        raise RuntimeError(f"{method}: runs are missing for some learning rates or seeds")
    order = {lr: i for i, lr in enumerate(grid)}
    table = table.assign(grid_order=table["lr"].map(order)).sort_values("grid_order")
    best = table.loc[table["val_loss_mean"].idxmin()]
    lowest = table[table["val_loss_mean"] == best["val_loss_mean"]].iloc[0]
    table = table.drop(columns="grid_order").assign(selected=lambda t: t["lr"] == lowest["lr"])
    return float(lowest["lr"]), table.reset_index(drop=True)
