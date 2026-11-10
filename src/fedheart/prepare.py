"""Per-seed preparation: split, federated preprocessing fit, initial weights, client arrays.

Only training and validation arrays are materialized here. Test arrays can only be built
through `fedheart.testset`, which refuses until model selection has been locked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fedheart.data import HOSPITALS
from fedheart.model import initial_state
from fedheart.preprocess import N_FEATURES, FederatedPreprocessor, fit_federated
from fedheart.splits import load_split, make_split, save_split, split_summary
from fedheart.utils import results_dir, seed_tag, write_json

DEVELOPMENT_PARTITIONS = ("train", "val")


@dataclass
class HospitalArrays:
    hospital: str
    ids_train: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    ids_val: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_val(self) -> int:
        return len(self.y_val)


def seed_dir(cfg: dict[str, Any], seed: int) -> Path:
    return results_dir(cfg) / "seeds" / seed_tag(seed)


def partition_frame(df: pd.DataFrame, split: pd.DataFrame, hospital: str, partition: str):
    """Records of one hospital and partition, with the partition label attached."""
    ids = split.loc[(split["hospital"] == hospital) & (split["partition"] == partition)]
    out = df.merge(ids[["record_id", "partition"]], on="record_id", how="inner")
    return out.sort_values("row_index").reset_index(drop=True)


def prepare_seed(cfg: dict[str, Any], df: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Create and save everything one seed needs before training."""
    out = seed_dir(cfg, seed)
    split = make_split(df, seed, cfg["split"]["fractions"])
    save_split(split, out / "split.csv")

    train_frames = {h: partition_frame(df, split, h, "train") for h in HOSPITALS}
    pre, stats = fit_federated(train_frames)
    pre.save(out)
    write_json([s.__dict__ for s in stats], out / "client_statistics.json")

    init = initial_state(N_FEATURES, cfg["model"]["hidden_units"], seed)
    torch.save(init, out / "init.pt")

    for h in HOSPITALS:
        arrays = build_hospital_arrays(df, split, pre, h)
        save_client_arrays(arrays, out / "client_data" / f"{h}.npz")
    return {"split": split, "preprocessor": pre, "summary": split_summary(split, seed)}


def build_hospital_arrays(
    df: pd.DataFrame, split: pd.DataFrame, pre: FederatedPreprocessor, hospital: str
) -> HospitalArrays:
    parts = {p: partition_frame(df, split, hospital, p) for p in DEVELOPMENT_PARTITIONS}
    return HospitalArrays(
        hospital=hospital,
        ids_train=parts["train"]["record_id"].to_numpy(dtype=str),
        X_train=pre.transform(parts["train"]),
        y_train=parts["train"]["target"].to_numpy(dtype=np.int64),
        ids_val=parts["val"]["record_id"].to_numpy(dtype=str),
        X_val=pre.transform(parts["val"]),
        y_val=parts["val"]["target"].to_numpy(dtype=np.int64),
    )


def save_client_arrays(a: HospitalArrays, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, hospital=a.hospital, ids_train=a.ids_train, X_train=a.X_train, y_train=a.y_train,
        ids_val=a.ids_val, X_val=a.X_val, y_val=a.y_val,
    )  # fmt: skip


def load_client_arrays(path: str | Path) -> HospitalArrays:
    with np.load(path, allow_pickle=False) as z:
        return HospitalArrays(
            hospital=str(z["hospital"]), ids_train=z["ids_train"], X_train=z["X_train"],
            y_train=z["y_train"], ids_val=z["ids_val"], X_val=z["X_val"], y_val=z["y_val"],
        )  # fmt: skip


def load_seed(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Load a prepared seed: split, preprocessor, initial weights and train/val arrays."""
    d = seed_dir(cfg, seed)
    return {
        "dir": d,
        "split": load_split(d / "split.csv"),
        "preprocessor": FederatedPreprocessor.load(d),
        "init": torch.load(d / "init.pt"),
        "arrays": {h: load_client_arrays(d / "client_data" / f"{h}.npz") for h in HOSPITALS},
    }
