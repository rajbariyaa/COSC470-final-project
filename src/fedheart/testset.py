"""The test partitions stay sealed until model selection is locked.

`06_lock_selection.py` writes `selection.json` and a lock file holding its SHA-256. Test
arrays can only be produced by `load_test_arrays`, which refuses to run unless the lock is
present and matches, and appends every access to `test_access_log.jsonl`. Nothing in the
training, tuning or selection code imports this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fedheart.data import HOSPITALS
from fedheart.prepare import partition_frame, seed_dir
from fedheart.preprocess import FederatedPreprocessor
from fedheart.splits import load_split
from fedheart.utils import now_iso, read_json, results_dir, sha256_file, write_json

SELECTION_FILE = "selection.json"
LOCK_FILE = "selection.lock.json"
ACCESS_LOG = "test_access_log.jsonl"


class SealedTestSetError(RuntimeError):
    """Raised when test data are requested before model selection is locked."""


def lock_selection(cfg: dict[str, Any], selection: dict, force: bool = False) -> Path:
    rdir = results_dir(cfg)
    if (rdir / ACCESS_LOG).exists() and not force:
        raise SealedTestSetError(
            "test data have already been accessed; re-locking a different selection would "
            "let test results influence model choice (pass force=True only to start over)"
        )
    path = rdir / SELECTION_FILE
    write_json(selection, path)
    write_json({"sha256": sha256_file(path), "locked_at": now_iso()}, rdir / LOCK_FILE)
    return path


def read_locked_selection(cfg: dict[str, Any]) -> dict:
    rdir = results_dir(cfg)
    lock, path = rdir / LOCK_FILE, rdir / SELECTION_FILE
    if not lock.exists() or not path.exists():
        raise SealedTestSetError("model selection is not locked; run 06_lock_selection.py first")
    if sha256_file(path) != read_json(lock)["sha256"]:
        raise SealedTestSetError("selection.json changed after it was locked")
    return read_json(path)


def load_test_arrays(
    cfg: dict[str, Any], df: pd.DataFrame, seed: int, purpose: str
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(record_ids, X, y) of each hospital's test partition for one seed."""
    selection = read_locked_selection(cfg)
    d = seed_dir(cfg, seed)
    split = load_split(d / "split.csv")
    pre = FederatedPreprocessor.load(d)
    out = {}
    for h in HOSPITALS:
        part = partition_frame(df, split, h, "test")
        out[h] = (
            part["record_id"].to_numpy(dtype=str),
            pre.transform(part),
            part["target"].to_numpy(dtype=np.int64),
        )
    entry = {
        "time": now_iso(), "seed": seed, "purpose": purpose,
        "selection_sha256": sha256_file(results_dir(cfg) / SELECTION_FILE),
        "selection_locked_at": selection.get("locked_at"),
    }  # fmt: skip
    with open(results_dir(cfg) / ACCESS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return out
