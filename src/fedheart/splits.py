"""Reproducible, stratified, duplicate-aware train/validation/test partitions per hospital.

Each hospital is split on its own (data never leave the hospital to be split). The unit of
splitting is the duplicate group from `data.duplicate_groups`, so exact copies of a record
always land in the same partition. Stratification is on the binary target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedheart.data import HOSPITALS
from fedheart.utils import derive_seed

PARTITIONS: tuple[str, ...] = ("train", "val", "test")
# Group-level splitting can move a partition by up to one duplicate group from the ideal
# row count; the largest group here has 2 rows.
ROW_TOLERANCE = 2


class SplitError(ValueError):
    """Raised when a split violates a required property."""


def _fractions(fractions: dict[str, float]) -> tuple[float, float, float]:
    f = (fractions["train"], fractions["val"], fractions["test"])
    if not np.isclose(sum(f), 1.0) or min(f) <= 0:
        raise SplitError(f"split fractions must be positive and sum to 1, got {f}")
    return f


def minority_class_feasibility(df: pd.DataFrame, fractions: dict[str, float]) -> pd.DataFrame:
    """Expected minority-class count per partition; a split is feasible if each is >= 1."""
    f = _fractions(fractions)
    rows = []
    for h in HOSPITALS:
        y = df.loc[df["hospital"] == h, "target"]
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        minority, n_min = ("negative", n_neg) if n_neg < n_pos else ("positive", n_pos)
        expected = {p: n_min * fp for p, fp in zip(PARTITIONS, f, strict=True)}
        rows.append({
            "hospital": h, "n_records": len(y), "n_positive": n_pos, "n_negative": n_neg,
            "minority_class": minority, "n_minority": n_min,
            **{f"expected_minority_{p}": round(v, 2) for p, v in expected.items()},
            "feasible": all(v >= 1 for v in expected.values()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def make_split(df: pd.DataFrame, seed: int, fractions: dict[str, float]) -> pd.DataFrame:
    """Assign every record to train/val/test. Returns one row per record."""
    f_train, f_val, f_test = _fractions(fractions)
    parts = []
    for h in HOSPITALS:
        sub = df[df["hospital"] == h]
        groups = sub.groupby("group_id", sort=True)["target"].agg(["first", "nunique"])
        if (groups["nunique"] > 1).any():
            raise SplitError(f"{h}: a duplicate group has conflicting labels")
        y = groups["first"]
        rs = derive_seed(seed, "split", h)
        g_train, g_hold = train_test_split(
            groups.index.to_numpy(), test_size=f_val + f_test, stratify=y, random_state=rs
        )
        g_val, g_test = train_test_split(
            g_hold, test_size=f_test / (f_val + f_test), stratify=y.loc[g_hold], random_state=rs
        )
        assignment = {g: "train" for g in g_train} | {g: "val" for g in g_val}
        assignment |= {g: "test" for g in g_test}
        parts.append(
            sub[["record_id", "hospital", "row_index", "group_id", "target"]].assign(
                partition=sub["group_id"].map(assignment)
            )
        )
    split = pd.concat(parts, ignore_index=True)
    validate_split(split, df, fractions)
    return split


def validate_split(split: pd.DataFrame, df: pd.DataFrame, fractions: dict[str, float]) -> None:
    """Raise SplitError unless the split is complete, disjoint, group-safe, stratified enough."""
    f = dict(zip(PARTITIONS, _fractions(fractions), strict=True))
    if split["record_id"].duplicated().any():
        raise SplitError("a record is assigned to more than one partition")
    if set(split["record_id"]) != set(df["record_id"]):
        raise SplitError("split does not cover exactly the dataset's records")
    if not split["partition"].isin(PARTITIONS).all():
        raise SplitError("unknown or missing partition label")
    if (split.groupby("group_id")["partition"].nunique() > 1).any():
        raise SplitError("a duplicate group crosses partitions")
    for h in HOSPITALS:
        sub = split[split["hospital"] == h]
        for p in PARTITIONS:
            part = sub[sub["partition"] == p]
            if part["target"].nunique() < 2:
                raise SplitError(f"{h}/{p} does not contain both classes")
            if abs(len(part) - f[p] * len(sub)) > ROW_TOLERANCE + 1:
                raise SplitError(f"{h}/{p} has {len(part)} rows, far from {f[p]:.0%}")


def split_summary(split: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Row and class counts per hospital and partition."""
    g = split.groupby(["hospital", "partition"])["target"]
    out = pd.DataFrame({"n": g.size(), "n_positive": g.sum()}).reset_index()
    out["n_negative"] = out["n"] - out["n_positive"]
    out.insert(0, "seed", seed)
    return out


def save_split(split: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    split.sort_values(["hospital", "row_index"]).to_csv(path, index=False)


def load_split(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
