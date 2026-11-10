"""Federated preprocessing with a fixed feature schema.

Protocol (chosen because it is feasible without pooling raw records):

1. The categorical encoding is fixed in advance from the UCI documentation
   (`data.CATEGORICAL_LEVELS`) plus one explicit "missing" level per variable, so it is
   not learned from any data and is identical at every hospital.
2. For each numeric feature every hospital computes, on its *training partition only*, the
   number of observed values, their sum and their sum of squares (`ClientStatistics`).
   Only these aggregates are sent to the coordinator.
3. The coordinator adds them up to obtain the mean and the (population) standard deviation.
   Because sums are additive, these equal exactly what a scaler fitted on the pooled
   training rows would produce; the centralized model therefore receives the very same
   representation, and tests verify the equality against scikit-learn.
4. Transform: missing numeric value -> mean imputation (0 after standardization) plus a
   missing-indicator column; categorical -> one-hot over the fixed levels plus "missing".

Every hospital therefore produces the same 41 columns in the same order, including
Switzerland, where cholesterol is never observed.

Limitations (also in docs/LIMITATIONS.md): the shared sums are still information about the
hospital's patients (no secure aggregation or differential privacy is used); mean rather
than median imputation is used because medians cannot be combined from sums; and the
missing-indicator pattern differs strongly between hospitals, so it is an indirect signal of
the site even though hospital identity is never an input.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fedheart.data import CATEGORICAL_LEVELS, NUMERIC_FEATURES
from fedheart.utils import read_json, write_json


def feature_names() -> list[str]:
    names = list(NUMERIC_FEATURES) + [f"{c}__missing" for c in NUMERIC_FEATURES]
    for c, levels in CATEGORICAL_LEVELS.items():
        names += [f"{c}={lv}" for lv in levels] + [f"{c}=missing"]
    return names


FEATURE_NAMES: tuple[str, ...] = tuple(feature_names())
N_FEATURES = len(FEATURE_NAMES)


def _require_training_rows(df: pd.DataFrame) -> None:
    if "partition" not in df.columns:
        raise ValueError("statistics must be computed from rows that carry a 'partition' column")
    bad = set(df["partition"]) - {"train"}
    if bad:
        raise ValueError(f"preprocessing may only be fitted on training rows, got {bad}")


@dataclass(frozen=True)
class ClientStatistics:
    """What one hospital shares with the coordinator (training partition only)."""

    hospital: str
    n_rows: int
    n_observed: dict[str, int]
    sum: dict[str, float]
    sum_sq: dict[str, float]


def compute_client_statistics(train_df: pd.DataFrame, hospital: str) -> ClientStatistics:
    _require_training_rows(train_df)
    n_obs, s, ss = {}, {}, {}
    for c in NUMERIC_FEATURES:
        x = train_df[c].to_numpy(dtype=np.float64)
        obs = x[~np.isnan(x)]
        n_obs[c], s[c], ss[c] = int(obs.size), float(obs.sum()), float(np.square(obs).sum())
    return ClientStatistics(hospital, len(train_df), n_obs, s, ss)


@dataclass
class FederatedPreprocessor:
    mean: dict[str, float]
    std: dict[str, float]
    n_observed: dict[str, int]
    contributors: dict[str, dict] = field(default_factory=dict)
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def from_client_statistics(cls, stats: list[ClientStatistics]) -> FederatedPreprocessor:
        mean, std, n_total = {}, {}, {}
        for c in NUMERIC_FEATURES:
            n = sum(s.n_observed[c] for s in stats)
            if n == 0:
                raise ValueError(f"{c} is not observed in any hospital's training data")
            m = sum(s.sum[c] for s in stats) / n
            var = max(sum(s.sum_sq[c] for s in stats) / n - m * m, 0.0)
            sd = math.sqrt(var)
            mean[c], std[c], n_total[c] = m, (sd if sd > 1e-12 else 1.0), n
        contributors = {
            s.hospital: {"n_rows": s.n_rows, "n_observed": dict(s.n_observed)} for s in stats
        }
        return cls(mean, std, n_total, contributors)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Map records to the fixed 41-column representation. Reads predictor columns only."""
        n, k = len(df), len(NUMERIC_FEATURES)
        out = np.zeros((n, N_FEATURES), dtype=np.float32)
        for i, c in enumerate(NUMERIC_FEATURES):
            x = df[c].to_numpy(dtype=np.float64)
            miss = np.isnan(x)
            z = (x - self.mean[c]) / self.std[c]
            z[miss] = 0.0  # mean imputation, expressed on the standardized scale
            out[:, i] = z
            out[:, k + i] = miss
        j = 2 * k
        for c, levels in CATEGORICAL_LEVELS.items():
            x = df[c].to_numpy(dtype=np.float64)
            miss = np.isnan(x)
            unknown = ~miss & ~np.isin(x, levels)
            if unknown.any():
                raise ValueError(f"{c} has values outside the fixed schema {levels}")
            for lv in levels:
                out[:, j] = x == lv
                j += 1
            out[:, j] = miss
            j += 1
        assert j == N_FEATURES and not np.isnan(out).any()
        return out

    # ---- persistence -----------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feature_names"] = list(self.feature_names)
        d["protocol"] = (
            "federated sufficient statistics (count, sum, sum of squares) from each "
            "hospital's training partition; fixed categorical schema"
        )
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FederatedPreprocessor:
        if tuple(d["feature_names"]) != FEATURE_NAMES:
            raise ValueError("saved preprocessor uses a different feature schema")
        return cls(d["mean"], d["std"], d["n_observed"], d.get("contributors", {}))

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        write_json(self.to_dict(), directory / "preprocessor.json")
        with open(directory / "preprocessor.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, directory: str | Path) -> FederatedPreprocessor:
        return cls.from_dict(read_json(Path(directory) / "preprocessor.json"))


def fit_federated(train_by_hospital: dict[str, pd.DataFrame]) -> tuple[FederatedPreprocessor, list]:
    """Simulate the protocol: each hospital computes statistics locally, coordinator combines."""
    stats = [compute_client_statistics(df, h) for h, df in train_by_hospital.items()]
    return FederatedPreprocessor.from_client_statistics(stats), stats
