"""Classification metrics with explicit, honest handling of undefined values.

Scopes reported for every seed and method:
* one row per hospital;
* "pooled": all test records of all hospitals concatenated (large hospitals dominate);
* "equal_hospital_mean": unweighted mean of the per-hospital values, taken over the hospitals
  where the metric is defined; `<metric>_n_hospitals` records how many contributed.

A metric whose denominator is zero (e.g. precision with no positive predictions, ROC-AUC with
a single class present) is NaN and the reason is written to `undefined`; it is never replaced
by 0 or 1. ROC-AUC is computed from predicted probabilities, the other metrics from the fixed
threshold in the config (0.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from fedheart.data import HOSPITALS

METHODS: tuple[str, ...] = ("centralized", "fedavg", "fedavg_ft")
METHOD_LABELS = {
    "centralized": "Centralized",
    "fedavg": "FedAvg",
    "fedavg_ft": "FedAvg with local fine-tuning",
}
METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc")
REPORTED_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "roc_auc")
SCOPES: tuple[str, ...] = (*HOSPITALS, "pooled", "equal_hospital_mean")
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("fedavg_ft", "fedavg"),
    ("fedavg", "centralized"),
    ("fedavg_ft", "centralized"),
)


def binary_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    if y.size == 0:
        raise ValueError("cannot compute metrics on an empty set")
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    undefined: list[str] = []

    def ratio(num: int, den: int, name: str, reason: str) -> float:
        if den == 0:
            undefined.append(f"{name}: {reason}")
            return np.nan
        return num / den

    out = {
        "n": int(y.size), "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / y.size,
        "precision": ratio(tp, tp + fp, "precision", "no positive predictions"),
        "recall": ratio(tp, tp + fn, "recall", "no positive cases"),
        "specificity": ratio(tn, tn + fp, "specificity", "no negative cases"),
        "f1": ratio(2 * tp, 2 * tp + fp + fn, "f1", "no positive cases or predictions"),
    }  # fmt: skip
    if y.min() == y.max():
        undefined.append("roc_auc: only one class present")
        out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = float(roc_auc_score(y, p))
    out["undefined"] = "; ".join(undefined)
    return out


def metrics_table(predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Long table: one row per (seed, method, scope).

    `predictions` needs columns seed, method, hospital, record_id, y_true, prob.
    """
    rows = []
    for (seed, method), g in predictions.groupby(["seed", "method"], sort=True):
        per_hospital = {}
        for h in HOSPITALS:
            gh = g[g["hospital"] == h]
            per_hospital[h] = binary_metrics(gh["y_true"], gh["prob"], threshold)
            rows.append({"seed": seed, "method": method, "scope": h, **per_hospital[h]})
        rows.append(
            {"seed": seed, "method": method, "scope": "pooled",
             **binary_metrics(g["y_true"], g["prob"], threshold)}
        )  # fmt: skip
        eq: dict = {"seed": seed, "method": method, "scope": "equal_hospital_mean"}
        eq["n"] = sum(m["n"] for m in per_hospital.values())
        notes = []
        for k in METRICS:
            vals = {h: per_hospital[h][k] for h in HOSPITALS}
            defined = [v for v in vals.values() if not np.isnan(v)]
            eq[k] = float(np.mean(defined)) if defined else np.nan
            eq[f"{k}_n_hospitals"] = len(defined)
            missing = [h for h, v in vals.items() if np.isnan(v)]
            if missing:
                notes.append(f"{k}: undefined at {', '.join(missing)}")
        eq["undefined"] = "; ".join(notes)
        rows.append(eq)
    return pd.DataFrame(rows)


def summarize_over_seeds(table: pd.DataFrame, metrics=REPORTED_METRICS) -> pd.DataFrame:
    """Mean, sample SD, min, max and number of seeds with a defined value."""
    rows = []
    for (method, scope), g in table.groupby(["method", "scope"], sort=False):
        row = {"method": method, "scope": scope, "n_seeds": g["seed"].nunique()}
        for k in metrics:
            v = g[k].dropna()
            row[f"{k}_mean"] = v.mean() if len(v) else np.nan
            row[f"{k}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            row[f"{k}_min"] = v.min() if len(v) else np.nan
            row[f"{k}_max"] = v.max() if len(v) else np.nan
            row[f"{k}_n_defined"] = len(v)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(
    table: pd.DataFrame, comparisons=COMPARISONS, metrics=REPORTED_METRICS
) -> pd.DataFrame:
    """Per-seed paired differences (a - b) with a t-based 95% CI over seeds.

    With 5-10 seeds the interval is wide; it describes variability across splits and
    initializations, it is not a test across hospitals or patients.
    """
    rows = []
    for scope in table["scope"].unique():
        sub = table[table["scope"] == scope]
        for k in metrics:
            wide = sub.pivot(index="seed", columns="method", values=k)
            for a, b in comparisons:
                d = (wide[a] - wide[b]).dropna()
                n = len(d)
                mean = d.mean() if n else np.nan
                sd = d.std(ddof=1) if n > 1 else np.nan
                if n > 1:
                    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
                    lo, hi = mean - half, mean + half
                else:
                    lo = hi = np.nan
                rows.append({
                    "scope": scope, "metric": k, "comparison": f"{a} - {b}", "a": a, "b": b,
                    "n_pairs": n, "mean_diff": mean, "sd_diff": sd, "ci95_low": lo,
                    "ci95_high": hi, "wins": int((d > 1e-12).sum()),
                    "losses": int((d < -1e-12).sum()), "ties": int((d.abs() <= 1e-12).sum()),
                })  # fmt: skip
    return pd.DataFrame(rows)


def confusion_totals(table: pd.DataFrame) -> pd.DataFrame:
    """Confusion-matrix counts summed over seeds (hospital and pooled scopes)."""
    sub = table[table["scope"] != "equal_hospital_mean"]
    counts = ["tn", "fp", "fn", "tp"]
    out = sub.groupby(["method", "scope"], sort=False)[counts].sum().reset_index()
    return out.astype(dict.fromkeys(counts, "int64"))


def calibration_in_the_large(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean predicted probability vs observed prevalence per hospital (averaged over seeds)."""
    per_seed = predictions.groupby(["seed", "method", "hospital"]).agg(
        observed=("y_true", "mean"), predicted=("prob", "mean")
    )
    out = per_seed.groupby(["method", "hospital"]).mean().reset_index()
    out["difference"] = out["predicted"] - out["observed"]
    return out
