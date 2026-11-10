"""Retrieve, verify, parse, clean and audit the UCI Heart Disease processed files.

The four "processed" files are kept separate so that each record keeps the identity of the
hospital that collected it (Cleveland Clinic Foundation; Hungarian Institute of Cardiology,
Budapest; University Hospitals Zurich and Basel; V.A. Medical Center, Long Beach). Records
are never pooled and re-partitioned into artificial clients.

Cleaning rules (see docs/CLEANING_DECISIONS.md for the rationale behind each one):

C1  "?" and "-9"/"-9.0" are missing-value markers -> NaN.
C2  chol == 0 -> NaN. A serum cholesterol of 0 mg/dl is physiologically impossible; in this
    data it occurs for every Switzerland record and 49 VA records, i.e. it encodes "not
    measured".
C3  trestbps == 0 -> NaN (resting blood pressure of 0 mm Hg is impossible; 1 VA record).
C4  Negative oldpeak values are retained unchanged (documented, not treated as errors).
C5  Categorical values outside the documented UCI levels stop the pipeline.
C6  num (0-4) -> binary target: 0 = no disease, 1-4 = disease.
C7  Exact duplicate rows (all 14 raw fields identical) are kept but share a duplicate group
    id, so the splitter never places copies in different partitions.
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fedheart.utils import now_iso, resolve, sha256_file, write_json

HOSPITALS: tuple[str, ...] = ("cleveland", "hungary", "switzerland", "va")
HOSPITAL_LABELS = {
    "cleveland": "Cleveland",
    "hungary": "Hungary",
    "switzerland": "Switzerland",
    "va": "VA Long Beach",
}

RAW_COLUMNS: tuple[str, ...] = (
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
)  # fmt: skip
PREDICTORS: tuple[str, ...] = RAW_COLUMNS[:13]
NUMERIC_FEATURES: tuple[str, ...] = ("age", "trestbps", "chol", "thalach", "oldpeak")
# Fixed categorical schema from the UCI documentation (heart-disease.names), not from data.
CATEGORICAL_LEVELS: dict[str, tuple[int, ...]] = {
    "sex": (0, 1),
    "cp": (1, 2, 3, 4),
    "fbs": (0, 1),
    "restecg": (0, 1, 2),
    "exang": (0, 1),
    "slope": (1, 2, 3),
    "ca": (0, 1, 2, 3),
    "thal": (3, 6, 7),
}
assert set(NUMERIC_FEATURES) | set(CATEGORICAL_LEVELS) == set(PREDICTORS)

MISSING_MARKERS: tuple[str, ...] = ("?", "-9", "-9.0")
ZERO_AS_MISSING: tuple[str, ...] = ("trestbps", "chol")
# Broad physiological plausibility ranges, used only to *report* suspicious values.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "age": (18, 100),
    "trestbps": (60, 250),
    "chol": (80, 700),
    "thalach": (50, 250),
    "oldpeak": (-3, 7),
}
ID_COLUMNS: tuple[str, ...] = ("record_id", "hospital", "row_index", "group_id")


class DataIntegrityError(RuntimeError):
    """Raised when retrieved data do not match the documented schema or checksums."""


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------


def download_raw(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Download the four processed files (and the UCI documentation), verify, write manifest."""
    data_cfg = cfg["data"]
    raw_dir = resolve(data_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    base = data_cfg["base_url"]
    manifest: dict[str, Any] = {"source": base, "retrieved_or_verified_at": now_iso(), "files": {}}

    for hospital in HOSPITALS:
        spec = data_cfg["hospitals"][hospital]
        path = raw_dir / spec["file"]
        if force or not path.exists():
            _fetch(base + spec["file"], path)
        digest = sha256_file(path)
        if spec.get("sha256") and digest != spec["sha256"]:
            raise DataIntegrityError(
                f"{path.name}: SHA-256 {digest} does not match the recorded {spec['sha256']}"
            )
        n_rows = len(read_hospital_raw(path))
        if n_rows != spec["expected_rows"]:
            raise DataIntegrityError(
                f"{path.name}: {n_rows} records, expected {spec['expected_rows']}"
            )
        manifest["files"][hospital] = {
            "url": base + spec["file"],
            "path": path.name,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "records": n_rows,
        }

    doc = data_cfg.get("documentation_file")
    if doc:
        path = raw_dir / doc
        if force or not path.exists():
            _fetch(base + doc, path)
        manifest["documentation"] = {"url": base + doc, "sha256": sha256_file(path)}

    write_json(manifest, raw_dir / "manifest.json")
    return manifest


def _fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    tmp.replace(dest)


# --------------------------------------------------------------------------------------
# Parsing and cleaning
# --------------------------------------------------------------------------------------


def read_hospital_raw(path: str | Path) -> pd.DataFrame:
    """Read one processed file as strings, checking every record has 14 fields."""
    rows: list[list[str]] = []
    with open(path, newline="") as f:
        for line_no, row in enumerate(csv.reader(f), start=1):
            if not row or all(not c.strip() for c in row):
                continue
            if len(row) != len(RAW_COLUMNS):
                raise DataIntegrityError(
                    f"{Path(path).name}:{line_no} has {len(row)} fields, expected 14"
                )
            rows.append([c.strip() for c in row])
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))


def parse_values(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert raw strings to floats. Returns (values, marker_mask) where marker_mask marks C1."""
    marker = raw.isin(MISSING_MARKERS)
    values = raw.mask(marker).apply(pd.to_numeric, errors="raise").astype("float64")
    return values, marker


def clean_hospital(values: pd.DataFrame, hospital: str) -> tuple[pd.DataFrame, list[dict]]:
    """Apply cleaning rules C2-C6 to one hospital's parsed values."""
    df = values.copy()
    log: list[dict] = []

    for col, rule in (("chol", "C2"), ("trestbps", "C3")):
        mask = df[col] == 0
        log.append({
            "hospital": hospital, "rule": rule, "column": col, "n_rows": int(mask.sum()),
            "action": "set to missing",
            "rationale": f"{col} == 0 is physiologically impossible; treated as not measured",
        })  # fmt: skip
        df.loc[mask, col] = np.nan

    neg = df["oldpeak"] < 0
    log.append({
        "hospital": hospital, "rule": "C4", "column": "oldpeak", "n_rows": int(neg.sum()),
        "action": "retained",
        "rationale": "negative ST depression is plausible (ST elevation); no documented error code",
    })  # fmt: skip

    for col, levels in CATEGORICAL_LEVELS.items():
        bad = df[col].notna() & ~df[col].isin(levels)
        if bad.any():
            found = sorted(df.loc[bad, col].unique())
            raise DataIntegrityError(f"{hospital}: {col} has values outside {levels}: {found}")

    if df["num"].isna().any() or not df["num"].isin([0, 1, 2, 3, 4]).all():
        raise DataIntegrityError(f"{hospital}: diagnosis 'num' must be an integer 0-4")
    df["target"] = (df["num"] > 0).astype("int64")
    log.append({
        "hospital": hospital, "rule": "C6", "column": "num", "n_rows": int((df["num"] > 0).sum()),
        "action": "num 1-4 -> target 1", "rationale": "binary disease presence",
    })  # fmt: skip
    return df, log


def duplicate_groups(raw: pd.DataFrame, hospital: str) -> pd.Series:
    """Group id per record; records whose 14 raw fields are identical share one id (rule C7)."""
    key = raw[list(RAW_COLUMNS)].agg("|".join, axis=1)
    first_index = key.groupby(key).transform(lambda s: s.index.min())
    return first_index.map(lambda i: f"{hospital}_g{i:03d}")


def load_hospital(raw_dir: str | Path, hospital: str, filename: str) -> tuple[pd.DataFrame, dict]:
    """Load one hospital: cleaned frame (with ids) plus intermediate artefacts for the audit."""
    raw = read_hospital_raw(Path(raw_dir) / filename)
    values, marker = parse_values(raw)
    clean, log = clean_hospital(values, hospital)
    clean.insert(0, "group_id", duplicate_groups(raw, hospital))
    clean.insert(0, "row_index", np.arange(len(clean)))
    clean.insert(0, "hospital", hospital)
    clean.insert(0, "record_id", [f"{hospital}_{i:03d}" for i in range(len(clean))])
    return clean, {"raw": raw, "values": values, "marker": marker, "log": log}


def load_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    """All four hospitals, cleaned, one row per record, hospital identity retained."""
    raw_dir = resolve(cfg["data"]["raw_dir"])
    frames = [
        load_hospital(raw_dir, h, cfg["data"]["hospitals"][h]["file"])[0] for h in HOSPITALS
    ]
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------------------


def build_audit(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Per-hospital audit: summary, per-feature missingness/ranges, cleaning log, duplicates."""
    raw_dir = resolve(cfg["data"]["raw_dir"])
    summary_rows, feature_rows, log_rows, dup_rows = [], [], [], []
    raw_keys: dict[str, set[str]] = {}

    for h in HOSPITALS:
        clean, art = load_hospital(raw_dir, h, cfg["data"]["hospitals"][h]["file"])
        raw, values, marker = art["raw"], art["values"], art["marker"]
        log_rows.extend(art["log"])

        group_sizes = clean.groupby("group_id")["record_id"].transform("size")
        dups = clean[group_sizes > 1]
        for gid, g in dups.groupby("group_id"):
            dup_rows.append({
                "hospital": h, "group_id": gid, "record_ids": ";".join(g["record_id"]),
                "n_copies": len(g), "target": int(g["target"].iloc[0]),
            })  # fmt: skip
        # Same 13 predictors but a different label would be a conflicting duplicate.
        pred_key = raw[list(PREDICTORS)].agg("|".join, axis=1)
        n_conflicting = int((raw.assign(k=pred_key).groupby("k")["num"].nunique() > 1).sum())
        raw_keys[h] = set(raw.agg("|".join, axis=1))

        missing = clean[list(PREDICTORS)].isna()
        fully_missing = [c for c in PREDICTORS if missing[c].all()]
        summary_rows.append({
            "hospital": h,
            "hospital_label": HOSPITAL_LABELS[h],
            "n_records": len(clean),
            "n_positive": int(clean["target"].sum()),
            "n_negative": int((clean["target"] == 0).sum()),
            "prevalence": float(clean["target"].mean()),
            "num_distribution": ";".join(
                f"{int(k)}:{int(v)}" for k, v in clean["num"].value_counts().sort_index().items()
            ),
            "n_duplicate_groups": int(dups["group_id"].nunique()),
            "n_extra_duplicate_copies": int(len(dups) - dups["group_id"].nunique()),
            "n_conflicting_label_duplicates": n_conflicting,
            "n_complete_records": int((~missing.any(axis=1)).sum()),
            "fully_missing_predictors": ";".join(fully_missing),
        })

        for c in PREDICTORS:
            x = clean[c]
            row = {
                "hospital": h,
                "feature": c,
                "type": "numeric" if c in NUMERIC_FEATURES else "categorical",
                "n_marker_missing": int(marker[c].sum()),
                "n_zero_set_missing": int(((values[c] == 0) & (c in ZERO_AS_MISSING)).sum()),
                "n_missing_total": int(x.isna().sum()),
                "pct_missing": float(100 * x.isna().mean()),
                "n_observed": int(x.notna().sum()),
                "fully_missing": bool(x.isna().all()),
            }
            if c in NUMERIC_FEATURES:
                lo, hi = PLAUSIBLE_RANGES[c]
                obs = x.dropna()
                row |= {
                    "min": float(obs.min()) if len(obs) else np.nan,
                    "max": float(obs.max()) if len(obs) else np.nan,
                    "mean": float(obs.mean()) if len(obs) else np.nan,
                    "n_outside_plausible_range": int(((obs < lo) | (obs > hi)).sum()),
                    "n_negative": int((obs < 0).sum()),
                }
            else:
                row["levels_observed"] = ";".join(
                    f"{int(k)}:{int(v)}" for k, v in x.value_counts().sort_index().items()
                )
            feature_rows.append(row)

    cross = sum(
        len(raw_keys[a] & raw_keys[b])
        for i, a in enumerate(HOSPITALS)
        for b in HOSPITALS[i + 1 :]
    )
    summary = pd.DataFrame(summary_rows)
    summary.attrs["n_identical_records_across_hospitals"] = cross
    return {
        "hospital_summary": summary,
        "feature_audit": pd.DataFrame(feature_rows),
        "cleaning_log": pd.DataFrame(log_rows),
        "duplicates": pd.DataFrame(
            dup_rows, columns=["hospital", "group_id", "record_ids", "n_copies", "target"]
        ),
        "cross_hospital": pd.DataFrame([{"n_identical_records_across_hospitals": cross}]),
    }
