import shutil

import numpy as np
import pandas as pd
import pytest

from fedheart.data import (
    HOSPITALS,
    ID_COLUMNS,
    PREDICTORS,
    RAW_COLUMNS,
    DataIntegrityError,
    clean_hospital,
    download_raw,
    parse_values,
)
from fedheart.utils import resolve

EXPECTED = {
    "cleveland": (303, 139), "hungary": (294, 106), "switzerland": (123, 115), "va": (200, 149),
}  # fmt: skip


def test_record_counts_labels_and_hospital_identity(dataset):
    assert len(dataset) == 920
    assert dataset["record_id"].is_unique
    for h, (n, n_pos) in EXPECTED.items():
        sub = dataset[dataset["hospital"] == h]
        assert len(sub) == n
        assert sub["target"].sum() == n_pos
        assert sub["record_id"].str.startswith(f"{h}_").all()


def test_binary_target_from_num(dataset):
    assert ((dataset["num"] > 0).astype(int) == dataset["target"]).all()
    assert set(dataset["num"].unique()) <= {0, 1, 2, 3, 4}


def test_missing_markers_become_nan():
    raw = pd.DataFrame([["?", "-9", "-9.0", "1.5"]], columns=list("abcd"))
    values, marker = parse_values(raw)
    assert values.loc[0, ["a", "b", "c"]].isna().all()
    assert values.loc[0, "d"] == 1.5
    assert marker.loc[0, ["a", "b", "c"]].all()


def test_zero_cholesterol_and_bp_are_missing(dataset):
    assert not (dataset["chol"] == 0).any()
    assert not (dataset["trestbps"] == 0).any()
    swiss = dataset[dataset["hospital"] == "switzerland"]
    assert swiss["chol"].isna().all()  # every Swiss chol was recorded as 0
    va = dataset[dataset["hospital"] == "va"]
    assert va["chol"].isna().sum() == 49 + 7  # 49 zeros + 7 "?" markers


def test_negative_oldpeak_retained(dataset):
    swiss = dataset[dataset["hospital"] == "switzerland"]
    assert (swiss["oldpeak"] < 0).sum() == 11


def test_out_of_schema_category_raises():
    row = dict.fromkeys(RAW_COLUMNS, 1.0) | {"cp": 5.0, "num": 0.0}
    with pytest.raises(DataIntegrityError):
        clean_hospital(pd.DataFrame([row]), "test")


def test_exact_duplicates_share_a_group(dataset):
    groups = dataset.set_index("record_id")["group_id"]
    assert groups["hungary_101"] == groups["hungary_102"]
    assert groups["va_139"] == groups["va_187"]
    sizes = dataset.groupby("group_id").size()
    assert (sizes > 1).sum() == 2 and sizes.max() == 2


def test_identifiers_are_not_predictors():
    assert not set(ID_COLUMNS) & set(PREDICTORS)
    assert len(PREDICTORS) == 13


def test_checksum_mismatch_stops_pipeline(cfg, tmp_path):
    raw = resolve(cfg["data"]["raw_dir"])
    for h in HOSPITALS:
        name = cfg["data"]["hospitals"][h]["file"]
        shutil.copy(raw / name, tmp_path / name)
    doc = cfg["data"]["documentation_file"]
    shutil.copy(raw / doc, tmp_path / doc)
    bad = {**cfg, "data": {**cfg["data"], "raw_dir": str(tmp_path)}}
    bad["data"]["hospitals"] = {h: dict(v) for h, v in cfg["data"]["hospitals"].items()}
    bad["data"]["hospitals"]["va"]["sha256"] = "0" * 64
    with pytest.raises(DataIntegrityError, match="SHA-256"):
        download_raw(bad)


def test_values_are_numeric(dataset):
    assert all(np.issubdtype(dataset[c].dtype, np.floating) for c in PREDICTORS)
