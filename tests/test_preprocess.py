import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from fedheart.data import CATEGORICAL_LEVELS, HOSPITALS, ID_COLUMNS, NUMERIC_FEATURES
from fedheart.prepare import partition_frame
from fedheart.preprocess import (
    FEATURE_NAMES,
    N_FEATURES,
    FederatedPreprocessor,
    compute_client_statistics,
    fit_federated,
)


@pytest.fixture(scope="module")
def train_frames(dataset, split0):
    return {h: partition_frame(dataset, split0, h, "train") for h in HOSPITALS}


@pytest.fixture(scope="module")
def pre(train_frames):
    return fit_federated(train_frames)[0]


def test_fixed_schema_dimension_and_order():
    assert N_FEATURES == 41
    assert FEATURE_NAMES[:5] == NUMERIC_FEATURES
    assert not any(n.split("=")[0].split("__")[0] in ID_COLUMNS for n in FEATURE_NAMES)


def test_federated_statistics_equal_pooled_training_fit(train_frames, pre):
    """Combining per-hospital sums reproduces a scaler fitted on pooled training rows."""
    pooled = pd.concat(train_frames.values(), ignore_index=True)
    scaler = StandardScaler().fit(pooled[list(NUMERIC_FEATURES)].to_numpy())  # ignores NaN
    np.testing.assert_allclose([pre.mean[c] for c in NUMERIC_FEATURES], scaler.mean_, rtol=1e-12)
    np.testing.assert_allclose([pre.std[c] for c in NUMERIC_FEATURES], scaler.scale_, rtol=1e-9)
    expected = np.nan_to_num(scaler.transform(pooled[list(NUMERIC_FEATURES)].to_numpy()), nan=0.0)
    np.testing.assert_allclose(pre.transform(pooled)[:, :5], expected, atol=1e-5)


def test_every_hospital_gets_same_dimension(dataset, split0, pre):
    for h in HOSPITALS:
        for p in ("train", "val", "test"):
            X = pre.transform(partition_frame(dataset, split0, h, p))
            assert X.shape[1] == N_FEATURES
            assert not np.isnan(X).any()


def test_entirely_missing_feature_keeps_dimension(train_frames, pre):
    X = pre.transform(train_frames["switzerland"])
    chol = FEATURE_NAMES.index("chol")
    assert (X[:, chol] == 0).all()
    assert (X[:, FEATURE_NAMES.index("chol__missing")] == 1).all()


def test_one_hot_blocks_sum_to_one(dataset, pre):
    X = pre.transform(dataset)
    for c in CATEGORICAL_LEVELS:
        cols = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith(f"{c}=")]
        assert (X[:, cols].sum(axis=1) == 1).all()


def test_fit_only_uses_training_rows(dataset, split0, train_frames, pre):
    with pytest.raises(ValueError, match="training rows"):
        compute_client_statistics(partition_frame(dataset, split0, "cleveland", "val"), "cleveland")
    # Corrupting validation/test values must not change the fitted parameters.
    corrupted = dataset.copy()
    held_out = corrupted["record_id"].isin(split0.loc[split0["partition"] != "train", "record_id"])
    corrupted.loc[held_out, "age"] = 999.0
    refit = fit_federated({h: partition_frame(corrupted, split0, h, "train") for h in HOSPITALS})[0]
    assert refit.mean == pre.mean and refit.std == pre.std


def test_identifiers_do_not_affect_inputs(train_frames, pre):
    df = train_frames["cleveland"]
    scrambled = df.assign(hospital="va", record_id="x", row_index=0, group_id="g")
    np.testing.assert_array_equal(pre.transform(df), pre.transform(scrambled))


def test_unknown_category_raises(train_frames, pre):
    df = train_frames["cleveland"].copy()
    df.loc[df.index[0], "thal"] = 5.0
    with pytest.raises(ValueError, match="fixed schema"):
        pre.transform(df)


def test_save_load_roundtrip(tmp_path, train_frames, pre):
    pre.save(tmp_path)
    loaded = FederatedPreprocessor.load(tmp_path)
    df = train_frames["va"]
    np.testing.assert_array_equal(pre.transform(df), loaded.transform(df))
