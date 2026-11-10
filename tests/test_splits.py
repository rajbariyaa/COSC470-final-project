import pandas as pd
import pytest

from fedheart.data import HOSPITALS
from fedheart.splits import (
    PARTITIONS,
    SplitError,
    make_split,
    minority_class_feasibility,
    validate_split,
)

SEEDS = range(10)


@pytest.fixture(scope="module")
def splits(cfg, dataset):
    return {s: make_split(dataset, s, cfg["split"]["fractions"]) for s in SEEDS}


def test_minority_counts_support_the_split(cfg, dataset):
    feas = minority_class_feasibility(dataset, cfg["split"]["fractions"])
    assert feas["feasible"].all()
    swiss = feas.set_index("hospital").loc["switzerland"]
    assert swiss["n_minority"] == 8 and swiss["minority_class"] == "negative"


def test_split_is_reproducible(cfg, dataset, splits):
    again = make_split(dataset, 0, cfg["split"]["fractions"])
    pd.testing.assert_frame_equal(splits[0], again)
    assert not splits[0]["partition"].equals(splits[1]["partition"])


def test_partitions_complete_and_disjoint(dataset, splits):
    for split in splits.values():
        assert split["record_id"].is_unique
        assert set(split["record_id"]) == set(dataset["record_id"])
        assert set(split["partition"]) == set(PARTITIONS)


def test_every_partition_has_both_classes(splits):
    for split in splits.values():
        counts = split.groupby(["hospital", "partition"])["target"].agg(["min", "max"])
        assert (counts["min"] == 0).all() and (counts["max"] == 1).all()


def test_proportions_close_to_60_20_20(splits):
    for split in splits.values():
        for h in HOSPITALS:
            sub = split[split["hospital"] == h]
            share = sub["partition"].value_counts(normalize=True)
            assert abs(share["train"] - 0.6) < 0.02
            assert abs(share["val"] - 0.2) < 0.02
            assert abs(share["test"] - 0.2) < 0.02


def test_duplicates_never_cross_partitions(splits):
    for split in splits.values():
        assert (split.groupby("group_id")["partition"].nunique() == 1).all()


def test_validator_rejects_crossing_duplicate(cfg, dataset, splits):
    broken = splits[0].copy()
    idx = broken.index[broken["record_id"] == "hungary_102"][0]
    other = broken.loc[broken["record_id"] == "hungary_101", "partition"].iloc[0]
    broken.loc[idx, "partition"] = next(p for p in PARTITIONS if p != other)
    with pytest.raises(SplitError, match="crosses"):
        validate_split(broken, dataset, cfg["split"]["fractions"])


def test_validator_rejects_single_class_partition(cfg, dataset, splits):
    broken = splits[0].copy()
    swiss_test_neg = (
        (broken["hospital"] == "switzerland")
        & (broken["partition"] == "test")
        & (broken["target"] == 0)
    )
    broken.loc[swiss_test_neg, "partition"] = "train"
    with pytest.raises(SplitError):
        validate_split(broken, dataset, cfg["split"]["fractions"])
