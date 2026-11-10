import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from fedheart.data import HOSPITALS
from fedheart.metrics import binary_metrics, metrics_table, paired_differences


def test_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 200), 0, 1)
    m = binary_metrics(y, p, 0.5)
    pred = (p >= 0.5).astype(int)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 200
    assert m["undefined"] == ""


def test_precision_undefined_without_positive_predictions():
    m = binary_metrics([0, 1, 1], [0.1, 0.2, 0.3])
    assert np.isnan(m["precision"])
    assert m["recall"] == 0.0 and m["f1"] == 0.0
    assert "precision: no positive predictions" in m["undefined"]


def test_auc_undefined_with_one_class():
    m = binary_metrics([1, 1, 1], [0.9, 0.2, 0.7])
    assert np.isnan(m["roc_auc"]) and np.isnan(m["specificity"])
    assert "roc_auc" in m["undefined"]


def _toy_predictions():
    # Hospital sizes differ so pooled and equal-hospital means differ.
    rows = []
    sizes = {"cleveland": 10, "hungary": 4, "switzerland": 2, "va": 4}
    correct = {"cleveland": 10, "hungary": 2, "switzerland": 1, "va": 2}
    for h in HOSPITALS:
        for i in range(sizes[h]):
            y = i % 2
            right = i < correct[h]
            prob = (0.9 if y else 0.1) if right else (0.1 if y else 0.9)
            rows.append({"seed": 0, "method": "fedavg", "hospital": h, "record_id": f"{h}{i}",
                         "y_true": y, "prob": prob})  # fmt: skip
    return pd.DataFrame(rows)


def test_pooled_versus_equal_hospital_mean():
    t = metrics_table(_toy_predictions()).set_index("scope")
    assert t.loc["pooled", "accuracy"] == pytest.approx(15 / 20)
    per = [1.0, 0.5, 0.5, 0.5]
    assert t.loc["equal_hospital_mean", "accuracy"] == pytest.approx(np.mean(per))
    assert t.loc["equal_hospital_mean", "accuracy_n_hospitals"] == 4


def test_equal_mean_skips_undefined_and_reports_it():
    preds = _toy_predictions()
    # Make Switzerland single-class: ROC-AUC undefined there.
    preds.loc[preds["hospital"] == "switzerland", "y_true"] = 1
    t = metrics_table(preds).set_index("scope")
    assert np.isnan(t.loc["switzerland", "roc_auc"])
    assert t.loc["equal_hospital_mean", "roc_auc_n_hospitals"] == 3
    assert "roc_auc: undefined at switzerland" in t.loc["equal_hospital_mean", "undefined"]


def test_paired_difference_ci():
    t = pd.DataFrame({
        "seed": [0, 1, 2] * 2, "method": ["a"] * 3 + ["b"] * 3, "scope": "pooled",
        "accuracy": [0.6, 0.7, 0.8, 0.5, 0.5, 0.5],
    })  # fmt: skip
    d = paired_differences(t, comparisons=(("a", "b"),), metrics=("accuracy",)).iloc[0]
    assert d["mean_diff"] == pytest.approx(0.2)
    half = stats.t.ppf(0.975, 2) * 0.1 / np.sqrt(3)
    assert d["ci95_low"] == pytest.approx(0.2 - half)
    assert d["ci95_high"] == pytest.approx(0.2 + half)
    assert (d["wins"], d["losses"], d["ties"]) == (3, 0, 0)
