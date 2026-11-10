"""Step 7 - evaluate the locked models once on the sealed test partitions."""

from __future__ import annotations

import argparse

import pandas as pd
import torch

from fedheart.data import HOSPITALS, load_dataset
from fedheart.engine import predict_proba
from fedheart.metrics import (
    METHODS,
    confusion_totals,
    metrics_table,
    paired_differences,
    summarize_over_seeds,
)
from fedheart.model import build_model
from fedheart.testset import load_test_arrays, read_locked_selection
from fedheart.utils import load_config, results_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    rdir, hidden = results_dir(cfg), cfg["model"]["hidden_units"]
    selection = read_locked_selection(cfg)
    df = load_dataset(cfg)

    rows = []
    for seed in selection["seeds"]:
        test = load_test_arrays(cfg, df, seed, purpose="final evaluation of locked models")
        k = str(seed)
        for h in HOSPITALS:
            ids, X, y = test[h]
            paths = {
                "centralized": selection["centralized"]["checkpoints"][k]["path"],
                "fedavg": selection["fedavg"]["checkpoints"][k]["path"],
                "fedavg_ft": selection["fedavg_ft"]["checkpoints"][k][h]["path"],
            }
            for method in METHODS:
                state = torch.load(rdir / paths[method] / "best.pt")
                prob = predict_proba(build_model(state, X.shape[1], hidden), X)
                rows.extend(
                    {"seed": seed, "method": method, "hospital": h, "record_id": rid,
                     "y_true": int(t), "prob": float(pr)}
                    for rid, t, pr in zip(ids, y, prob, strict=True)
                )  # fmt: skip

    out = rdir / "test"
    out.mkdir(parents=True, exist_ok=True)
    preds = pd.DataFrame(rows)
    preds["pred"] = (preds["prob"] >= selection["threshold"]).astype(int)
    preds.to_csv(out / "predictions.csv", index=False)
    table = metrics_table(preds, selection["threshold"])
    table.to_csv(out / "metrics_long.csv", index=False)
    summary = summarize_over_seeds(table)
    summary.to_csv(out / "summary_over_seeds.csv", index=False)
    paired_differences(table).to_csv(out / "paired_differences.csv", index=False)
    confusion_totals(table).to_csv(out / "confusion_totals.csv", index=False)

    view = summary[summary["scope"].isin(["pooled", "equal_hospital_mean"])]
    cols = ["scope", "method", "accuracy_mean", "f1_mean", "roc_auc_mean"]
    print(view[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
