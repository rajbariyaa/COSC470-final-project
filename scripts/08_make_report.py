"""Step 8 - publication tables and figures from the saved results."""

from __future__ import annotations

import argparse

import pandas as pd

from fedheart import report as R
from fedheart.data import HOSPITALS
from fedheart.experiment import RUN_METHODS, collect_summaries
from fedheart.metrics import METHODS, calibration_in_the_large, confusion_totals
from fedheart.testset import read_locked_selection
from fedheart.utils import load_config, results_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    rdir = results_dir(cfg)
    tables, figures = rdir / "tables", rdir / "figures"
    R.apply_style()

    audit = {n: pd.read_csv(rdir / "audit" / f"{n}.csv")
             for n in ("hospital_summary", "feature_audit")}  # fmt: skip
    split_summary = pd.read_csv(rdir / "seeds" / "split_summary.csv")
    selection = read_locked_selection(cfg)
    summaries = {m: collect_summaries(cfg, m) for m in RUN_METHODS}
    validation = {m: pd.DataFrame(selection["validation_by_lr"][m]) for m in RUN_METHODS}
    saved = ("predictions", "metrics_long", "summary_over_seeds", "paired_differences")
    test = {n: pd.read_csv(rdir / "test" / f"{n}.csv") for n in saved}
    # Recomputed from the saved per-seed metrics (integer counts); no test data are re-read.
    test["confusion_totals"] = confusion_totals(test["metrics_long"])
    test["confusion_totals"].to_csv(rdir / "test" / "confusion_totals.csv", index=False)
    n_seeds = len(selection["seeds"])

    R.write_table(R.dataset_table(audit["hospital_summary"], audit["feature_audit"], split_summary),
                  tables, "table1_dataset",
                  "Hospitals, class balance, missingness and mean partition sizes")  # fmt: skip
    R.write_table(R.results_table(test["summary_over_seeds"]), tables, "table2_test_results",
                  f"Test metrics, mean ± SD over {n_seeds} paired seeds (threshold 0.5)")
    R.write_table(R.paired_table(test["paired_differences"]), tables, "table3_paired_differences",
                  "Paired differences over seeds: mean [95% CI] (seeds in which the first method "
                  "was higher/lower/tied)")  # fmt: skip
    R.write_table(R.selection_table(validation), tables, "table4_model_selection",
                  "Learning-rate selection on validation data only")
    R.write_table(R.budget_table(cfg, summaries, selection), tables, "table5_training_budget",
                  "Training and tuning budgets")
    ft = summaries["fedavg_ft"]
    ft_sel = ft[ft["lr"] == selection["fedavg_ft"]["lr"]]
    R.write_table(R.finetune_epoch_table(ft_sel), tables, "table6_finetune_epochs",
                  "Fine-tuning epoch selected on each hospital's validation data")
    R.write_table(R.confusion_table(test["confusion_totals"]), tables, "table7_confusion_totals",
                  "Confusion-matrix counts summed over seeds")
    R.write_table(R.undefined_table(test["metrics_long"]), tables, "table8_undefined_metrics",
                  "Seeds in which a metric was undefined (reported as n/a, never imputed)")
    R.write_table(R.calibration_table(calibration_in_the_large(test["predictions"])), tables,
                  "table9_calibration_in_the_large",
                  "Observed test prevalence vs mean predicted probability (mean over seeds)")

    def histories(method: str, hospital: str | None = None) -> list[pd.DataFrame]:
        s = summaries[method]
        s = s[s["lr"] == selection[method]["lr"]]
        if hospital:
            s = s[s["hospital"] == hospital]
        return [pd.read_csv(f"{d}/history.csv") for d in s.sort_values("seed")["dir"]]

    R.fig_class_balance(audit["hospital_summary"], figures)
    R.fig_missingness(audit["feature_audit"], figures)
    R.fig_validation_curves({m: histories(m) for m in ("centralized", "fedavg")}, figures)
    R.fig_finetune_curves({h: histories("fedavg_ft", h) for h in HOSPITALS}, figures)
    R.fig_test_metrics(test["metrics_long"], figures, n_seeds)
    R.fig_paired_differences(test["paired_differences"], figures)
    R.fig_roc(test["predictions"], figures)
    R.fig_confusion(test["confusion_totals"], figures)
    print(f"tables -> {tables}\nfigures -> {figures}")
    assert set(METHODS) == set(RUN_METHODS)


if __name__ == "__main__":
    main()
