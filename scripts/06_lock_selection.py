"""Step 6 - select every method's configuration on validation data and lock the selection.

Only after this step can the sealed test partitions be read (step 7).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fedheart.data import HOSPITALS
from fedheart.experiment import RUN_METHODS, collect_summaries, select_lr
from fedheart.testset import lock_selection
from fedheart.utils import load_config, now_iso, results_dir

GRID_KEYS = {"centralized": "centralized", "fedavg": "fedavg", "fedavg_ft": "finetune"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument(
        "--force", action="store_true",
        help="re-lock even though test data were already accessed (starts over)",
    )  # fmt: skip
    args = p.parse_args()
    cfg = load_config(args.config)
    rdir, seeds = results_dir(cfg), cfg["seeds"]

    summaries = {m: collect_summaries(cfg, m) for m in RUN_METHODS}
    lrs, tables = {}, {}
    for m in RUN_METHODS:
        grid = cfg["training"][GRID_KEYS[m]]["lr_grid"]
        lrs[m], tables[m] = select_lr(summaries[m], m, grid, seeds)

    ft = summaries["fedavg_ft"]
    if not (ft["base_fedavg_lr"] == lrs["fedavg"]).all():
        raise SystemExit("fine-tuning started from a FedAvg learning rate that is no longer "
                         "selected; re-run 05_finetune.py --force")  # fmt: skip

    def rel(d: str) -> str:
        return str(Path(d).relative_to(rdir))

    def pick(m: str) -> pd.DataFrame:
        return summaries[m][summaries[m]["lr"] == lrs[m]].set_index("seed")

    c, f, t = pick("centralized"), pick("fedavg"), summaries["fedavg_ft"]
    t = t[t["lr"] == lrs["fedavg_ft"]]
    selection = {
        "locked_at": now_iso(),
        "criterion": "lowest mean (over seeds) pooled validation BCE; checkpoint epoch/round "
                     "chosen per seed (and per hospital for fine-tuning) on validation BCE",
        "threshold": cfg["evaluation"]["threshold"],
        "seeds": seeds,
        "centralized": {"lr": lrs["centralized"], "checkpoints": {
            str(s): {"path": rel(c.loc[s, "dir"]), "best_epoch": int(c.loc[s, "best_epoch"])}
            for s in seeds}},
        "fedavg": {"lr": lrs["fedavg"], "checkpoints": {
            str(s): {"path": rel(f.loc[s, "dir"]), "best_round": int(f.loc[s, "best_round"])}
            for s in seeds}},
        "fedavg_ft": {"lr": lrs["fedavg_ft"], "base_fedavg_lr": lrs["fedavg"], "checkpoints": {
            str(s): {h: {"path": rel(r["dir"]), "best_epoch": int(r["best_epoch"])}
                     for h in HOSPITALS
                     for _, r in t[(t["seed"] == s) & (t["hospital"] == h)].iterrows()}
            for s in seeds}},
        "validation_by_lr": {m: tables[m].to_dict("records") for m in RUN_METHODS},
    }  # fmt: skip
    if any(len(v) != len(HOSPITALS) for v in selection["fedavg_ft"]["checkpoints"].values()):
        raise SystemExit("fine-tuning checkpoints are missing for some hospital/seed")

    path = lock_selection(cfg, selection, force=args.force)
    pd.concat([tb.assign(method=m) for m, tb in tables.items()]).to_csv(
        rdir / "tuning_validation.csv", index=False
    )
    print(f"selection locked: {path}")
    for m in RUN_METHODS:
        print(f"  {m}: lr={lrs[m]:g}")


if __name__ == "__main__":
    main()
