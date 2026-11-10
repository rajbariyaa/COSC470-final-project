"""Step 5 - choose the FedAvg learning rate on validation loss, then fine-tune a copy of each
seed's selected FedAvg checkpoint independently at every hospital, for every fine-tuning
learning rate (train/validation only)."""

from __future__ import annotations

import argparse

import torch

from fedheart.data import HOSPITALS
from fedheart.experiment import collect_summaries, is_done, run_dir, save_run, select_lr
from fedheart.finetune import finetune_hospital
from fedheart.prepare import load_seed
from fedheart.utils import load_config, read_json, results_dir, set_determinism, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--force", action="store_true", help="re-run fine-tuning that already exists")
    args = p.parse_args()
    cfg = load_config(args.config)
    tc = cfg["training"]

    fed_lr, table = select_lr(
        collect_summaries(cfg, "fedavg"), "fedavg", tc["fedavg"]["lr_grid"], cfg["seeds"]
    )
    write_json({"fedavg_lr": fed_lr, "validation_by_lr": table.to_dict("records")},
               results_dir(cfg) / "selection_stage1.json")  # fmt: skip
    print(f"FedAvg learning rate selected on validation loss: {fed_lr:g}")
    print(table.to_string(index=False))

    for lr in tc["finetune"]["lr_grid"]:
        for seed in cfg["seeds"]:
            base_dir = run_dir(cfg, "fedavg", fed_lr, seed)
            global_state = torch.load(base_dir / "best.pt")
            s = load_seed(cfg, seed)
            for h in HOSPITALS:
                out = run_dir(cfg, "fedavg_ft", lr, seed, h)
                # Reuse a finished run only if it started from the currently selected FedAvg lr.
                if is_done(out) and not args.force:
                    if read_json(out / "summary.json")["base_fedavg_lr"] == fed_lr:
                        continue
                set_determinism(seed)
                res = finetune_hospital(
                    global_state, s["arrays"][h], lr=lr, max_epochs=tc["finetune"]["max_epochs"],
                    batch_size=tc["batch_size"], seed=seed, hidden=cfg["model"]["hidden_units"],
                )  # fmt: skip
                save_run(out, res, {
                    "method": "fedavg_ft", "lr": lr, "seed": seed, "hospital": h,
                    "base_fedavg_lr": fed_lr,
                    "base_checkpoint": str(base_dir.relative_to(results_dir(cfg))),
                })  # fmt: skip
            print(f"fine-tune lr={lr:g} seed={seed}: done")


if __name__ == "__main__":
    main()
