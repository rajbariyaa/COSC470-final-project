"""Step 4 - FedAvg in Flower simulation for every learning rate and seed (train/validation only)."""

from __future__ import annotations

import argparse
import time

from fedheart.experiment import is_done, run_dir, save_run
from fedheart.fedavg_flower import run_fedavg
from fedheart.prepare import load_seed, seed_dir
from fedheart.utils import load_config, set_determinism


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--force", action="store_true", help="retrain runs that already exist")
    p.add_argument("--verbose", action="store_true", help="show Flower's per-round log")
    args = p.parse_args()
    cfg = load_config(args.config)
    tc = cfg["training"]

    for lr in tc["fedavg"]["lr_grid"]:
        for seed in cfg["seeds"]:
            out = run_dir(cfg, "fedavg", lr, seed)
            if is_done(out) and not args.force:
                continue
            set_determinism(seed)
            s = load_seed(cfg, seed)
            t0 = time.time()
            res = run_fedavg(
                data_dir=seed_dir(cfg, seed) / "client_data", init_state=s["init"], lr=lr,
                num_rounds=tc["fedavg"]["num_rounds"], local_epochs=tc["fedavg"]["local_epochs"],
                batch_size=tc["batch_size"], seed=seed, hidden=cfg["model"]["hidden_units"],
                verbose=args.verbose,
            )  # fmt: skip
            meta = {"method": "fedavg", "lr": lr, "seed": seed,
                    "elapsed_s": round(time.time() - t0, 2)}  # fmt: skip
            save_run(out, res, meta)
            print(f"fedavg lr={lr:g} seed={seed}: best round {res['best_round']}, "
                  f"val BCE {res['best_val_loss']:.4f} ({time.time() - t0:.0f}s)")  # fmt: skip


if __name__ == "__main__":
    main()
