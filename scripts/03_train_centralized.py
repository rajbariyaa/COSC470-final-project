"""Step 3 - centralized training for every learning rate and seed (train/validation only)."""

from __future__ import annotations

import argparse
import time

from fedheart.centralized import train_centralized
from fedheart.experiment import is_done, run_dir, save_run
from fedheart.prepare import load_seed
from fedheart.utils import load_config, set_determinism


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--force", action="store_true", help="retrain runs that already exist")
    args = p.parse_args()
    cfg = load_config(args.config)
    tc = cfg["training"]

    for lr in tc["centralized"]["lr_grid"]:
        for seed in cfg["seeds"]:
            out = run_dir(cfg, "centralized", lr, seed)
            if is_done(out) and not args.force:
                continue
            set_determinism(seed)
            s = load_seed(cfg, seed)
            t0 = time.time()
            res = train_centralized(
                s["arrays"], s["init"], lr=lr, max_epochs=tc["centralized"]["max_epochs"],
                batch_size=tc["batch_size"], seed=seed, hidden=cfg["model"]["hidden_units"],
            )  # fmt: skip
            meta = {"method": "centralized", "lr": lr, "seed": seed,
                    "elapsed_s": round(time.time() - t0, 2)}  # fmt: skip
            save_run(out, res, meta)
            print(f"centralized lr={lr:g} seed={seed}: best epoch {res['best_epoch']}, "
                  f"val BCE {res['best_val_loss']:.4f}")  # fmt: skip


if __name__ == "__main__":
    main()
