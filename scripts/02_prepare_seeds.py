"""Step 2 - per seed: stratified split, federated preprocessing, initial weights, client arrays."""

from __future__ import annotations

import argparse

import pandas as pd

from fedheart.data import load_dataset
from fedheart.prepare import prepare_seed
from fedheart.utils import load_config, results_dir, set_determinism


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    df = load_dataset(cfg)

    summaries = []
    for seed in cfg["seeds"]:
        set_determinism(seed)
        out = prepare_seed(cfg, df, seed)
        summaries.append(out["summary"])
        sizes = out["summary"].pivot(index="hospital", columns="partition", values="n")
        print(f"seed {seed}: " + ", ".join(
            f"{h} {r['train']}/{r['val']}/{r['test']}" for h, r in sizes.iterrows()))  # fmt: skip
    pd.concat(summaries).to_csv(results_dir(cfg) / "seeds" / "split_summary.csv", index=False)


if __name__ == "__main__":
    main()
