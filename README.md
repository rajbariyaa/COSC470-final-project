# Federated vs Centralized Machine Learning for Heart Disease Prediction

**Gulam Uddin and Raj Bariya** · COSC 470, Fall 2026

This project compares three ways of training the *same* small neural network to predict heart
disease from the four hospitals of the UCI Heart Disease dataset (Cleveland, Hungary,
Switzerland, VA Long Beach), using *identical* patient splits:

| Approach | Where training happens |
|---|---|
| **Centralized** | one model trained on all hospitals' training records pooled together |
| **FedAvg** | Federated Averaging with [Flower](https://flower.ai) 1.43; each hospital trains locally, the server averages the weights in proportion to each hospital's training-set size |
| **FedAvg with local fine-tuning** | the selected FedAvg model is fine-tuned independently at each hospital on its own data |

Research questions: does local fine-tuning improve standard FedAvg, and how do both compare
with centralized training? Results are in [`docs/RESULTS.md`](docs/RESULTS.md),
`results/tables/` and `results/figures/`; the walkthrough notebook is
[`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb).

**Results at a glance** (10 paired seeds; details, intervals and caveats in
[`docs/RESULTS.md`](docs/RESULTS.md)):

| Pooled test set | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| Centralized | 0.821 ± 0.027 | 0.837 ± 0.029 | 0.898 ± 0.019 |
| FedAvg | 0.818 ± 0.021 | 0.835 ± 0.021 | 0.887 ± 0.022 |
| FedAvg with local fine-tuning | 0.837 ± 0.029 | 0.852 ± 0.028 | 0.904 ± 0.021 |

FedAvg came close to centralized training except at Switzerland; local fine-tuning improved
FedAvg mainly there, apparently by re-centring predictions on each hospital's disease
prevalence rather than by ranking patients better within a hospital. All differences are
small (≈0.01–0.02) and come from one small, heterogeneous dataset.

> The personalized method is plain **FedAvg with local fine-tuning**. It is not Ditto,
> FedPer or a personalization-layer method. Keeping records at the hospitals does **not** by
> itself guarantee privacy; see [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Setup

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/) ≥ 0.10, internet access for
the first download of the data (UCI) and packages. No GPU is needed.

```bash
uv sync            # creates .venv with Python 3.12 and the exact versions in uv.lock
```

Pinned core packages: `flwr[simulation]==1.43.0` (with Ray 2.55.1), `torch==2.15.0`,
`scikit-learn==1.9.1`, `pandas==3.1.1`. Flower 1.43 needs Python ≥ 3.11; the project uses
Python 3.12 (`requires-python = ">=3.11,<3.13"`), a conservative choice inside the range
supported by Flower, Ray and PyTorch. uv downloads it automatically if needed.

## Running the experiments

Everything, in order (about 1.5 hours on a laptop CPU, almost all of it FedAvg):

```bash
scripts/run_all.sh                 # or: scripts/run_all.sh path/to/other_config.yaml
```

Or step by step:

| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `uv run python scripts/01_download_audit.py` | download the four UCI files, verify SHA-256 and record counts, audit each hospital | `data/raw/`, `results/audit/`, `docs/DATA_AUDIT.md` |
| 2 | `uv run python scripts/02_prepare_seeds.py` | for each seed: stratified 60/20/20 split per hospital, federated preprocessing fit, initial weights, client arrays (train/val only) | `results/seeds/seed_XX/` |
| 3 | `uv run python scripts/03_train_centralized.py` | centralized training, 4 learning rates × 10 seeds (seconds) | `results/runs/centralized/` |
| 4 | `uv run python scripts/04_train_fedavg.py` | FedAvg in Flower simulation, 4 learning rates × 10 seeds (~2 min per run) | `results/runs/fedavg/` |
| 5 | `uv run python scripts/05_finetune.py` | select the FedAvg learning rate on validation loss, then local fine-tuning at each hospital, 4 learning rates × 10 seeds × 4 hospitals | `results/runs/fedavg_ft/` |
| 6 | `uv run python scripts/06_lock_selection.py` | validation-only selection for every method, written to `selection.json` and locked | `results/selection*.json` |
| 7 | `uv run python scripts/07_evaluate_test.py` | the only step that reads test data; refuses to run before step 6 | `results/test/` |
| 8 | `uv run python scripts/08_make_report.py` | tables (CSV/Markdown/LaTeX) and figures (PNG/PDF) | `results/tables/`, `results/figures/` |

Steps 3–5 skip runs that already exist (pass `--force` to redo them), so an interrupted
run can be resumed. Once step 7 has read the test data, step 6 refuses to lock a new
selection (by design); to reproduce everything from scratch, point `paths.results_dir` in a
copy of the config to an empty folder, e.g. `scripts/run_all.sh configs/my_rerun.yaml`. FedAvg's wall time is dominated by Flower's simulation engine, whose
strategy loop polls for client replies once per second; it does not affect results.
`run_simulation` prints a deprecation warning (Flower now recommends `flwr run`); see
`docs/LIMITATIONS.md`.

Settings (seeds, grids, budgets, paths) live in `configs/default.yaml`.

## Tests and checks

```bash
uv run ruff check .                 # lint (incl. the notebook)
uv run pytest                       # 53 fast tests (data, splits, preprocessing, model, metrics, test-set lock, FedAvg aggregation)
uv run pytest -m slow               # Flower simulation reproduces an independent FedAvg loop (~25 s)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb
```

## Walkthrough notebook

```bash
uv run jupyter lab notebooks/walkthrough.ipynb
```

It reruns the data, split and preprocessing steps live, runs a short validation-only
demonstration of each method on seed 0, and then loads the locked results. It needs the
`results/` folder from `scripts/run_all.sh`.

## What is where

```
configs/default.yaml        all settings (data URLs + checksums, seeds, grids, budgets)
src/fedheart/
  data.py                   download, checksum, parse, clean (rules C1-C7), per-hospital audit
  splits.py                 stratified, duplicate-aware 60/20/20 split within each hospital
  preprocess.py             fixed 41-column schema; federated count/sum/sum-of-squares statistics
  prepare.py                per-seed artefacts; builds train/val arrays only
  model.py                  HeartMLP: 41 -> 16 (ReLU) -> 1 logit
  engine.py                 shared training/evaluation loops (Adam, BCE-with-logits, batch 16)
  centralized.py            centralized baseline
  fedavg_flower.py          Flower ClientApp/ServerApp, FedAvg weighted by "num-examples"
  fedavg_reference.py       plain-PyTorch FedAvg used only to test the Flower implementation
  finetune.py               FedAvg with local fine-tuning
  experiment.py             run bookkeeping and validation-only model selection
  testset.py                sealed test partitions (lock + access log)
  metrics.py                per-hospital, pooled and equal-hospital metrics; paired differences
  report.py                 publication tables and figures
scripts/01_...08_*.py       the pipeline steps; run_all.sh runs them in order
tests/                      pytest suite
notebooks/walkthrough.ipynb walkthrough
docs/                       METHODS, RESULTS, LIMITATIONS, CLEANING_DECISIONS, DATA_AUDIT (generated)
results/                    everything the pipeline produces (see docs/METHODS.md)
assignment.ipynb            the course's original file (left untouched; it was empty)
```

## Reproducibility

* Every random choice has its own seeded stream (`derive_seed`), so results do not depend on
  execution order or on which Ray worker runs a client. CPU only, deterministic PyTorch
  kernels, one thread.
* Seed *s* fixes the split, the preprocessing fit, the initial weights and the shuffling for
  all three methods (paired design). Ten seeds are used.
* Saved: split identifiers, preprocessing objects and the statistics each hospital shared,
  initial weights, configurations, every checkpoint and training history, package versions
  (`results/environment.json`, `uv.lock`), raw-file checksums, the locked selection, the test
  access log and all test predictions.
* Re-running the pipeline on another machine should reproduce the splits and preprocessing
  exactly; trained weights may differ in the last digits across CPU architectures or library
  builds.

## Data and citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset].
UCI Machine Learning Repository. <https://doi.org/10.24432/C52P4X> (CC BY 4.0).
Introductory paper: Detrano, R. et al. (1989). International application of a new probability
algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*.

FedAvg: McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017).
Communication-efficient learning of deep networks from decentralized data. *AISTATS*.
Flower: Beutel, D. J. et al. (2020). Flower: A friendly federated learning research
framework. arXiv:2007.14390.
