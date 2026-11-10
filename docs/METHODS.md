# Methods

*Federated vs Centralized Machine Learning for Heart Disease Prediction* — Gulam Uddin and
Raj Bariya.

## Research questions

1. Does fine-tuning the FedAvg model locally at each hospital ("FedAvg with local
   fine-tuning") improve on standard FedAvg?
2. How do FedAvg and FedAvg with local fine-tuning compare with centralized training on the
   same patients?

All three approaches share the network architecture, the patient partitions, the input
representation, the initial weights (per seed), the optimizer, the loss and the selection
criterion. They differ only in where and how the model is trained.

## Data

UCI Heart Disease, the four original "processed" files, each treated as one hospital client:
Cleveland (303 records), Hungary (294), Switzerland (123) and VA Long Beach (200). The 13
standard predictors are used; `num` is binarized (0 = no disease, 1–4 = disease). Cleaning
rules and their evidence: `docs/CLEANING_DECISIONS.md`; full audit: `docs/DATA_AUDIT.md`.

The hospitals are strongly heterogeneous: disease prevalence ranges from 36% (Hungary) to
93% (Switzerland), and missingness ranges from almost none (Cleveland) to entire variables
(Switzerland never records cholesterol; `ca` is missing for >95% of Hungary, Switzerland
and VA records).

## Partitions

Within each hospital, records are split 60/20/20 into training, validation and test
partitions, stratified on the target (`src/fedheart/splits.py`).

* The split is done at the level of duplicate groups (exact duplicate rows share a group),
  so confirmed duplicates never cross partitions.
* Feasibility was checked before splitting: Switzerland's minority class (8 patients without
  disease) gives an expected 4.8 / 1.6 / 1.6 per partition. Every split is validated to
  contain both classes in every partition at every hospital; Switzerland's test partitions
  contain only 1–2 negatives, so its specificity and ROC-AUC are extremely noisy.
* Seeds 0–9 each define a different split. Seed *s* is shared by all three methods (paired
  design), so differences between methods are computed on identical patients.
* The split of every seed is saved as `results/seeds/seed_XX/split.csv`
  (`record_id, hospital, row_index, group_id, target, partition`).

## Preprocessing protocol (no pooling of raw records)

Implemented in `src/fedheart/preprocess.py`.

**Fixed schema.** The categorical encoding is defined in advance from the UCI documentation
(cp 1–4, restecg 0–2, slope 1–3, ca 0–3, thal 3/6/7, sex/fbs/exang 0/1), with an explicit
"missing" level for every categorical variable. Each numeric variable (age, trestbps, chol,
thalach, oldpeak) has a standardized value and a missing-indicator column. The result is the
same 41 columns, in the same order, at every hospital, including features that a hospital
never observes (Switzerland's cholesterol becomes "imputed value 0 after standardization,
indicator 1").

**Federated statistics.** The only learned preprocessing parameters are the mean and
standard deviation of the five numeric variables. Each hospital computes, on its *training
partition only*, the count, sum and sum of squares of each observed numeric value and shares
only these aggregates (saved per seed in `client_statistics.json`). The coordinator adds them
to get the mean and population standard deviation. Missing numeric values are imputed with
the mean (0 on the standardized scale).

**Why this protocol.** It needs one extra round of communication of 15 numbers per hospital
and no raw records. Because sums are additive, the resulting parameters are *exactly* those a
scaler fitted on the pooled training rows would produce (verified against scikit-learn in
`tests/test_preprocess.py`). The centralized model therefore receives the same input
representation as the federated models, so the comparison isolates the training procedure.
Fully local per-hospital standardization was not used for the main comparison because it
would itself be a form of site-specific adaptation and would blur the FedAvg versus
fine-tuning comparison.

**What this is not.** It is not fully local preprocessing: the hospitals share aggregate
statistics with a coordinator. The limitations of that choice are listed in
`docs/LIMITATIONS.md`.

## Model

`HeartMLP` (`src/fedheart/model.py`): Linear(41 → 16) → ReLU → Linear(16 → 1), 689
parameters, one output logit, trained with `BCEWithLogitsLoss`. PyTorch's default
initialization is drawn from a seed-specific random stream and saved as
`results/seeds/seed_XX/init.pt`; the centralized and FedAvg runs of that seed load the same
file (fine-tuning starts from the FedAvg checkpoint and so inherits it).

## Training procedures

Common to all: Adam, batch size 16, mini-batch shuffling from seeded generators
(`derive_seed(seed, method, hospital, round/epoch)`), deterministic CPU kernels, one thread.

**Centralized** (`src/fedheart/centralized.py`). One model trained on the union of the four
hospitals' training partitions for 50 epochs. The checkpoint with the lowest pooled
validation loss is kept.

**FedAvg** (`src/fedheart/fedavg_flower.py`), implemented with Flower 1.43's Message API
(`ClientApp`, `ServerApp`, `flwr.serverapp.strategy.FedAvg`) in simulation (one SuperNode per
hospital; `context.node_config["partition-id"]` selects the hospital). Each of 50 rounds:
all four hospitals receive the global weights, train one local epoch with a newly created
Adam optimizer, and return their weights and `"num-examples"` = training-partition size;
Flower's `FedAvg` averages the weights in proportion to `"num-examples"`. Every hospital then
evaluates the new global model on its validation partition; the sample-weighted mean of
these losses (equal to the pooled validation loss) selects the checkpoint round. A
server-side hook snapshots the global weights after every round so that the selected round
can be restored; it reads no data. Flower silently drops failed client replies, so the run
is rejected unless every round received all four training and evaluation replies.

The Flower implementation is cross-checked in `tests/test_fedavg.py` against an independent
plain-PyTorch FedAvg loop (`fedavg_reference.py`); weights and validation losses agree to
within 1e-5 after three rounds. Reported results always come from Flower.

**FedAvg with local fine-tuning** (`src/fedheart/finetune.py`). For each seed, each hospital
takes its own copy of the selected FedAvg checkpoint and trains it on its training partition
with a fresh Adam optimizer for up to 10 epochs, keeping the epoch (0–10) with the lowest
loss on its own validation partition. Epoch 0 is the unmodified FedAvg model, so a hospital
keeps the global model when fine-tuning does not help on its validation data. No
regularization toward the global model is used and no layers are frozen or kept local: this
is plain post-hoc fine-tuning, not Ditto, FedPer or a personalization-layer method.

**Training budget.** One centralized epoch is about 35 optimizer steps over ~550 training
rows; one FedAvg round is about 36 steps in total over the same rows (sum of
⌈n_k / 16⌉ over the hospitals). 50 epochs and 50 rounds therefore process the same number of
examples. The fine-tuning stage adds up to 10 local epochs per hospital. Exact counts are in
`results/tables/table5_training_budget.*`.

## Model selection (validation data only)

The settings in the brief were treated as starting points and tuned on validation data with
the same effort for each method: four learning rates each (centralized and FedAvg client
learning rate: 3e-4, **1e-3**, 3e-3, 1e-2; fine-tuning: 3e-5, **1e-4**, 3e-4, 1e-3; bold =
the brief's starting value).

*Grid extension, disclosed:* the grids initially had three values (without 1e-2 and 1e-3).
With those grids, validation loss selected the largest value for all three methods, which
means the optimum may lie outside the grid. Before any test data had been accessed, every
grid was therefore extended by one step in that direction (the provisional lock was
replaced; `test_access_log.jsonl` shows the first test access happened after the final
lock). The validation losses of all values are in `results/tuning_validation.csv`.

For each method the learning rate with the lowest mean (over seeds) pooled validation
loss is chosen; for fine-tuning the hospitals' selected validation losses are first pooled
(weighted by validation size). The FedAvg learning rate is chosen before fine-tuning starts,
because fine-tuning begins from the selected FedAvg checkpoint. All other settings (batch
size, epochs/rounds, local epochs, architecture) were fixed at the brief's values.

The selection is written to `results/selection.json` and locked (`selection.lock.json`
holds its SHA-256). Test data can only be loaded through `fedheart.testset`, which refuses
to run before the lock exists and records every access in `results/test_access_log.jsonl`.
The training, tuning and selection modules never import it (checked by a test).

## Evaluation

Each locked model predicts probabilities for its test patients; the fine-tuned method uses
each hospital's own fine-tuned model for that hospital's patients.

* Metrics: accuracy, precision, recall, F1-score (threshold 0.5, fixed before testing),
  ROC-AUC from probabilities, confusion-matrix counts; specificity is also saved.
* Per hospital, **pooled** (all test patients of all hospitals together, dominated by
  Cleveland and Hungary) and **equal-hospital mean** (unweighted mean of the four hospitals'
  values).
* Undefined metrics (zero denominator, or a single class for ROC-AUC) are reported as n/a
  with the reason; equal-hospital means are taken over the hospitals where the metric is
  defined, and the number of contributing hospitals is stored.
* Variability: mean and sample SD over the 10 seeds; paired differences between methods per
  seed with a t-based 95% confidence interval and the number of seeds in which each method
  was higher. With 10 seeds the intervals describe variability over splits and
  initializations; they are not a hypothesis test over hospitals or patient populations.

## Reproducibility artefacts (all under `results/`)

| Artefact | Location |
|---|---|
| Configuration used | `config_used.yaml` |
| Package versions, Python, platform | `environment.json`, `uv.lock` |
| Raw-file checksums | `data/raw/manifest.json` |
| Data audit | `audit/*.csv`, `docs/DATA_AUDIT.md` |
| Split identifiers | `seeds/seed_XX/split.csv`, `seeds/split_summary.csv` |
| Preprocessing objects and shared statistics | `seeds/seed_XX/preprocessor.{json,pkl}`, `client_statistics.json` |
| Initial weights | `seeds/seed_XX/init.pt` |
| Checkpoints and training histories | `runs/<method>/<lr>/seed_XX/[hospital/]{best.pt,history.csv,summary.json}` |
| Model selection | `selection.json`, `selection.lock.json`, `tuning_validation.csv` |
| Test predictions | `test/predictions.csv` |
| Metrics | `test/metrics_long.csv`, `test/summary_over_seeds.csv`, `test/paired_differences.csv`, `test/confusion_totals.csv` |
| Tables and figures | `tables/`, `figures/` |
