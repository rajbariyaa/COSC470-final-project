# Limitations

## Privacy

* **Keeping records at the hospital does not by itself guarantee privacy.** Model updates
  (689 weights per hospital per round) and the shared preprocessing statistics (counts, sums
  and sums of squares of five variables) carry information about the training patients and
  can, in principle, be attacked (e.g. membership or attribute inference). This project uses
  no secure aggregation, no differential privacy and no formal privacy analysis. Statements
  about privacy in this project are limited to "raw records did not leave the client in the
  simulated protocol".
* Small hospitals are more exposed: Switzerland's statistics come from about 73 training
  patients.

## Simulation, not deployment

* All four "hospitals" run on one machine in Flower's simulation engine. Network latency,
  dropped clients, asynchronous updates and heterogeneous hardware are not modelled; every
  round uses all four clients.
* `flwr.simulation.run_simulation` is deprecated in Flower 1.43 in favour of `flwr run`. It
  was used so that the experiment loop stays in a single Python process; the ClientApp and
  ServerApp would be unchanged under `flwr run`. Versions are pinned in `uv.lock`.

## Data

* Small samples: 920 patients in total. Test partitions hold about 61 / 59 / 25 / 40 patients
  per hospital, and Switzerland's contain only 1–2 patients without disease, so
  Switzerland's specificity, precision for the negative class and ROC-AUC are dominated by
  one or two patients. Per-hospital results, especially for Switzerland, should be read with
  their spread over seeds.
* Strong label shift between hospitals (prevalence 36%–93%). Part of any benefit of local
  fine-tuning may come from re-calibrating the model to a hospital's prevalence rather than
  from learning different feature-outcome relationships.
* Heavy, hospital-specific missingness (ca, thal, slope, chol). The missing-indicator columns
  therefore reveal much of the hospital's identity to the model even though hospital is not
  an input; models can partly learn site rather than physiology.
* Mean imputation (needed so the statistics can be combined from sums) is a simple choice;
  multiple imputation or models that handle missingness natively were not explored.
* The data were collected in the 1980s with the diagnostic and coding practices of that
  time; nothing here is evidence of clinical usefulness today. This is a methods comparison,
  not a clinical tool.
* Two exact duplicate pairs were kept (see `docs/CLEANING_DECISIONS.md`); negative oldpeak
  values were kept as recorded.

## Method and analysis

* Tuning was restricted to the learning rate (four values per method) to keep effort equal;
  batch size, epochs/rounds, local epochs and architecture stayed at the brief's values.
  Other settings could change the ranking.
* FedAvg re-creates each client's Adam optimizer every round, whereas centralized training
  keeps Adam's moment estimates for the whole run. This follows common FedAvg practice but is
  a genuine difference between the procedures.
* The learning-rate grids were extended once, after the validation results showed that every
  method had selected the largest value of its original grid (before any test access; see
  `docs/METHODS.md`). After the extension, fine-tuning still selected its largest learning
  rate (1e-3), and Switzerland and Hungary usually used all 10 fine-tuning epochs; the
  fine-tuning stage may therefore be under-tuned, and its benefit could be larger (or its
  over-fitting risk different) with a wider search.
* The fine-tuning stage receives extra computation (up to 10 local epochs per hospital) and
  an extra learning-rate search on top of FedAvg; its budget is reported separately.
* "FedAvg with local fine-tuning" is compared with a centralized model that receives *no*
  local adaptation. The matched comparison, centralized training followed by the same local
  fine-tuning, was not run, so the small advantage over centralized training should not be
  attributed to federated training itself.
* The mechanism suggested for the fine-tuning gain (re-centring predictions on each
  hospital's prevalence; `results/tables/table9_calibration_in_the_large.*`) is an
  interpretation, not tested by an ablation such as fine-tuning only the output bias.
* Checkpoint epoch/round selection uses the same validation data that is used to choose the
  learning rate, for every method alike; validation losses are therefore optimistic, test
  results are not.
* The classification threshold is fixed at 0.5 for all hospitals; threshold-dependent
  metrics (accuracy, precision, recall, F1) would change under hospital-specific thresholds.
* Ten seeds give confidence intervals over splits and initializations. They do not account
  for uncertainty about which hospitals participate, and they are not independent
  replications on new patients.
* The ROC curves in Figure 7 concatenate test predictions over seeds (each seed has a
  different test split) to give one curve per method; the per-seed ROC-AUC values in the
  tables are the primary results.
