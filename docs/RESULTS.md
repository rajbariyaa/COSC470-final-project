# Results

*Federated vs Centralized Machine Learning for Heart Disease Prediction* — Gulam Uddin and
Raj Bariya.

All numbers below are copied from the generated tables in `results/tables/` (CSV, Markdown and
LaTeX versions) and can be regenerated with `scripts/run_all.sh`. Test metrics are mean ± SD
over 10 paired seeds; differences are paired by seed (same patients, same initial weights) and
reported as mean [95% t-interval] with the number of seeds in which the first method was
higher / lower / tied. Many scopes, metrics and comparisons are reported without correction
for multiple comparisons, so the intervals should be read as descriptive.

The test partitions were read once, after model selection was locked: the lock was written at
2026-11-09T06:40:20Z and all 10 test accesses (one per seed) are logged at 06:40:29Z under
the same selection hash (`results/test_access_log.jsonl`).

## 1. Model selection (validation data only)

`results/tables/table4_model_selection.*`

| Method | Learning rates tried | Selected | Pooled validation BCE at selection |
|---|---|---|---|
| Centralized | 3e-4, 1e-3, 3e-3, 1e-2 | **3e-3** | 0.3856 ± 0.0463 |
| FedAvg (client learning rate) | 3e-4, 1e-3, 3e-3, 1e-2 | **1e-2** | 0.4149 ± 0.0434 |
| FedAvg with local fine-tuning | 3e-5, 1e-4, 3e-4, 1e-3 | **1e-3** | 0.3748 ± 0.0424 |

* With the original three-value grids every method selected its largest value, so each grid
  was extended by one step before any test access (`docs/METHODS.md`). After the extension the
  centralized optimum is interior (1e-3 and 3e-3 are practically tied: 0.3856 vs 0.3856);
  FedAvg's 1e-2 and 3e-3 are practically tied (0.4149 vs 0.4151); **fine-tuning still selects
  the largest value (1e-3)** and its validation loss was still falling (0.3913 at 3e-4), so a
  larger fine-tuning learning rate might do better. We stopped extending at this point.
* Selected checkpoints (`table5_training_budget.*`): centralized epoch median 8 (range 3–26),
  FedAvg round median 8 (3–18). Both validation curves rise after roughly 5–10 epochs/rounds
  (Figure 3), so the 50-epoch/50-round budget was ample and checkpoint selection mattered.
* Fine-tuning epoch selected per hospital (`table6_finetune_epochs.*`): Cleveland kept the
  unmodified FedAvg model in 3/10 seeds (median epoch 4); VA in 2/10 (median 6); Hungary
  never (median 10); Switzerland always used all 10 epochs. Switzerland's and Hungary's
  validation losses were still decreasing at the 10-epoch cap (Figure 4).
* Budgets: 50 centralized epochs = 1,750 optimizer steps; 50 FedAvg rounds = 1,800 steps
  summed over the four clients; both process 27,475 training examples per run. FedAvg
  transfers 1.65 MB of model weights per run (689 float32 parameters; four clients; download,
  upload and evaluation download each round). Fine-tuning adds 360 steps per seed (all four
  hospitals together) and an extra four-value learning-rate search.

## 2. Test results

`results/tables/table2_test_results.*`, Figure 5

**Overall**

| Scope | Method | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|---|
| Pooled | Centralized | 0.821 ± 0.027 | 0.845 ± 0.026 | 0.832 ± 0.059 | 0.837 ± 0.029 | 0.898 ± 0.019 |
| Pooled | FedAvg | 0.818 ± 0.021 | 0.841 ± 0.036 | 0.833 ± 0.049 | 0.835 ± 0.021 | 0.887 ± 0.022 |
| Pooled | FedAvg with local fine-tuning | 0.837 ± 0.029 | 0.857 ± 0.030 | 0.849 ± 0.046 | 0.852 ± 0.028 | 0.904 ± 0.021 |
| Equal-hospital mean | Centralized | 0.827 ± 0.021 | 0.848 ± 0.025 | 0.830 ± 0.063 | 0.835 ± 0.034 | 0.837 ± 0.043 |
| Equal-hospital mean | FedAvg | 0.811 ± 0.020 | 0.848 ± 0.033 | 0.828 ± 0.052 | 0.834 ± 0.022 | 0.839 ± 0.047 |
| Equal-hospital mean | FedAvg with local fine-tuning | 0.845 ± 0.026 | 0.861 ± 0.032 | 0.845 ± 0.048 | 0.849 ± 0.031 | 0.843 ± 0.045 |

For every method, pooled ROC-AUC (0.887–0.904) is higher than the equal-hospital mean of the
per-hospital ROC-AUCs (0.837–0.843). Pooled ROC-AUC also ranks patients of different
hospitals against each other, so it can benefit from prevalence differences between hospitals;
within-hospital discrimination is lowest at VA Long Beach (≈0.72) and Switzerland (≈0.82).

**Per hospital** (full table with precision and recall in `table2_test_results.*`)

| Hospital | Method | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|---|
| Cleveland | Centralized | 0.846 ± 0.048 | 0.828 ± 0.064 | 0.914 ± 0.024 |
| Cleveland | FedAvg | 0.849 ± 0.041 | 0.836 ± 0.044 | 0.910 ± 0.028 |
| Cleveland | FedAvg with local fine-tuning | 0.848 ± 0.048 | 0.831 ± 0.057 | 0.911 ± 0.029 |
| Hungary | Centralized | 0.824 ± 0.052 | 0.744 ± 0.087 | 0.895 ± 0.039 |
| Hungary | FedAvg | 0.834 ± 0.043 | 0.774 ± 0.060 | 0.896 ± 0.048 |
| Hungary | FedAvg with local fine-tuning | 0.834 ± 0.047 | 0.752 ± 0.073 | 0.902 ± 0.047 |
| Switzerland | Centralized | 0.912 ± 0.041 | 0.952 ± 0.023 | 0.820 ± 0.166 |
| Switzerland | FedAvg | 0.800 ± 0.082 | 0.882 ± 0.053 | 0.822 ± 0.154 |
| Switzerland | FedAvg with local fine-tuning | 0.932 ± 0.027 | 0.964 ± 0.014 | 0.833 ± 0.151 |
| VA Long Beach | Centralized | 0.725 ± 0.051 | 0.815 ± 0.039 | 0.718 ± 0.079 |
| VA Long Beach | FedAvg | 0.759 ± 0.050 | 0.843 ± 0.033 | 0.727 ± 0.082 |
| VA Long Beach | FedAvg with local fine-tuning | 0.767 ± 0.057 | 0.848 ± 0.040 | 0.727 ± 0.079 |

No metric was undefined in any seed (`table8_undefined_metrics.*` is empty). Switzerland's
test partitions contain only 1–2 patients without disease (16 over all seeds), so its
ROC-AUC (SD ≈ 0.15) and specificity rest on one or two patients per seed.

## 3. Does local fine-tuning improve FedAvg? (RQ1)

`table3_paired_differences.*`, Figure 6

| Scope | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| Pooled | +0.019 [+0.009, +0.029] (8/1/1) | +0.017 [+0.005, +0.029] (8/2/0) | +0.017 [+0.013, +0.020] (10/0/0) |
| Equal-hospital mean | +0.034 [+0.018, +0.051] (10/0/0) | +0.015 [+0.002, +0.029] (8/2/0) | +0.004 [−0.002, +0.010] (7/3/0) |
| Cleveland | −0.002 [−0.008, +0.005] (1/2/7) | −0.005 [−0.017, +0.007] (1/2/7) | +0.002 [−0.001, +0.005] (4/2/4) |
| Hungary | +0.000 [−0.015, +0.015] (2/3/5) | −0.021 [−0.042, −0.000] (2/8/0) | +0.005 [+0.000, +0.010] (6/3/1) |
| Switzerland | +0.132 [+0.073, +0.191] (9/0/1) | +0.082 [+0.044, +0.119] (10/0/0) | +0.011 [−0.012, +0.033] (3/2/5) |
| VA Long Beach | +0.007 [−0.011, +0.026] (4/1/5) | +0.005 [−0.010, +0.020] (4/1/5) | −0.000 [−0.009, +0.009] (2/4/4) |

**Answer.** In this setting, FedAvg with local fine-tuning improved on standard FedAvg overall
(pooled accuracy, F1 and ROC-AUC; equal-hospital accuracy and F1), but the gain is uneven
across hospitals:

* It is concentrated at **Switzerland**, where FedAvg missed many disease cases (recall 0.816,
  43 false negatives over all seeds) and fine-tuning recovered them (recall 0.975, 6 false
  negatives; `table7_confusion_totals.*`).
* At **Hungary** fine-tuning traded recall (0.782 → 0.695) for precision (0.775 → 0.828),
  lowering F1 (−0.021) while ROC-AUC rose slightly (+0.005).
* At **Cleveland** and **VA** there was no clear change.
* Within-hospital ROC-AUC barely changed anywhere (−0.000 to +0.011, intervals including or
  touching 0), whereas pooled ROC-AUC rose by +0.017 in every seed.

**Interpretation (supported by `table9_calibration_in_the_large.*`).** The single FedAvg model
predicts a similar disease-probability level everywhere: its mean predicted probability was
0.725 at Switzerland, whose test prevalence is 0.936, and 0.429 at Hungary (prevalence 0.366).
After fine-tuning these became 0.890 and 0.361. Most of the benefit therefore looks like each
hospital re-centring the model on its own prevalence (adaptation to label shift) rather than
learning a better ranking of its own patients. That changes threshold-dependent metrics at
0.5 and the cross-hospital (pooled) ranking, but hardly the within-hospital ROC-AUC. We did
not isolate this mechanism experimentally (for example by fine-tuning only the output bias),
so it is an interpretation consistent with the data, not a demonstrated cause.

## 4. How do the federated approaches compare with centralized training? (RQ2)

**FedAvg − Centralized**

| Scope | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| Pooled | −0.003 [−0.018, +0.011] (4/6/0) | −0.002 [−0.017, +0.013] (5/5/0) | −0.011 [−0.016, −0.006] (0/9/1) |
| Equal-hospital mean | −0.016 [−0.033, +0.001] (3/7/0) | −0.001 [−0.018, +0.015] (5/5/0) | +0.002 [−0.005, +0.009] (6/4/0) |
| Switzerland | −0.112 [−0.167, −0.057] (0/9/1) | −0.070 [−0.106, −0.035] (0/10/0) | +0.003 [−0.033, +0.038] (2/2/6) |
| VA Long Beach | +0.035 [−0.003, +0.072] (5/1/4) | +0.028 [−0.001, +0.056] (7/1/2) | +0.009 [−0.010, +0.028] (7/3/0) |

Standard FedAvg came close to centralized training on pooled accuracy and F1 (differences of
−0.003 and −0.002 with intervals spanning zero), but its pooled ROC-AUC was lower by 0.011
(lower in 9 seeds, tied in 1), and it was clearly worse at Switzerland (accuracy −0.112),
where the centralized model's mean prediction (0.859) sat much closer to the local prevalence
than FedAvg's (0.725). At VA, FedAvg tended to do better than centralized training, with
intervals that touch zero. FedAvg's mean validation loss was also higher than centralized
training's throughout training (Figure 3).

**FedAvg with local fine-tuning − Centralized**

| Scope | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| Pooled | +0.016 [+0.001, +0.031] (6/2/2) | +0.015 [−0.000, +0.031] (6/3/1) | +0.006 [+0.002, +0.010] (10/0/0) |
| Equal-hospital mean | +0.018 [+0.002, +0.035] (8/2/0) | +0.014 [−0.003, +0.030] (6/4/0) | +0.006 [+0.000, +0.012] (8/2/0) |
| VA Long Beach | +0.042 [+0.006, +0.078] (7/1/2) | +0.033 [+0.005, +0.060] (8/1/1) | +0.009 [−0.014, +0.031] (6/4/0) |

FedAvg with local fine-tuning matched or slightly exceeded centralized training: small
positive differences overall (pooled ROC-AUC +0.006, higher in all 10 seeds; pooled and
equal-hospital accuracy +0.016 and +0.018 with lower interval bounds near zero), mostly from
VA Long Beach.

**Answer.** In this experiment, keeping the training data at the four hospitals cost little:
FedAvg was close to centralized training on threshold-based metrics but slightly worse in
pooled ROC-AUC and clearly worse for Switzerland, and adding local fine-tuning closed that gap
and ended slightly ahead. Two caveats limit this comparison:

1. The fine-tuned approach uses four hospital-specific models plus extra compute and tuning,
   whereas the centralized approach uses one model with no local adaptation. A centralized
   model followed by the same local fine-tuning would be the matched comparison for the
   second result; it was not run.
2. The differences are of the order of 0.01–0.02 on 920 patients and 10 seeds, with
   uncorrected intervals; they describe this dataset and protocol, not a general ranking of
   the methods.

## 5. Figures

| Figure | File (`results/figures/`, PNG + PDF) |
|---|---|
| 1. Class balance per hospital | `fig1_class_balance` |
| 2. Missing values per predictor and hospital | `fig2_missingness` |
| 3. Validation loss during training (selected learning rates) | `fig3_validation_curves` |
| 4. Change in validation loss during local fine-tuning | `fig4_finetune_curves` |
| 5. Test metrics per hospital, pooled and equal-hospital mean | `fig5_test_metrics` |
| 6. Paired differences between methods | `fig6_paired_differences` |
| 7. ROC curves (test predictions concatenated over seeds) | `fig7_roc_curves` |
| 8. Confusion matrices summed over seeds | `fig8_confusion_matrices` |

See `docs/LIMITATIONS.md` for the limitations that apply to all of these results.
