# EXPERIMENTS.md

This document logs exploratory benchmarking experiments performed during model development. The **official submission benchmark** is defined by the locked evaluation configuration in `outputs_submission/eval_lock.json` and the headline metrics in `outputs_submission/overall_metrics.json` (also summarized in `README.md`).

## Dataset and label
- Dataset: `ListeriaSoil_clean.csv`
- Outcome column: `Number of Listeria isolates obtained`
- Binary label: `y = 1` if isolates > 0, else `y = 0`

## Conventions
- Unless stated otherwise, metrics are computed from **out-of-fold (OOF) probabilities**.
- Thresholds are selected by maximizing **F1** on OOF predictions (F1-optimized threshold).
- Metrics reported: ROC AUC, PR AUC, F1, sensitivity, specificity.

---

## A. Official locked submission run
For the final benchmark used in the report and README:
- Locked configuration: `outputs_submission/eval_lock.json`
- Locked metrics: `outputs_submission/overall_metrics.json`

This run uses spatially aware CV and a fixed threshold policy and is the only benchmark used for the final submission.

---

## B. Baseline: Random CV (CatBoost)
Evaluation:
- RepeatedStratifiedKFold, 5 folds, 3 repeats, seed 42 (random splits, not spatially grouped)

| Model | ROC AUC | PR AUC | F1 | Sensitivity | Specificity | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.9459 | 0.9398 | 0.9011 | 0.9228 | 0.8746 | 0.435 |

Notes:
- Random CV can be optimistic for geospatial data due to spatial autocorrelation.
- This baseline is included to show the magnitude of potential leakage versus spatial CV.

---

## C. Spatial CV: grid sensitivity (CatBoost, GroupKFold)
Evaluation:
- GroupKFold, 5 folds
- Spatial groups defined by latitude and longitude grid cells of size `grid_size_deg`

| Spatial grid (deg) | ROC AUC | PR AUC | F1 | Sensitivity | Specificity | Threshold |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.9361 | 0.9268 | 0.8836 | 0.9035 | 0.8585 | 0.490 |
| 0.50 | 0.9336 | 0.9301 | 0.8801 | 0.8971 | 0.8585 | 0.505 |
| 1.00 | 0.9276 | 0.9269 | 0.8717 | 0.9068 | 0.8264 | 0.470 |

Notes:
- Smaller grid sizes create more, smaller groups and typically stricter generalization tests.
- Performance decreases as groups become larger and distribution shift becomes more pronounced.

---

## D. Model comparison under Spatial CV (grid = 0.25, GroupKFold)
Evaluation:
- GroupKFold, 5 folds
- Spatial groups: 0.25° grid cells

| Model | ROC AUC | PR AUC | F1 | Sensitivity | Specificity | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.9361 | 0.9268 | 0.8836 | 0.9035 | 0.8585 | 0.490 |
| LightGBM | 0.9337 | 0.9355 | 0.8707 | 0.8875 | 0.8489 | 0.460 |
| CatBoost + LightGBM (avg probs) | 0.9382 | 0.9356 | 0.8800 | 0.8842 | 0.8746 | 0.540 |
| CatBoost seed ensemble (avg across seeds 42, 202, 777) | 0.9376 | 0.9323 | 0.8809 | 0.9035 | 0.8521 | 0.490 |

Notes:
- Simple probability averaging improved AUC and specificity but did not exceed CatBoost on F1 in this setting.
- Seed ensembling reduced variance but did not materially outperform the best single CatBoost model.

---

## E. Model comparison under Spatial CV (grid = 0.50, GroupKFold)
Evaluation:
- GroupKFold, 5 folds
- Spatial groups: 0.50° grid cells

| Model | ROC AUC | PR AUC | F1 | Sensitivity | Specificity | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.9336 | 0.9301 | 0.8801 | 0.8971 | 0.8585 | 0.505 |
| LightGBM | 0.9288 | 0.9275 | 0.8621 | 0.8842 | 0.8328 | 0.465 |
| CatBoost + LightGBM (avg probs) | 0.9358 | 0.9341 | 0.8709 | 0.8457 | 0.9035 | 0.610 |

Notes:
- Ensemble again increased specificity but reduced sensitivity relative to CatBoost, which can reduce F1 depending on threshold.

---

## F. Calibration and decision policy experiments (OOF-based)
Calibration method:
- Isotonic regression fit on OOF probabilities (no training leakage)

Probability accuracy:
- Brier score (raw): 0.1008
- Brier score (isotonic calibrated): 0.0945

Capacity-based sampling enrichment using calibrated risk score:
- Top 10% highest predicted risk: 98.4% observed positivity vs 50.0% overall
- Top 20% highest predicted risk: 96.8% observed positivity vs 50.0% overall
- Top 30% highest predicted risk: 95.2% observed positivity vs 50.0% overall

Notes:
- Calibration improves interpretability of predicted probabilities as operational risk scores.
- Top-k enrichment provides an actionable surveillance policy when sampling resources are limited.

---

## Reproducing experiments
- Official submission benchmark: run `final_1.ipynb` and use outputs in `outputs_submission/`.
- Exploratory benchmarking: runs were performed in the development notebook (for example `data_trial.ipynb`), using:
  - Random CV baseline
  - Spatial GroupKFold with grid sensitivity
  - LightGBM comparisons and probability averaging ensemble
  - Calibration and top-k policy analysis

For clean review, treat `README.md` and `outputs_submission/overall_metrics.json` as the authoritative locked benchmark for submission.
