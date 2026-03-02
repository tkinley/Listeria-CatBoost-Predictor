# Spatially Aware CatBoost Model for Predicting Listeria Presence in Soil

**Team:** Decaying β-Amyloid  
**Competition:** IAFP AI Benchmarking Student Competition on Predictive Food Safety Models  
**Track:** GIS-based pathogen presence prediction (Listeria in soil)

## Overview
This repository contains an end-to-end benchmark pipeline to predict *Listeria* spp. presence in U.S. soil samples using soil chemistry, climate, land use, and geographic coordinates. The workflow emphasizes spatially aware validation, threshold tuning, and probability calibration for operational decision-making.

## Dataset and citation
Cornell Food Safety ML Repository: “Listeria in soil”  
Primary source publication:  
Liao, J., Guo, X., Weller, D.L. et al. (2021) Nationwide genomic atlas of soil-dwelling *Listeria* reveals effects of selection and population ecology on pangenome evolution. *Nature Microbiology* 6, 1021–1030.

## Task definition
Outcome column: `Number of Listeria isolates obtained`  
Binary label:
- y = 1 if isolates > 0
- y = 0 otherwise

## Locked evaluation protocol
A locked spatial cross-validation protocol is used to reduce spatial leakage:
- CV: StratifiedGroupKFold
- Spatial grouping: 0.25° latitude/longitude grid cells
- Folds: 5
- Seed: 42
- Threshold policy: maximize F1 on out-of-fold predictions

Locked configuration: `outputs_submission/eval_lock.json`

## Final benchmark results (locked protocol)
Source: `outputs_submission/overall_metrics.json`

| Metric | Value |
|---|---:|
| ROC AUC | 0.936 |
| PR AUC | 0.932 |
| F1 | 0.872 |
| Sensitivity | 0.897 |
| Specificity | 0.839 |
| Locked threshold (F1-optimized) | 0.475 |

## Calibration and actionability
Isotonic calibration improved probability accuracy (Brier score 0.1008 → 0.0945).  
Capacity-based sampling enrichment using calibrated risk score:
- Top 10% highest risk: 98.4% observed positivity (vs 50.0% overall)
- Top 20% highest risk: 96.8% observed positivity
- Top 30% highest risk: 95.2% observed positivity

## Figures
**Figure 1.** Top 10 feature importances  
![Figure 1](outputs_submission/fig_1_feature_importance.png)

**Figure 2.** ROC, PR, confusion matrix (locked threshold), and calibration curve  
![Figure 2](outputs_submission/fig_2_panel_roc_pr_cm_calibration.png)

## Reproducibility

### System requirements
- CPU: 4+ cores recommended
- RAM: 8 GB minimum (16 GB recommended)
- Disk: < 200 MB for artifacts (dataset stored separately)

### Environment
Versions recorded in: `outputs_submission/versions.json`

### Run
Open and run the notebook:
- `final_1.ipynb`

Outputs are written to:
- `outputs_submission/`
