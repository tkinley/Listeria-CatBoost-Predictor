# Spatially Aware CatBoost Model for Predicting Listeria Presence in Soil 🦠🌾

**Team:** Decaying β-Amyloid  
**Competition:** IAFP AI Benchmarking Student Competition on Predictive Food Safety Models  
**Track:** GIS-based pathogen presence prediction (Listeria in soil)

## 📌 Overview
This repository contains the end-to-end machine learning pipeline developed to predict the presence of *Listeria spp.* in U.S. soil samples. By leveraging soil physiochemical properties, local climate metrics, and land-use classifications, we built a highly robust `CatBoostClassifier`. 

To prevent spatial data leakage—a common pitfall in environmental modeling—we engineered a custom 5-fold `StratifiedGroupKFold` spatial cross-validation strategy based on a 0.25-degree latitude/longitude grid. The pipeline also features Isotonic Regression for probability calibration and mathematical decision threshold tuning.

## ⚙️ System Requirements
As per the competition reproducibility guidelines, this pipeline was developed and tested under the following system specifications:
* **OS:** Windows / Linux / macOS
* **Compute:** Standard Multi-core CPU (CatBoost CPU implementation used; GPU is supported but not required). Minimum 4+ cores recommended.
* **Memory (RAM):** 8 GB minimum (16 GB recommended).
* **Storage:** < 100 MB for datasets and model artifacts.

## 🛠️ Installation & Dependencies
The exact environment dependencies are locked in the `outputs_submission/versions.json` file. The core libraries are:
* Python 3.10+
* `pandas` == 1.4.2
* `numpy` == 1.22.3
* `scikit-learn` == 1.7.2
* `catboost`
* `matplotlib` & `seaborn` (for visualization)

You can install the required packages via pip:
```bash
pip install pandas==1.4.2 numpy==1.22.3 scikit-learn==1.7.2 catboost matplotlib seaborn
