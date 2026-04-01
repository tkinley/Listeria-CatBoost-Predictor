"""
Logistic Regression baseline under the exact locked spatial protocol.
Matches: StratifiedGroupKFold, 0.25° grid, 5 folds, seed 42.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    precision_recall_curve, auc, brier_score_loss
)
import math
import json

# === Load data ===
df = pd.read_csv("/home/user/workspace/ListeriaSoil_clean.csv")
target_col = "Number of Listeria isolates obtained"
y = (df[target_col] > 0).astype(int).values
X = df.drop(columns=[target_col])

feature_names = list(X.columns)
print(f"Samples: {len(y)}, Features: {X.shape[1]}")
print(f"Class balance: {y.sum()} positive / {(1-y).sum()} negative")

# === Spatial groups (0.25° grid) ===
grid_size = 0.25
groups = np.array([
    f"{math.floor(lat / grid_size)}_{math.floor(lon / grid_size)}"
    for lat, lon in zip(df["Latitude"], df["Longitude"])
])
unique_groups = np.unique(groups)
print(f"Grid groups: {len(unique_groups)}")

# === Locked CV protocol ===
cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
# Note: StratifiedGroupKFold doesn't use shuffle/random_state in sklearn;
# the seed 42 in the original is for CatBoost's random_seed.

oof_probs = np.zeros(len(y), dtype=float)
fold_metrics = []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    # StandardScaler (same as Cornell starter)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    
    # Logistic Regression with default regularization
    lr = LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")
    lr.fit(X_train_sc, y_train)
    
    val_prob = lr.predict_proba(X_val_sc)[:, 1]
    oof_probs[val_idx] = val_prob
    
    # Fold-level metrics at locked threshold 0.475
    y_pred_locked = (val_prob >= 0.475).astype(int)
    
    # ROC AUC
    fold_roc = roc_auc_score(y_val, val_prob)
    
    # PR AUC
    prec_arr, rec_arr, _ = precision_recall_curve(y_val, val_prob)
    fold_pr_auc = auc(rec_arr, prec_arr)
    
    # F1, sensitivity, specificity at locked threshold
    fold_f1 = f1_score(y_val, y_pred_locked)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_locked).ravel()
    fold_sens = tp / (tp + fn)
    fold_spec = tn / (tn + fp)
    
    fold_metrics.append({
        "fold": fold_idx,
        "n_val": len(val_idx),
        "roc_auc": fold_roc,
        "pr_auc": fold_pr_auc,
        "f1_locked_0475": fold_f1,
        "sensitivity_locked": fold_sens,
        "specificity_locked": fold_spec,
    })
    
    print(f"Fold {fold_idx}: n={len(val_idx)}, ROC AUC={fold_roc:.4f}, F1(0.475)={fold_f1:.4f}, Sens={fold_sens:.4f}, Spec={fold_spec:.4f}")

# === Pooled OOF metrics at locked threshold 0.475 ===
print("\n=== POOLED OOF METRICS (threshold=0.475) ===")
y_pred_pooled = (oof_probs >= 0.475).astype(int)
pooled_roc = roc_auc_score(y, oof_probs)
prec_arr, rec_arr, _ = precision_recall_curve(y, oof_probs)
pooled_pr_auc = auc(rec_arr, prec_arr)
pooled_f1 = f1_score(y, y_pred_pooled)
tn, fp, fn, tp = confusion_matrix(y, y_pred_pooled).ravel()
pooled_sens = tp / (tp + fn)
pooled_spec = tn / (tn + fp)
pooled_brier = brier_score_loss(y, oof_probs)

print(f"ROC AUC:     {pooled_roc:.6f}")
print(f"PR AUC:      {pooled_pr_auc:.6f}")
print(f"F1:          {pooled_f1:.6f}")
print(f"Sensitivity: {pooled_sens:.6f}")
print(f"Specificity: {pooled_spec:.6f}")
print(f"Brier Score: {pooled_brier:.6f}")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# === Fold-wise mean ± SD ===
print("\n=== FOLD-WISE MEAN ± SD ===")
for metric in ["roc_auc", "pr_auc", "f1_locked_0475", "sensitivity_locked", "specificity_locked"]:
    vals = [f[metric] for f in fold_metrics]
    mean = np.mean(vals)
    sd = np.std(vals, ddof=0)  # population SD to match original
    print(f"{metric}: {mean:.6f} ± {sd:.6f}")

# === F1-optimized threshold on OOF ===
print("\n=== F1-OPTIMIZED THRESHOLD (OOF) ===")
best_f1 = 0
best_thresh = 0.5
for thresh in np.arange(0.1, 0.9, 0.005):
    y_pred_t = (oof_probs >= thresh).astype(int)
    f1_t = f1_score(y, y_pred_t)
    if f1_t > best_f1:
        best_f1 = f1_t
        best_thresh = thresh

print(f"Best threshold: {best_thresh:.3f}")
print(f"Best F1:        {best_f1:.6f}")

# Re-compute all metrics at optimized threshold
y_pred_opt = (oof_probs >= best_thresh).astype(int)
tn2, fp2, fn2, tp2 = confusion_matrix(y, y_pred_opt).ravel()
print(f"Sensitivity:    {tp2/(tp2+fn2):.6f}")
print(f"Specificity:    {tn2/(tn2+fp2):.6f}")
print(f"TP={tp2}, FP={fp2}, FN={fn2}, TN={tn2}")

# === Also compute fold-wise at optimized threshold ===
print("\n=== FOLD-WISE MEAN ± SD (at optimized threshold) ===")
fold_metrics_opt = []
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
    val_prob = oof_probs[val_idx]
    y_val = y[val_idx]
    y_pred_opt_fold = (val_prob >= best_thresh).astype(int)
    f1_opt = f1_score(y_val, y_pred_opt_fold)
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_val, y_pred_opt_fold).ravel()
    fold_metrics_opt.append({
        "f1": f1_opt,
        "sensitivity": tp_f / (tp_f + fn_f),
        "specificity": tn_f / (tn_f + fp_f),
    })

for metric in ["f1", "sensitivity", "specificity"]:
    vals = [f[metric] for f in fold_metrics_opt]
    print(f"{metric}: {np.mean(vals):.6f} ± {np.std(vals, ddof=0):.6f}")

# === Summary comparison ===
print("\n" + "="*60)
print("SUMMARY: LR vs CatBoost under locked spatial protocol")
print("="*60)
print(f"{'Metric':<20} {'LR (pooled OOF)':<20} {'CatBoost (pooled OOF)':<20}")
print(f"{'ROC AUC':<20} {pooled_roc:<20.4f} {'0.9362':<20}")
print(f"{'PR AUC':<20} {pooled_pr_auc:<20.4f} {'0.9324':<20}")
print(f"{'F1 (t=0.475)':<20} {pooled_f1:<20.4f} {'0.8719':<20}")
print(f"{'Sensitivity':<20} {pooled_sens:<20.4f} {'0.8971':<20}")
print(f"{'Specificity':<20} {pooled_spec:<20.4f} {'0.8392':<20}")
print(f"{'Brier Score':<20} {pooled_brier:<20.4f} {'0.1008':<20}")
print(f"{'F1 (opt thresh)':<20} {best_f1:<20.4f} {'0.8719':<20}")
print(f"{'Opt threshold':<20} {best_thresh:<20.3f} {'0.475':<20}")
