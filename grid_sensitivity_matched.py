"""
grid_sensitivity_matched.py
============================
Clean spatial grid sensitivity analysis for the Listeria CatBoost project.

Protocol (identical across all three grid sizes):
  - Dataset : ListeriaSoil_clean.csv
  - Target  : y = 1 if isolates_obtained > 0 else 0
  - CV      : StratifiedGroupKFold, n_splits=5, shuffle=True, random_state=42
  - Model   : CatBoostClassifier with exact locked hyperparameters
  - The ONLY variable is the spatial grid size (0.25 / 0.50 / 1.00 degrees)

Two threshold approaches:
  A. Fixed threshold = 0.475 (locked operational threshold)
  B. Per-grid OOF-tuned threshold (max F1 on pooled OOF predictions)

Outputs saved to:
  outputs_grid_sensitivity_matched_protocol/
"""

import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    precision_recall_curve, auc, brier_score_loss,
)
from catboost import CatBoostClassifier, Pool

# ─── Output directory ────────────────────────────────────────────────────────
OUT = "outputs_grid_sensitivity_matched_protocol"
os.makedirs(OUT, exist_ok=True)

# ─── Locked protocol (read from eval_lock.json to stay faithful) ─────────────
with open("outputs_submission/eval_lock.json") as f:
    LOCKED = json.load(f)

FIXED_THRESHOLD = 0.475   # locked operational threshold
GRID_SIZES      = [0.25, 0.50, 1.00]
N_SPLITS        = LOCKED["n_splits"]      # 5
SEED            = LOCKED["seed"]          # 42

# Exact hyperparameters from notebook Cell 15 — do not change
CAT_PARAMS = dict(
    loss_function      = "Logloss",
    eval_metric        = "AUC",
    iterations         = 20000,
    learning_rate      = 0.03,
    depth              = 8,
    l2_leaf_reg        = 3.0,
    random_seed        = SEED,
    allow_writing_files= False,
    verbose            = 0,        # silent during batch run; set to 200 for debugging
    od_type            = "Iter",
    od_wait            = 300,
)

# ─── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("../ListeriaSoil_clean.csv")
TARGET_COL = "Number of Listeria isolates obtained"
y = (df[TARGET_COL] > 0).astype(int).values
X = df.drop(columns=[TARGET_COL])
print(f"Dataset: {len(y)} samples, {X.shape[1]} features")
print(f"Class balance: {y.sum()} pos / {(1-y).sum()} neg")

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
cat_idx  = [X.columns.get_loc(c) for c in cat_cols]

# ─── Helper functions (copied faithfully from notebook Cell 9) ────────────────
def pr_auc_score(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return auc(r, p)

def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec

def tune_threshold_for_f1(y_true, y_prob, t_min=0.05, t_max=0.95, steps=181):
    ts = np.linspace(t_min, t_max, steps)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        f1 = f1_score(y_true, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_t  = float(t)
    return best_t, float(best_f1)

def summarize_probs(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    sens, spec = sensitivity_specificity(y_true, y_pred)
    return {
        "roc_auc"    : float(roc_auc_score(y_true, y_prob)),
        "pr_auc"     : float(pr_auc_score(y_true, y_prob)),
        "f1"         : float(f1_score(y_true, y_pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "brier"      : float(brier_score_loss(y_true, y_prob)),
        "threshold"  : float(threshold),
    }

# Group construction (from notebook Cell 11)
def make_spatial_groups(df_features, grid_size_deg):
    lat_bin = np.floor(df_features["Latitude"].astype(float)  / grid_size_deg).astype(int)
    lon_bin = np.floor(df_features["Longitude"].astype(float) / grid_size_deg).astype(int)
    return (lat_bin.astype(str) + "_" + lon_bin.astype(str)).values

# ─── Main sensitivity loop ────────────────────────────────────────────────────
all_rows  = []    # one row per grid size, full metrics
fold_rows = []    # one row per (grid size, fold), for mean±SD

for grid_deg in GRID_SIZES:
    label = f"{grid_deg:.2f}deg"
    print(f"\n{'='*60}")
    print(f"  Grid size: {grid_deg}°")
    print(f"{'='*60}")

    groups = make_spatial_groups(X, grid_size_deg=grid_deg)
    n_groups = len(set(groups))
    print(f"  Number of spatial groups: {n_groups}")

    # CV splits — identical protocol for every grid size
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, y, groups))

    oof = np.zeros(len(y), dtype=float)

    t0 = time.time()
    for fold_i, (tr, va) in enumerate(splits, start=1):
        X_tr, y_tr = X.iloc[tr], y[tr]
        X_va, y_va = X.iloc[va], y[va]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        valid_pool = Pool(X_va, y_va, cat_features=cat_idx)

        model = CatBoostClassifier(**CAT_PARAMS)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        va_prob = model.predict_proba(valid_pool)[:, 1]
        oof[va] = va_prob

        fold_roc = roc_auc_score(y_va, va_prob)
        fold_f1_fixed = f1_score(y_va, (va_prob >= FIXED_THRESHOLD).astype(int))
        print(f"  Fold {fold_i}: AUC={fold_roc:.4f}  F1@{FIXED_THRESHOLD}={fold_f1_fixed:.4f}")

        fold_rows.append({
            "grid_deg": grid_deg,
            "fold"    : fold_i,
            "n_val"   : len(va),
            "roc_auc" : fold_roc,
            "f1_fixed": fold_f1_fixed,
        })

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min")

    # ── Pooled OOF metrics ────────────────────────────────────────────────────
    # A. Fixed threshold
    m_fixed = summarize_probs(y, oof, threshold=FIXED_THRESHOLD)

    # B. Per-grid OOF-tuned threshold
    best_t, best_f1 = tune_threshold_for_f1(y, oof)
    m_tuned = summarize_probs(y, oof, threshold=best_t)

    row = {
        "grid_deg"          : grid_deg,
        "n_groups"          : n_groups,
        # Fixed threshold
        "roc_auc"           : m_fixed["roc_auc"],
        "pr_auc"            : m_fixed["pr_auc"],
        "f1_fixed_0475"     : m_fixed["f1"],
        "sensitivity_fixed" : m_fixed["sensitivity"],
        "specificity_fixed" : m_fixed["specificity"],
        "brier"             : m_fixed["brier"],
        # Per-grid tuned threshold
        "best_threshold"    : best_t,
        "f1_tuned"          : best_f1,
    }
    all_rows.append(row)
    print(f"  Pooled OOF @ fixed t=0.475 : AUC={m_fixed['roc_auc']:.4f}  F1={m_fixed['f1']:.4f}")
    print(f"  Pooled OOF @ tuned t={best_t:.3f} : F1={best_f1:.4f}")

    # Save OOF predictions for this grid
    oof_path = f"{OUT}/oof_predictions_grid{label}.csv"
    pd.DataFrame({"y": y, "oof_prob": oof}).to_csv(oof_path, index=False)
    print(f"  OOF saved → {oof_path}")

# ─── Build output tables ──────────────────────────────────────────────────────
results = pd.DataFrame(all_rows)
fold_df  = pd.DataFrame(fold_rows)

# Add fold-wise mean ± SD for ROC AUC and F1
fold_stats = (
    fold_df.groupby("grid_deg")
    .agg(
        roc_auc_mean=("roc_auc",  "mean"),
        roc_auc_sd  =("roc_auc",  "std"),
        f1_fixed_mean=("f1_fixed","mean"),
        f1_fixed_sd  =("f1_fixed","std"),
    )
    .round(6)
    .reset_index()
)
results = results.merge(fold_stats, on="grid_deg")

# Full summary table
print("\n\n" + "="*60)
print("FULL SUMMARY TABLE")
print("="*60)
print(results.to_string(index=False))

# Compact presentation table
compact = results[[
    "grid_deg", "roc_auc", "f1_fixed_0475",
    "f1_tuned", "best_threshold",
]].copy()
compact.columns = [
    "Grid (deg)", "ROC AUC", "F1 (t=0.475)",
    "F1 (tuned t)", "Best threshold",
]
print("\nCOMPACT PRESENTATION TABLE")
print(compact.to_string(index=False))

# ─── Save tables ─────────────────────────────────────────────────────────────
results.to_csv(f"{OUT}/grid_sensitivity_full.csv", index=False)
compact.to_csv(f"{OUT}/grid_sensitivity_compact.csv", index=False)
fold_df.to_csv(f"{OUT}/grid_sensitivity_fold_metrics.csv", index=False)
results.to_json(f"{OUT}/grid_sensitivity_full.json", orient="records", indent=2)

# Protocol snapshot
protocol = {
    "cv"               : "StratifiedGroupKFold",
    "n_splits"         : N_SPLITS,
    "shuffle"          : True,
    "random_state"     : SEED,
    "fixed_threshold"  : FIXED_THRESHOLD,
    "grid_sizes_tested": GRID_SIZES,
    "cat_params"       : CAT_PARAMS,
    "dataset"          : "ListeriaSoil_clean.csv",
    "note"             : (
        "All three grid sizes evaluated under identical protocol. "
        "Only the spatial grouping changes. "
        "Exploratory results (GroupKFold, per-run thresholds) are NOT mixed here."
    ),
}
with open(f"{OUT}/matched_protocol_config.json", "w") as f:
    json.dump(protocol, f, indent=2)

# ─── Bar chart ───────────────────────────────────────────────────────────────
C = dict(accent="#01696F", terra="#A84B2F", blue="#1A4F8A",
         bg="#FFFFFF", muted="#5C6370", border="#DADDE3", text="#28251D")

fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=C["bg"])
ax.set_facecolor(C["bg"])

x     = np.arange(len(GRID_SIZES))
w     = 0.26
lbls  = [f"{g}°" for g in GRID_SIZES]

b1 = ax.bar(x - w,   results["roc_auc"],       w, label="ROC AUC",          color=C["accent"], alpha=0.90, zorder=3)
b2 = ax.bar(x,       results["f1_fixed_0475"], w, label="F1 (t=0.475)",     color=C["terra"],  alpha=0.85, zorder=3)
b3 = ax.bar(x + w,   results["f1_tuned"],      w, label="F1 (tuned t)",     color=C["blue"],   alpha=0.80, zorder=3)

ax.set_ylim(0.82, 0.97)
ax.set_xticks(x)
ax.set_xticklabels(lbls, fontsize=13)
ax.set_ylabel("Score", fontsize=11)
ax.set_title(
    "Matched-Protocol Grid Sensitivity\n"
    "StratifiedGroupKFold · Same model & hyperparameters · Same seed",
    fontsize=12, fontweight="bold", color=C["text"], pad=10
)
ax.yaxis.grid(True, zorder=0, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(C["border"])
ax.spines["bottom"].set_color(C["border"])

for bar in [*b1, *b2, *b3]:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.001,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8.5,
            color=C["muted"])

ax.legend(fontsize=10, frameon=True, loc="lower left")
plt.tight_layout(pad=0.6)
plt.savefig(f"{OUT}/grid_sensitivity_chart.png", dpi=180,
            bbox_inches="tight", facecolor=C["bg"])
plt.close()
print(f"\nChart saved → {OUT}/grid_sensitivity_chart.png")

# ─── List all outputs ─────────────────────────────────────────────────────────
print("\nAll output files:")
for fn in sorted(os.listdir(OUT)):
    sz = os.path.getsize(f"{OUT}/{fn}")
    print(f"  {fn}  ({sz:,} bytes)")

print("\nDone.")
