"""
generate_autocorrelation.py
============================
Generates a two-panel spatial autocorrelation figure for the Listeria soil dataset.

Panel 1: Geographic scatter of sampling sites colored by Max Temperature (C),
         demonstrating that nearby sites share similar environmental conditions.
Panel 2: Pairwise geographic distance vs. absolute temperature difference,
         illustrating Tobler's First Law — near things are more related.

This figure motivates the spatial cross-validation strategy used in the project.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Design tokens ────────────────────────────────────────────────────────────
BG     = "#F7F6F2"
TEXT   = "#28251D"
MUTED  = "#7A7974"
BORDER = "#D4D1CA"
TEAL   = "#20808D"
TERRA  = "#A84B2F"

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("ListeriaSoil_clean.csv")

lat = df["Latitude"].values
lon = df["Longitude"].values

# Use Max Temperature — clear spatial gradient, matches reference figure
temp_col = [c for c in df.columns if "Max temperature" in c][0]
temp = df[temp_col].values

# ── Pairwise haversine distances (km) ────────────────────────────────────────
def haversine_matrix(lat, lon):
    """Vectorised pairwise haversine distance matrix in km."""
    lat_r = np.radians(lat)[:, None]
    lon_r = np.radians(lon)[:, None]
    dlat = lat_r - lat_r.T
    dlon = lon_r - lon_r.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(lat_r.T) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

print("Computing pairwise distances...")
dist_matrix = haversine_matrix(lat, lon)

# Upper-triangle pairs
idx = np.triu_indices(len(lat), k=1)
pair_dist = dist_matrix[idx]
pair_diff = np.abs(temp[idx[0]] - temp[idx[1]])

# ── Binned trend line ────────────────────────────────────────────────────────
n_bins = 30
bin_edges = np.linspace(0, pair_dist.max(), n_bins + 1)
bin_centers, bin_means = [], []
for i in range(n_bins):
    mask = (pair_dist >= bin_edges[i]) & (pair_dist < bin_edges[i + 1])
    if mask.sum() > 10:
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        bin_means.append(np.mean(pair_diff[mask]))
bin_centers = np.array(bin_centers)
bin_means = np.array(bin_means)

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), facecolor=BG,
                          gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.30})

# --- Panel A: Geographic scatter colored by max temperature -----------------
ax1 = axes[0]
ax1.set_facecolor(BG)

# Diverging palette centred on 0 °C
vmin, vmax = temp.min(), temp.max()
norm = mcolors.TwoSlopeNorm(vcenter=np.median(temp), vmin=vmin, vmax=vmax)

sc = ax1.scatter(lon, lat, c=temp, cmap="RdBu_r", norm=norm,
                 s=32, alpha=0.88, edgecolors="white", linewidths=0.45, zorder=3)

cbar = fig.colorbar(sc, ax=ax1, fraction=0.04, pad=0.03, shrink=0.85)
cbar.set_label("Max Temperature (°C)", fontsize=10, color=TEXT)
cbar.ax.tick_params(labelsize=9, colors=MUTED)
cbar.outline.set_color(BORDER)

ax1.set_xlabel("Longitude", fontsize=10.5, color=TEXT)
ax1.set_ylabel("Latitude", fontsize=10.5, color=TEXT)
ax1.set_title("A.  Visualizing Spatial Correlation\n(Similar temperatures cluster together)",
              fontsize=12, fontweight="bold", color=TEXT, pad=12, loc="left")
ax1.tick_params(labelsize=9, colors=MUTED)
for sp in ax1.spines.values():
    sp.set_color(BORDER)
ax1.grid(True, alpha=0.25, color=BORDER, linewidth=0.5)

# --- Panel B: Distance vs temperature difference (Tobler's Law) ------------
ax2 = axes[1]
ax2.set_facecolor(BG)

# Subsample for plot legibility
np.random.seed(42)
n_pairs = len(pair_dist)
n_show = min(20000, n_pairs)
si = np.random.choice(n_pairs, n_show, replace=False)

ax2.scatter(pair_dist[si], pair_diff[si],
            s=3, alpha=0.10, color="#9E9E9E", zorder=2, rasterized=True,
            label="Individual pairs")

# Trend line
ax2.plot(bin_centers, bin_means, color=TERRA, linewidth=2.8, zorder=4,
         label="Average difference (trend)")
ax2.scatter(bin_centers, bin_means, color=TERRA, s=24, zorder=5,
            edgecolors="white", linewidths=0.6)

ax2.set_xlabel("Geographic Distance Between Two Samples (km)", fontsize=10.5, color=TEXT)
ax2.set_ylabel("Difference in Max Temperature (°C)", fontsize=10.5, color=TEXT)
ax2.set_title("B.  Tobler's First Law of Geography\n(Near things are more related)",
              fontsize=12, fontweight="bold", color=TEXT, pad=12, loc="left")
ax2.tick_params(labelsize=9, colors=MUTED)
for sp in ax2.spines.values():
    sp.set_color(BORDER)
ax2.grid(True, alpha=0.25, color=BORDER, linewidth=0.5)
ax2.legend(fontsize=9, frameon=True, facecolor=BG, edgecolor=BORDER,
           loc="upper left", markerscale=3)

# Force y-axis from 0
ax2.set_ylim(bottom=0)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = "fig_spatial_autocorrelation.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Figure saved → {out_path}")
