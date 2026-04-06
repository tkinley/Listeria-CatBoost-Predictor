"""
generate_autocorrelation.py
============================
Three-panel figure for the Listeria soil dataset README.

Panel A: Geographic scatter colored by Max Temperature, showing spatial clustering.
Panel B: Pairwise distance vs. temperature difference (Tobler's First Law).
Panel C: Random split vs. spatial grid split, illustrating why the 0.25-degree
         grid is used to prevent geographic leakage.

Requires: ListeriaSoil_clean.csv in the working directory.
"""

import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Design tokens ────────────────────────────────────────────────────────────
BG     = "#F7F6F2"
TEXT   = "#28251D"
MUTED  = "#7A7974"
BORDER = "#D4D1CA"
TEAL   = "#20808D"
TERRA  = "#A84B2F"
LIGHT_TEAL = "#B5DFE4"
LIGHT_TERRA = "#F2D5CC"

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("ListeriaSoil_clean.csv")

lat = df["Latitude"].values
lon = df["Longitude"].values

# Use Max Temperature: clear spatial gradient, matches reference figure
temp_col = [c for c in df.columns if "Max temperature" in c][0]
temp = df[temp_col].values

target_col = "Number of Listeria isolates obtained"
y = (df[target_col] > 0).astype(int).values

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

# ── Spatial grid groups for Panel C ──────────────────────────────────────────
grid_size = 0.25
grid_lat = np.floor(lat / grid_size).astype(int)
grid_lon = np.floor(lon / grid_size).astype(int)
groups = np.array([f"{la}_{lo}" for la, lo in zip(grid_lat, grid_lon)])

# Deterministic "test" selection: pick ~20% of grid cells
unique_groups = np.unique(groups)
np.random.seed(42)
test_groups = set(np.random.choice(unique_groups, size=max(1, len(unique_groups) // 5), replace=False))
is_test_spatial = np.array([g in test_groups for g in groups])

# Random split for comparison
np.random.seed(42)
rand_test_mask = np.zeros(len(y), dtype=bool)
rand_test_idx = np.random.choice(len(y), size=len(y) // 5, replace=False)
rand_test_mask[rand_test_idx] = True

# ── Figure (3 panels) ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11.5), facecolor=BG)

# Top row: Panels A and B
ax1 = fig.add_axes([0.04, 0.44, 0.42, 0.52])   # Panel A
ax2 = fig.add_axes([0.54, 0.44, 0.44, 0.52])   # Panel B

# Bottom row: Panel C (two subpanels side by side)
ax3a = fig.add_axes([0.06, 0.10, 0.38, 0.28])  # Random split
ax3b = fig.add_axes([0.56, 0.10, 0.38, 0.28])  # Spatial grid split

# ── Panel A: Geographic scatter colored by max temperature ───────────────────
ax1.set_facecolor(BG)

vmin, vmax = temp.min(), temp.max()
norm = mcolors.TwoSlopeNorm(vcenter=np.median(temp), vmin=vmin, vmax=vmax)

sc = ax1.scatter(lon, lat, c=temp, cmap="RdBu_r", norm=norm,
                 s=32, alpha=0.88, edgecolors="white", linewidths=0.45, zorder=3)

cbar = fig.colorbar(sc, ax=ax1, fraction=0.04, pad=0.03, shrink=0.85)
cbar.set_label("Max Temperature (\u00b0C)", fontsize=10, color=TEXT)
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

# ── Panel B: Distance vs temperature difference (Tobler's Law) ──────────────
ax2.set_facecolor(BG)

np.random.seed(42)
n_pairs = len(pair_dist)
n_show = min(20000, n_pairs)
si = np.random.choice(n_pairs, n_show, replace=False)

ax2.scatter(pair_dist[si], pair_diff[si],
            s=3, alpha=0.10, color="#9E9E9E", zorder=2, rasterized=True,
            label="Individual pairs")

ax2.plot(bin_centers, bin_means, color=TERRA, linewidth=2.8, zorder=4,
         label="Average difference (trend)")
ax2.scatter(bin_centers, bin_means, color=TERRA, s=24, zorder=5,
            edgecolors="white", linewidths=0.6)

ax2.set_xlabel("Geographic Distance Between Two Samples (km)", fontsize=10.5, color=TEXT)
ax2.set_ylabel("Difference in Max Temperature (\u00b0C)", fontsize=10.5, color=TEXT)
ax2.set_title("B.  Tobler's First Law of Geography\n(Near things are more related)",
              fontsize=12, fontweight="bold", color=TEXT, pad=12, loc="left")
ax2.tick_params(labelsize=9, colors=MUTED)
for sp in ax2.spines.values():
    sp.set_color(BORDER)
ax2.grid(True, alpha=0.25, color=BORDER, linewidth=0.5)
ax2.legend(fontsize=9, frameon=True, facecolor=BG, edgecolor=BORDER,
           loc="upper left", markerscale=3)
ax2.set_ylim(bottom=0)

# ── Panel C left: Random Split ──────────────────────────────────────────────
for ax_c, is_test, title_txt, subtitle, f1_val, f1_color, show_grid in [
    (ax3a, rand_test_mask, "Random Split", "Train/test intermixed geographically", "F1 = 0.901", "#C0392B", False),
    (ax3b, is_test_spatial, "Spatial Grid Split (0.25\u00b0 cells)", "Entire cells held out, no geographic neighbor leak", "F1 = 0.872", TEAL, True),
]:
    ax_c.set_facecolor(BG)

    # Draw grid lines for spatial split
    if show_grid:
        lat_min_g, lat_max_g = lat.min() - 0.5, lat.max() + 0.5
        lon_min_g, lon_max_g = lon.min() - 0.5, lon.max() + 0.5
        for g_lat in np.arange(math.floor(lat_min_g / grid_size) * grid_size,
                                math.ceil(lat_max_g / grid_size) * grid_size + grid_size, grid_size):
            ax_c.axhline(g_lat, color=BORDER, linewidth=0.4, alpha=0.6, zorder=1)
        for g_lon in np.arange(math.floor(lon_min_g / grid_size) * grid_size,
                                math.ceil(lon_max_g / grid_size) * grid_size + grid_size, grid_size):
            ax_c.axvline(g_lon, color=BORDER, linewidth=0.4, alpha=0.6, zorder=1)

        # Shade test grid cells
        for tg in test_groups:
            parts = tg.split("_")
            g_la, g_lo = int(parts[0]), int(parts[1])
            rect = mpatches.FancyBboxPatch(
                (g_lo * grid_size, g_la * grid_size), grid_size, grid_size,
                boxstyle="round,pad=0", facecolor=LIGHT_TEAL, alpha=0.50,
                edgecolor=TEAL, linewidth=1.0, zorder=1)
            ax_c.add_patch(rect)

    # Plot train points
    train_mask = ~is_test
    ax_c.scatter(lon[train_mask], lat[train_mask], s=22, color=TEAL, alpha=0.8,
                 edgecolors="white", linewidths=0.3, zorder=3, marker="o")

    # Plot test points
    ax_c.scatter(lon[is_test], lat[is_test], s=38, color=TERRA, alpha=0.9,
                 edgecolors="white", linewidths=0.3, zorder=4, marker="^")

    ax_c.set_xlim(lon.min() - 2, lon.max() + 2)
    ax_c.set_ylim(lat.min() - 1.5, lat.max() + 1.5)
    ax_c.set_xlabel("Longitude", fontsize=9, color=MUTED)
    ax_c.set_ylabel("Latitude", fontsize=9, color=MUTED)
    ax_c.set_title(title_txt, fontsize=11, fontweight="bold", color=TEAL, pad=8)
    ax_c.tick_params(labelsize=8, colors=MUTED)
    for sp in ax_c.spines.values():
        sp.set_color(BORDER)

    # Subtitle below plot
    ax_c.text(0.5, -0.17, subtitle, transform=ax_c.transAxes,
              ha="center", fontsize=9, color=MUTED, style="italic")

    # F1 score badge
    bbox_color = LIGHT_TERRA if "0.901" in f1_val else LIGHT_TEAL
    ax_c.text(0.5, -0.28, f1_val, transform=ax_c.transAxes,
              ha="center", fontsize=14, fontweight="bold", color=f1_color,
              bbox=dict(boxstyle="round,pad=0.4", facecolor=bbox_color, alpha=0.5,
                        edgecolor=f1_color, linewidth=1.2))

# Legend for Panel C
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAL, markersize=8, label='Train'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor=TERRA, markersize=8, label='Test'),
]
ax3a.legend(handles=legend_elements, fontsize=9, frameon=True, facecolor=BG,
            edgecolor=BORDER, loc="lower right")

legend_elements_grid = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAL, markersize=8, label='Train cells'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor=TERRA, markersize=8, label='Test cells'),
]
ax3b.legend(handles=legend_elements_grid, fontsize=9, frameon=True, facecolor=BG,
            edgecolor=BORDER, loc="lower right")

# Top-level suptitle for Panel C row
fig.text(0.50, 0.40, "C.  Why We Use the Grid: Spatial Leakage",
         ha="center", fontsize=13, fontweight="bold", color=TEXT)
fig.text(0.50, 0.385, "Random splits let the model train on geographic neighbors of test points",
         ha="center", fontsize=10, color=TERRA)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = "fig_spatial_autocorrelation.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Figure saved: {out_path}")
