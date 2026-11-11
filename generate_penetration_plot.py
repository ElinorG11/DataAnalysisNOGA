#!/usr/bin/env python3
"""
Publication-quality dual-line chart: Forecast Error vs Renewable Penetration.
Section 4.4 of full_grid_paper.tex.

Replaces fig4_forecast_vs_renewable.png with an IEEE-style annotated figure
that highlights the severe DA/SF asymmetry and Spearman rank correlation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

# ── Hardcoded data from Table 9 ───────────────────────────────────────────────
bin_centres = np.array([2.5, 7.5, 12.5, 17.5, 22.5, 27.5,
                         32.5, 37.5, 42.5, 47.5, 52.5, 57.5])
bin_labels  = ["0–5", "5–10", "10–15", "15–20", "20–25", "25–30",
               "30–35", "35–40", "40–45", "45–50", "50–55", "55–60"]
da_mape = np.array([2.91, 3.35, 3.61, 3.40, 3.16, 3.02,
                     3.04, 3.48, 3.92, 4.77, 4.94, 3.86])
sf_mape = np.array([1.19, 0.91, 1.01, 1.20, 1.24, 1.38,
                     1.64, 2.32, 2.99, 4.31, 5.50, 7.26])
counts  = np.array([115862, 14537, 9911, 10496, 13377, 16977,
                     15697, 10581, 6552, 3294, 718, 14])

# Bins with n >= 100 are "reliable"; last bin (n=14) is sparse
N_RELIABLE = 11  # indices 0..10
SPARSE_IDX = 11

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.linewidth":   0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
})

COLOR_DA  = "#c0392b"   # deep red
COLOR_SF  = "#2171b5"   # steel blue
COLOR_BAR = "#aaaaaa"   # count bars

fig, ax = plt.subplots(figsize=(7.2, 4.2))

# ── Count bars on secondary right axis ───────────────────────────────────────
# Cap at 20k so first bin (116k) doesn't visually crush all others
ax_r = ax.twinx()
capped = np.minimum(counts.astype(float), 20000)
ax_r.bar(bin_centres, capped, width=4.3, color=COLOR_BAR, alpha=0.14,
         zorder=1, linewidth=0)
ax_r.set_ylim(0, 20000 * 5.5)  # push bars low relative to MAPE lines
ax_r.set_yticks([0, 5000, 10000, 15000, 20000])
ax_r.set_yticklabels(["0", "5k", "10k", "15k", "≥20k"],
                      fontsize=7.5, color="#999999")
ax_r.set_ylabel("Interval count  (bin 0–5%: $n$=116k, capped)",
                 color="#999999", fontsize=8.0, labelpad=6)
for sp in ax_r.spines.values():
    sp.set_linewidth(0.4)
    sp.set_color("#cccccc")
ax_r.tick_params(axis="y", length=2, color="#cccccc")

# ── Reliable segment (solid lines + filled markers) ───────────────────────────
rel = slice(0, N_RELIABLE)
ax.plot(bin_centres[rel], da_mape[rel], "o-",
        color=COLOR_DA, lw=2.2, ms=6, zorder=4,
        label="DA MAPE  (day-ahead)")
ax.plot(bin_centres[rel], sf_mape[rel], "s-",
        color=COLOR_SF, lw=2.2, ms=6, zorder=4,
        label="SF MAPE  (short-term)")

# ── Sparse last bin (dashed extension + open markers) ─────────────────────────
for col, mape, mk in [(COLOR_DA, da_mape, "o"), (COLOR_SF, sf_mape, "s")]:
    ax.plot([bin_centres[N_RELIABLE - 1], bin_centres[SPARSE_IDX]],
            [mape[N_RELIABLE - 1], mape[SPARSE_IDX]],
            "--", color=col, lw=1.3, alpha=0.50, zorder=3)
    ax.plot(bin_centres[SPARSE_IDX], mape[SPARSE_IDX], mk,
            color=col, ms=6, mfc="white", mec=col, mew=1.6, zorder=5)

ax.text(bin_centres[SPARSE_IDX], sf_mape[SPARSE_IDX] + 0.28,
        "$n{=}14$", fontsize=7.5, color="#777777", ha="center", va="bottom")

# ── Double-headed arrows showing the rise from bin-0 to bin-10 ───────────────
# Placed vertically at x = 55 (just right of the last reliable bin at 52.5)
X_ARROW = 55.4

# SF arrow: 1.19 → 5.50
ax.annotate("",
    xy=(X_ARROW, sf_mape[N_RELIABLE - 1]),
    xytext=(X_ARROW, sf_mape[0]),
    arrowprops=dict(arrowstyle="<->", color=COLOR_SF,
                    lw=1.4, mutation_scale=9))
ax.text(X_ARROW + 1.1, (sf_mape[0] + sf_mape[N_RELIABLE - 1]) / 2,
        r"$\boldsymbol{\times 4.5}$",
        color=COLOR_SF, fontsize=9.5, va="center", ha="left", fontweight="bold")

# DA arrow: 2.91 → 4.94  (offset slightly so arrows don't overlap)
ax.annotate("",
    xy=(X_ARROW - 1.6, da_mape[N_RELIABLE - 1]),
    xytext=(X_ARROW - 1.6, da_mape[0]),
    arrowprops=dict(arrowstyle="<->", color=COLOR_DA,
                    lw=1.4, mutation_scale=9))
ax.text(X_ARROW - 2.8, (da_mape[0] + da_mape[N_RELIABLE - 1]) / 2,
        r"$\boldsymbol{\times 1.7}$",
        color=COLOR_DA, fontsize=9.5, va="center", ha="right", fontweight="bold")

# ── Spearman ρ statistics text box ────────────────────────────────────────────
stats_str = (
    r"Spearman $\rho$ (0–55\% bins, $n{=}11$):" "\n"
    r"SF:  $\rho = 0.973$,  $p < 0.0001$" "\n"
    r"DA:  $\rho = 0.673$,  $p = 0.023$"
)
ax.text(0.026, 0.975, stats_str,
        transform=ax.transAxes,
        fontsize=8.2, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                  edgecolor="#cccccc", linewidth=0.7, alpha=0.93))

# ── Axis formatting ───────────────────────────────────────────────────────────
ax.set_xlim(-1, 63)
ax.set_ylim(0, 8.6)

ax.set_xticks(bin_centres)
ax.set_xticklabels([f"{lb}%" for lb in bin_labels],
                    rotation=35, ha="right", fontsize=8.2)
ax.set_xlabel("Instantaneous renewable share (% of demand)", fontsize=10, labelpad=4)
ax.set_ylabel("Forecast error (MAPE %)", fontsize=10)

ax.yaxis.grid(True, color="#dddddd", linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)

ax.legend(loc="upper left", fontsize=8.8,
          frameon=True, framealpha=0.93,
          edgecolor="#cccccc", handlelength=2.2,
          bbox_to_anchor=(0.026, 0.72),   # below the stats box
          borderpad=0.5)

plt.tight_layout(pad=0.7)
out = DATA_DIR / "fig4_forecast_vs_renewable.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
