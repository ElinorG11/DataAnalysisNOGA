#!/usr/bin/env python3
"""
CO2 intensity vs. instantaneous renewable share.
Bin CO2 emission ratio by ren_share in 5% buckets (same bins as MAPE analysis).
Show mean ± std per bin, coloured by season, with count bars on secondary axis.
Provides quantitative link between renewable penetration and grid emissions.
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))

from forecast_ml_v2 import load_all_features, get_bad_days

print("Loading data …")
df_raw = load_all_features()
bad    = get_bad_days(df_raw)
df     = df_raw[~df_raw["Date"].isin(bad)].copy()

# ── Load CO2 data and merge ───────────────────────────────────────────────────
co2 = pd.read_csv(DATA_DIR / "noga_co2_5min.csv")
co2["key"] = co2["Date"].astype(str) + " " + co2["Time"].astype(str)
df["key"]  = df["Date"].astype(str)  + " " + df["Time"].astype(str)

co2_col = "Emission Ratio (tCO2/MWh)"
co2_sub = co2[["key", co2_col]].copy()
co2_sub[co2_col] = pd.to_numeric(co2_sub[co2_col], errors="coerce")
# Filter implausible values
co2_sub = co2_sub[co2_sub[co2_col].between(0.05, 1.5)]

df = df.merge(co2_sub, on="key", how="left")
df = df.dropna(subset=[co2_col, "ren_share"])

# ── Bin by renewable share (5% bins, 0–60%) ───────────────────────────────────
bins   = np.arange(0, 62, 5)
labels = [f"{b}–{b+5}%" for b in bins[:-1]]
df["ren_bin"] = pd.cut(df["ren_share"], bins=bins, labels=labels,
                        right=False, include_lowest=True)

grp = df.groupby("ren_bin", observed=True)[co2_col].agg(["mean", "std", "count"])
grp.index = grp.index.astype(str)

# Remove bins with < 50 observations (too sparse)
grp = grp[grp["count"] >= 50].copy()
x   = np.arange(len(grp))

# ── Seasonal breakdown: mean CO2 per bin per season ───────────────────────────
season_colors = {
    "Winter": "#2471a3",
    "Spring": "#27ae60",
    "Summer": "#e74c3c",
    "Autumn": "#e67e22",
}
seas_grp = (df.groupby(["ren_bin", "season"], observed=True)[co2_col]
              .mean()
              .unstack("season"))
seas_grp = seas_grp.loc[grp.index]  # align to filtered bins

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.linewidth":   0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

fig, ax = plt.subplots(figsize=(9, 4.5))

# Secondary axis: count bars
ax_r = ax.twinx()
capped = np.minimum(grp["count"].values.astype(float), 20000)
ax_r.bar(x, capped, width=0.7, color="#bbbbbb", alpha=0.18, zorder=1, linewidth=0)
ax_r.set_ylim(0, 20000 * 5)
ax_r.set_yticks([0, 5000, 10000, 15000, 20000])
ax_r.set_yticklabels(["0", "5k", "10k", "15k", "≥20k"],
                      fontsize=9.5, color="#999999")
ax_r.set_ylabel("Interval count", color="#999999", fontsize=11, labelpad=5)
for sp in ax_r.spines.values():
    sp.set_linewidth(0.4); sp.set_color("#cccccc")
ax_r.tick_params(axis="y", length=2, color="#cccccc")

# Mean CO2 + std band (overall, in black)
ax.fill_between(x,
                grp["mean"].values - grp["std"].values,
                grp["mean"].values + grp["std"].values,
                color="#555555", alpha=0.12, zorder=2, label="±1σ (all seasons)")
ax.plot(x, grp["mean"].values, "o-",
        color="#222222", lw=2.2, ms=6, zorder=5, label="Mean CO₂ (all seasons)")

# Seasonal lines
for season, col in season_colors.items():
    if season in seas_grp.columns:
        ys = seas_grp[season].values
        if np.any(~np.isnan(ys)):
            ax.plot(x, ys, "s--", color=col, lw=1.3, ms=4.5, alpha=0.85,
                    zorder=4, label=season)

ax.set_xticks(x)
ax.set_xticklabels(grp.index.tolist(), rotation=35, ha="right", fontsize=10)
ax.set_xlabel("Instantaneous renewable share (% of demand)", fontsize=12, fontweight="bold", labelpad=4)
ax.set_ylabel("Grid CO₂ intensity (tCO₂/MWh)", fontsize=12, fontweight="bold")
ax.yaxis.grid(True, color="#e0e0e0", lw=0.5, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

ax.legend(loc="upper right", fontsize=10, framealpha=0.92,
          edgecolor="#cccccc", handlelength=2.0)

plt.tight_layout(pad=0.8)
out = DATA_DIR / "fig_co2_vs_ren.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
