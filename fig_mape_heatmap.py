#!/usr/bin/env python3
"""
Hourly MAPE heatmap: hour-of-day (y) × calendar month (x).
Side-by-side: DA MAPE (left) | SF MAPE (right).
Reveals seasonal and diurnal concentration of forecast error.
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

from forecast_ml_v2 import (
    load_all_features, get_bad_days, mape_arr, DA_MIN, DA_MAX
)

SF_MIN = 2000  # SF forecasts below this are data artifacts (e.g. DST zeros)

print("Loading data …")
df_raw = load_all_features()
bad    = get_bad_days(df_raw)
df     = df_raw[~df_raw["Date"].isin(bad)].copy()

# ── Compute interval-level MAPE ───────────────────────────────────────────────
act  = df["Current Demand (MW)"].values
nda  = df["Day-Ahead Demand Forecast (MW)"].where(
       df["Day-Ahead Demand Forecast (MW)"].between(DA_MIN, DA_MAX)).values
nsf  = df["Updated Demand Forecast (MW)"].where(
       df["Updated Demand Forecast (MW)"] >= SF_MIN).values

df["mape_da"] = mape_arr(act, nda)
df["mape_sf"] = mape_arr(act, nsf)

# ── Pivot: hour (0–23) × month (1–12) ────────────────────────────────────────
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

piv_da = (df.groupby(["hour", "month"])["mape_da"]
            .mean()
            .unstack("month")    # columns = month 1..12
            .reindex(columns=range(1,13)))

piv_sf = (df.groupby(["hour", "month"])["mape_sf"]
            .mean()
            .unstack("month")
            .reindex(columns=range(1,13)))

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":       10,
    "axes.linewidth":  0.8,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5),
                                gridspec_kw={"wspace": 0.10})

def draw_heatmap(ax, piv, title, cmap, vmin, vmax, ylabel=True, cbar=False):
    im = ax.imshow(piv.values, aspect="auto", origin="upper",
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[-0.5, 11.5, 23.5, -0.5])
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=10)
    ax.set_yticks(range(0, 24, 2))
    if ylabel:
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Month", fontsize=12, fontweight="bold")
    if ylabel:
        ax.set_ylabel("Hour of day", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)

    # Annotate each cell with its value
    for r in range(piv.shape[0]):
        for c in range(piv.shape[1]):
            val = piv.values[r, c]
            if np.isnan(val):
                continue
            text_color = "white" if val > (vmin + 0.6 * (vmax - vmin)) else "black"
            ax.text(c, r, f"{val:.1f}",
                    ha="center", va="center", fontsize=7.5,
                    color=text_color)

    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("MAPE (%)", fontsize=11)
        cb.ax.tick_params(labelsize=10)
    return im

# Shared colour scale for comparability — use max across both
vmax = max(np.nanpercentile(piv_da.values, 97),
           np.nanpercentile(piv_sf.values, 97))

im1 = draw_heatmap(ax1, piv_da, "(a)  DA MAPE (%)",
                   "YlOrRd", vmin=0, vmax=vmax, ylabel=True, cbar=False)
im2 = draw_heatmap(ax2, piv_sf, "(b)  SF MAPE (%)",
                   "YlOrRd", vmin=0, vmax=vmax, ylabel=False, cbar=False)

# Shared colourbar on the right
cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.76])
cb = fig.colorbar(im2, cax=cbar_ax)
cb.set_label("MAPE (%)", fontsize=11)
cb.ax.tick_params(labelsize=10)

plt.tight_layout(rect=[0, 0, 0.91, 1])
out = DATA_DIR / "fig_mape_heatmap.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
