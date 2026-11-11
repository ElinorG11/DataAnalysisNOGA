#!/usr/bin/env python3
"""
Rolling 30-day DA MAPE time series with security events marked.
Shows temporal context: seasonal oscillations, event spikes,
and the True Promise II adaptation.
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))

from forecast_ml_v2 import (
    load_all_features, get_bad_days, mape_arr,
    HOUR_FROM, HOUR_TO, DA_MIN, DA_MAX
)

SF_MIN = 2000

print("Loading data …")
df_raw = load_all_features()
bad    = get_bad_days(df_raw)
df     = df_raw[~df_raw["Date"].isin(bad)].copy()

# ── Compute daily DA MAPE (08:00–20:00 window) ────────────────────────────────
daytime = df[(df["hour"] >= HOUR_FROM) & (df["hour"] < HOUR_TO)].copy()
act  = daytime["Current Demand (MW)"].values
nda  = daytime["Day-Ahead Demand Forecast (MW)"].where(
       daytime["Day-Ahead Demand Forecast (MW)"].between(DA_MIN, DA_MAX)).values
daytime = daytime.copy()
daytime["mape_da"] = mape_arr(act, nda)
daytime["mape_sf"] = mape_arr(
    daytime["Current Demand (MW)"].values,
    daytime["Updated Demand Forecast (MW)"].where(
        daytime["Updated Demand Forecast (MW)"] >= SF_MIN).values
)

daily = (daytime.groupby("Date")[["mape_da", "mape_sf"]]
         .mean()
         .dropna()
         .reset_index())
daily["date_dt"] = pd.to_datetime(daily["Date"])
daily = daily.sort_values("date_dt").reset_index(drop=True)

# Rolling 30-day mean
daily["roll_da"] = daily["mape_da"].rolling(30, center=True, min_periods=15).mean()
daily["roll_sf"] = daily["mape_sf"].rolling(30, center=True, min_periods=15).mean()

BASELINE_DA = daily["mape_da"].mean()
BASELINE_SF = daily["mape_sf"].mean()

# ── Security events (multi-day events: list all dates; diamond = mean of available) ──
EVENTS = [
    {"label": "Oct 8\n2023",         "dates": ["2023-10-08"],                              "vline": "2023-10-08", "ha": "right"},
    {"label": "TP-I\nApr 2024",      "dates": ["2024-04-13", "2024-04-14"],                "vline": "2024-04-13", "ha": "left"},
    {"label": "TP-II\nOct 2024",     "dates": ["2024-10-01"],                              "vline": "2024-10-01", "ha": "left"},
    {"label": "Rising Lion\nJun 2025","dates": ["2025-06-13","2025-06-14","2025-06-15","2025-06-16"], "vline": "2025-06-13", "ha": "left"},
]

COLOR_DA = "#c0392b"
COLOR_SF = "#2171b5"

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.linewidth":   0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

fig, ax = plt.subplots(figsize=(11, 4.2))

# Scatter: individual days (light)
ax.scatter(daily["date_dt"], daily["mape_da"],
           color=COLOR_DA, alpha=0.12, s=5, zorder=2)
ax.scatter(daily["date_dt"], daily["mape_sf"],
           color=COLOR_SF, alpha=0.12, s=5, zorder=2)

# Rolling means
ax.plot(daily["date_dt"], daily["roll_da"],
        color=COLOR_DA, lw=2.0, zorder=4, label=f"DA MAPE 30-day rolling mean")
ax.plot(daily["date_dt"], daily["roll_sf"],
        color=COLOR_SF, lw=2.0, zorder=4, label=f"SF MAPE 30-day rolling mean")

# Baselines
ax.axhline(BASELINE_DA, color=COLOR_DA, lw=0.9, ls="--", alpha=0.7,
           label=f"DA baseline {BASELINE_DA:.2f}%")
ax.axhline(BASELINE_SF, color=COLOR_SF, lw=0.9, ls="--", alpha=0.7,
           label=f"SF baseline {BASELINE_SF:.2f}%")

# Security event vertical lines + diamonds
for ev in EVENTS:
    xd_vline = pd.to_datetime(ev["vline"])
    ax.axvline(xd_vline, color="#555555", lw=0.9, ls=":", zorder=3)
    ax.text(xd_vline, 12.5, ev["label"],
            fontsize=8.5, color="#333333", ha=ev["ha"], va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85, linewidth=0.5))

    # Average MAPE across all available (non-bad) event dates
    rows = daily[daily["Date"].isin(ev["dates"])]
    if not rows.empty:
        mean_da = rows["mape_da"].mean()
        mean_sf = rows["mape_sf"].mean()
        # Position diamond at the first available event date
        xd_diamond = pd.to_datetime(rows["Date"].iloc[0])
        ax.scatter([xd_diamond], [mean_da],
                   color=COLOR_DA, s=60, zorder=6, marker="D")
        ax.scatter([xd_diamond], [mean_sf],
                   color=COLOR_SF, s=60, zorder=6, marker="D")

ax.set_xlim(daily["date_dt"].min(), daily["date_dt"].max())
ax.set_ylim(0, 14)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.set_xlabel("Date", fontsize=13, fontweight="bold")
ax.set_ylabel("Daily MAPE (%)", fontsize=13, fontweight="bold")
ax.yaxis.grid(True, color="#e0e0e0", lw=0.5, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

ax.legend(loc="upper right", fontsize=10, framealpha=0.92,
          edgecolor="#cccccc", ncol=2, handlelength=2.0)

plt.tight_layout(pad=0.8)
out = DATA_DIR / "fig_rolling_mape.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
