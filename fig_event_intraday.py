#!/usr/bin/env python3
"""
Security event intra-day demand profiles.
4-panel figure: actual demand vs DA forecast for each event,
with a normal-day reference band (same weekday group, same season,
from the same year, excluding event days and bad days).
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))

from forecast_ml_v2 import (
    load_all_features, get_bad_days, DA_MIN, DA_MAX
)

print("Loading data …")
df_raw = load_all_features()
bad    = get_bad_days(df_raw)

df = df_raw.copy()
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df["time_of_day"] = df["hour"] + df["minute"] / 60.0   # fractional hour

# Sanitise DA
df["da"] = df["Day-Ahead Demand Forecast (MW)"].where(
    df["Day-Ahead Demand Forecast (MW)"].between(DA_MIN, DA_MAX)
)

# ── Event definitions ────────────────────────────────────────────────────────
EVENTS = [
    {
        "label": "Conflict Outbreak day-after\n(Oct 8, 2023)",
        "dates": ["2023-10-08"],
        "color": "#c0392b",
        "ref_year": 2023,
        "ref_season": "Autumn",
        "ref_offday": False,   # Oct 8 was a Sunday — treat as weekend
    },
    {
        "label": "True Promise I\n(Apr 13–14, 2024)",
        "dates": ["2024-04-13", "2024-04-14"],
        "color": "#8e44ad",
        "ref_year": 2024,
        "ref_season": "Spring",
        "ref_offday": True,    # Apr 13 = Sat, Apr 14 = Sun
    },
    {
        "label": "True Promise II\n(Oct 1, 2024)",
        "dates": ["2024-10-01"],
        "color": "#e67e22",
        "ref_year": 2024,
        "ref_season": "Autumn",
        "ref_offday": False,
    },
    {
        "label": "Rising Lion\n(Jun 13–16, 2025)",
        "dates": ["2025-06-13", "2025-06-14", "2025-06-15", "2025-06-16"],
        "color": "#2471a3",
        "ref_year": 2025,
        "ref_season": "Summer",
        "ref_offday": False,
    },
]

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.linewidth":   0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5),
                         gridspec_kw={"hspace": 0.42, "wspace": 0.28})
axes = axes.flatten()

# ── Bin intervals to integer hour for plotting (mean per hour) ────────────────
def hourly_mean(sub, col):
    return sub.groupby("hour")[col].mean()

# ── Reference band: mean ± 1 std across ref days ──────────────────────────────
def ref_band(df, bad, year, season, offday, exclude_dates):
    mask = (
        (df["year"] == year) &
        (df["season"] == season) &
        (df["is_offday"] == offday) &
        (~df["Date"].isin(bad)) &
        (~df["Date"].isin(exclude_dates))
    )
    sub = df[mask].copy()
    if len(sub) == 0:
        return None, None, None
    grp = sub.groupby("hour")["Current Demand (MW)"]
    return grp.mean(), grp.mean() - grp.std(), grp.mean() + grp.std()

hours = np.arange(0, 24)

for ax, ev in zip(axes, EVENTS):
    ev_dates = ev["dates"]
    color    = ev["color"]

    # Reference band
    ref_mean, ref_lo, ref_hi = ref_band(
        df, bad,
        year=ev["ref_year"],
        season=ev["ref_season"],
        offday=ev["ref_offday"],
        exclude_dates=ev_dates
    )

    if ref_mean is not None:
        ax.fill_between(ref_mean.index, ref_lo.values, ref_hi.values,
                        color="#cccccc", alpha=0.55, zorder=1,
                        label="Normal ±1σ")
        ax.plot(ref_mean.index, ref_mean.values,
                color="#999999", lw=1.2, ls="--", zorder=2,
                label="Normal mean (DA)")

    # DA forecast reference: use the average DA forecast on those same ref days
    ref_da_mask = (
        (df["year"] == ev["ref_year"]) &
        (df["season"] == ev["ref_season"]) &
        (df["is_offday"] == ev["ref_offday"]) &
        (~df["Date"].isin(bad)) &
        (~df["Date"].isin(ev_dates))
    )
    ref_da_mean = df[ref_da_mask].groupby("hour")["da"].mean()
    ax.plot(ref_da_mean.index, ref_da_mean.values,
            color="#555555", lw=1.0, ls=":", zorder=2,
            label="Normal mean (DA fcst)")

    # Individual event days + mean
    day_actuals = []
    day_da      = []
    for d in ev_dates:
        sub = df[df["Date"] == d].copy()
        if sub.empty:
            continue
        act_h = hourly_mean(sub, "Current Demand (MW)")
        da_h  = hourly_mean(sub, "da")
        day_actuals.append(act_h)
        day_da.append(da_h)

        if len(ev_dates) > 1:
            ax.plot(act_h.index, act_h.values,
                    color=color, lw=0.9, alpha=0.35, zorder=4)

    if not day_actuals:
        continue

    # Mean across event days
    mean_act = pd.concat(day_actuals, axis=1).mean(axis=1)
    mean_da  = pd.concat(day_da,  axis=1).mean(axis=1)

    ax.plot(mean_act.index, mean_act.values,
            color=color, lw=2.5, zorder=5,
            label="Actual demand (event)")
    ax.plot(mean_da.index, mean_da.values,
            color=color, lw=1.8, ls="--", zorder=5,
            label="DA forecast (event)")

    ax.set_title(ev["label"], fontsize=12, fontweight="bold", pad=6)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_xticklabels(["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"],
                       fontsize=9.5)
    ax.set_xlabel("Hour of day", fontsize=11, fontweight="bold")
    ax.set_ylabel("Demand (MW)", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, color="#e0e0e0", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.legend(fontsize=9.0, loc="upper left", framealpha=0.90,
              edgecolor="#cccccc", handlelength=1.8)

plt.suptitle("Intra-day Demand vs DA Forecast: Security Events",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out = DATA_DIR / "fig_event_intraday.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
