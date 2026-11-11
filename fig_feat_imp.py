#!/usr/bin/env python3
"""
Feature importance (gain) for LGB-SF-W: 5-min and 60-min horizons.
Clean two-panel horizontal bar chart — no inset.
"""
import os, re, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))
OUT_DIR = Path(os.environ.get("NOGA_FIG_DIR", DATA_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)

from forecast_ml_v2 import (
    load_all_features, get_bad_days, add_derived_train_features,
    fit_lgb, SF_W_FEATS, SF_W1H_FEATS
)

print("Loading data …")
df_raw = load_all_features()
bad    = get_bad_days(df_raw)
df     = df_raw[~df_raw["Date"].isin(bad)].copy()
train  = df[df["year"] == 2024].copy()
train  = add_derived_train_features(train, train)
tr     = train[train["Current Demand (MW)"] > 1000].copy()
y      = tr["Current Demand (MW)"]

print("Training 5-min model …");  lgb_5m, _ = fit_lgb(tr, y, SF_W_FEATS)
print("Training 60-min model …"); lgb_1h, _ = fit_lgb(tr, y, SF_W1H_FEATS)

def get_pct(model, feats):
    fi = pd.Series(model.booster_.feature_importance("gain"), index=feats)
    return (fi / fi.sum() * 100).sort_values(ascending=False)

pct_5m = get_pct(lgb_5m,  SF_W_FEATS)
pct_1h = get_pct(lgb_1h, SF_W1H_FEATS)

NAME_MAP = {
    "lag_5min":        r"$d_{t-5\mathrm{min}}$",
    "lag_15min":       r"$d_{t-15\mathrm{min}}$",
    "lag_30min":       r"$d_{t-30\mathrm{min}}$",
    "lag_1h":          r"$d_{t-1\mathrm{h}}$",
    "lag_2h":          r"$d_{t-2\mathrm{h}}$",
    "lag_6h":          r"$d_{t-6\mathrm{h}}$",
    "lag_24h":         r"$d_{t-24\mathrm{h}}$",
    "lag_48h":         r"$d_{t-48\mathrm{h}}$",
    "lag_ren_1h":      r"$r_{t-1\mathrm{h}}$",
    "lag_ren_6h":      r"$r_{t-6\mathrm{h}}$",
    "lag_5min_solar":  r"$S_{t-5\mathrm{min}}$",
    "lag_1h_solar":    r"$S_{t-1\mathrm{h}}$",
    "delta_5min":      r"$\Delta d_{5\mathrm{min}}$",
    "delta_30min":     r"$\Delta d_{30\mathrm{min}}$",
    "solar_delta_5min":r"$\Delta S_{5\mathrm{min}}$",
    "temp_2m":         r"Air temp. $T$",
    "apparent_temp":   "Apparent temp.",
    "shortwave_rad":   "GHI",
    "pv_model":        r"PV model $\hat{S}$",
    "temp_anomaly":    r"Temp.\ anomaly",
    "lag_1h_temp":     r"$T_{t-1\mathrm{h}}$",
    "sin_hour":        r"$\sin(\mathrm{hour})$",
    "cos_hour":        r"$\cos(\mathrm{hour})$",
    "ren_expected":    r"$\bar{r}_{h,m}$",
    "is_offday":       "Off-day flag",
}
def lbl(f):
    return NAME_MAP.get(re.sub(r"_o\d+$", "", f), f)

GROUP_COLOR = {
    "demand": "#2171b5",   # blue
    "delta":  "#41ab5d",   # green
    "solar":  "#f16913",   # orange
    "weather":"#cb181d",   # red
    "other":  "#969696",   # grey
}
def grp_color(feat):
    feat = re.sub(r"_o\d+$", "", feat)
    if feat.startswith("lag_") and all(x not in feat for x in ("solar","ren","temp","ghi")):
        return GROUP_COLOR["demand"]
    if feat.startswith("delta"):
        return GROUP_COLOR["delta"]
    if any(x in feat for x in ("solar","ghi","pv_model")):
        return GROUP_COLOR["solar"]
    if feat in {"temp_2m","apparent_temp","cloud_cover","shortwave_rad",
                "direct_rad","temp_anomaly","lag_1h_temp"}:
        return GROUP_COLOR["weather"]
    return GROUP_COLOR["other"]

TOP_N = 8

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":       10,
    "axes.linewidth":   0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 0,
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                         gridspec_kw={"wspace": 0.55})

panels = [
    (axes[0], pct_5m,  "(a)  5-minute horizon"),
    (axes[1], pct_1h,  "(b)  60-minute horizon"),
]

for ax, pct, title in panels:
    top    = pct.head(TOP_N)
    feats  = list(top.index)
    vals   = top.values
    labels = [lbl(f) for f in feats]
    colors = [grp_color(f) for f in feats]

    # bars — largest at top (index 0 = top of chart)
    y_pos = np.arange(TOP_N)
    bars  = ax.barh(y_pos, vals, color=colors,
                    edgecolor="white", linewidth=0.5, height=0.62)
    ax.invert_yaxis()

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Gain importance (%)", fontsize=9.5)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)

    # remove distracting spines
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(left=False, bottom=True)

    # subtle grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.5)

    # value labels — inside bar if large enough, outside otherwise
    threshold = vals[0] * 0.18
    for bar, v in zip(bars, vals):
        w = bar.get_width()
        if w >= threshold:
            ax.text(w - vals[0] * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}%" if v < 10 else f"{v:.1f}%",
                    ha="right", va="center", fontsize=8, color="white",
                    fontweight="bold")
        else:
            ax.text(w + vals[0] * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}%",
                    ha="left", va="center", fontsize=8, color="#333333")

    ax.set_xlim(0, vals[0] * 1.28)

# shared legend
legend_handles = [
    Patch(facecolor=GROUP_COLOR["demand"],  label="Demand lag"),
    Patch(facecolor=GROUP_COLOR["delta"],   label="Rate-of-change"),
    Patch(facecolor=GROUP_COLOR["solar"],   label="Solar / PV"),
    Patch(facecolor=GROUP_COLOR["weather"], label="Weather"),
    Patch(facecolor=GROUP_COLOR["other"],   label="Calendar / other"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.06))

plt.savefig(OUT_DIR / "fig_feat_imp.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved fig_feat_imp.png")
