#!/usr/bin/env python3
"""
SHAP beeswarm summary for LGB-SF-W: 5-min and 60-min horizons.
Two-panel figure — matches fig_feat_imp.py aesthetic.
"""
import os, re, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap

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

# Use 2025 test set for SHAP (more interesting — higher penetration)
test_2025 = df[df["year"] == 2025].copy()
test_2025 = add_derived_train_features(test_2025, train)
te        = test_2025[test_2025["Current Demand (MW)"] > 1000].copy()

print("Training 5-min model …");  lgb_5m, _ = fit_lgb(tr, y, SF_W_FEATS)
print("Training 60-min model …"); lgb_1h, _ = fit_lgb(tr, y, SF_W1H_FEATS)

# ── SHAP computation ─────────────────────────────────────────────────────────
# Sample to keep compute manageable (~3k points each)
NSAMPLE = 3000
rng = np.random.default_rng(42)

def compute_shap(model, feats, te, n=NSAMPLE):
    ok  = te[feats].notna().all(axis=1)
    sub = te.loc[ok, feats]
    if len(sub) > n:
        idx = rng.choice(len(sub), n, replace=False)
        sub = sub.iloc[idx]
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sub)
    return shap_values, sub

print("Computing SHAP (5-min) …")
sv_5m, X_5m = compute_shap(lgb_5m, SF_W_FEATS, te)
print("Computing SHAP (60-min) …")
sv_1h, X_1h = compute_shap(lgb_1h, SF_W1H_FEATS, te)

# ── Feature display names ─────────────────────────────────────────────────────
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
    "temp_anomaly":    r"Temp. anomaly $\Delta T$",
    "lag_1h_temp":     r"$T_{t-1\mathrm{h}}$",
    "sin_hour":        r"$\sin(\mathrm{hour})$",
    "cos_hour":        r"$\cos(\mathrm{hour})$",
    "ren_expected":    r"$\bar{r}_{h,m}$",
    "is_offday":       "Off-day flag",
}
def lbl(f):
    return NAME_MAP.get(re.sub(r"_o\d+$", "", f), f)

# ── Beeswarm helper ───────────────────────────────────────────────────────────
def beeswarm_panel(ax, shap_vals, X, feats, title, top_n=8):
    """Manual beeswarm: mean |SHAP| ranking, dot scatter coloured by feature value."""
    # rank by mean |SHAP|
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:top_n]
    # reverse so largest is on top
    order    = order[::-1]

    n_feats = len(order)
    cmap    = plt.get_cmap("coolwarm")

    for row_idx, feat_idx in enumerate(order):
        feat_name = feats[feat_idx]
        sv   = shap_vals[:, feat_idx]
        fval = X.iloc[:, feat_idx].values.astype(float)

        # normalise feature values → [0,1] for colour
        fv_min, fv_max = np.nanpercentile(fval, 5), np.nanpercentile(fval, 95)
        if fv_max > fv_min:
            fv_norm = np.clip((fval - fv_min) / (fv_max - fv_min), 0, 1)
        else:
            fv_norm = np.full_like(fval, 0.5)

        # jitter on y-axis proportional to density
        jitter = rng.uniform(-0.38, 0.38, size=len(sv))
        colors = cmap(fv_norm)

        ax.scatter(sv, row_idx + jitter, c=colors, s=4, alpha=0.55,
                   linewidths=0, rasterized=True)

    # zero line
    ax.axvline(0, color="#888888", linewidth=0.7, linestyle="--")

    feat_labels = [lbl(feats[i]) for i in order]
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(feat_labels, fontsize=8.5)
    ax.set_xlabel("SHAP value (MW)", fontsize=9.5)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(left=False, bottom=True)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.5)

# ── Figure ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":       10,
    "axes.linewidth":   0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 0,
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                         gridspec_kw={"wspace": 0.58})

beeswarm_panel(axes[0], sv_5m, X_5m, SF_W_FEATS,
               "(a)  5-minute horizon", top_n=8)
beeswarm_panel(axes[1], sv_1h, X_1h, SF_W1H_FEATS,
               "(b)  60-minute horizon", top_n=8)

# Shared colourbar for feature-value scale
sm = plt.cm.ScalarMappable(cmap="coolwarm",
                            norm=mcolors.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                    fraction=0.025, pad=0.14, aspect=40)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["Low", "Mid", "High"], fontsize=8)
cbar.set_label("Feature value", fontsize=8.5)
cbar.ax.tick_params(size=0)
for spine in cbar.ax.spines.values():
    spine.set_linewidth(0.5)

plt.savefig(OUT_DIR / "fig_shap_summary.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved fig_shap_summary.png")
