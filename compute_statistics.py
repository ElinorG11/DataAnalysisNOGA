#!/usr/bin/env python3
# NOTE (revision): the Mann-Whitney test on the three above-baseline events below was replaced
# in the revised manuscript by the tests in revision_statistics.py (all four events).
"""
Statistical tests for the NOGA paper.
1. Security events: empirical rank p-values + bootstrap CI on DA MAPE
2. Renewable penetration: Spearman ρ (SF MAPE vs. bin centre) + Mann-Kendall
3. Seasonal ramp: Kruskal-Wallis across years for Winter ramp magnitude
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))

from forecast_ml_v2 import (
    load_all_features, get_bad_days, add_derived_train_features,
    mape_arr, mape_mean, HOUR_FROM, HOUR_TO, DA_MIN, DA_MAX, SECURITY_EVENTS
)

print("Loading data …")
df_raw = load_all_features()
bad    = get_bad_days(df_raw)
df     = df_raw[~df_raw["Date"].isin(bad)].copy()

# ── Compute daily DA MAPE for all 802 days ────────────────────────────────────
daytime = df[(df["hour"] >= HOUR_FROM) & (df["hour"] < HOUR_TO)].copy()
act  = daytime["Current Demand (MW)"].values
nda  = daytime["Day-Ahead Demand Forecast (MW)"].where(
       daytime["Day-Ahead Demand Forecast (MW)"].between(DA_MIN, DA_MAX)).values
daytime["mape_da"] = mape_arr(act, nda)

daily_da = (daytime.groupby("Date")["mape_da"]
            .mean()
            .dropna()
            .sort_values(ascending=False))
N = len(daily_da)
print(f"\nTotal clean days for ranking: {N}")

# ── 1. Security event rank p-values ──────────────────────────────────────────
print("\n=== Security Event Rank p-values (empirical, one-tailed) ===")
event_dates = {
    "Hamas (Oct 8, 2023)":   "2023-10-08",
    "True Promise I (Apr 13-14, 2024, avg)": ["2024-04-13","2024-04-14"],
    "True Promise II (Oct 1, 2024)":  "2024-10-01",
    "Rising Lion (avg Jun 13-16, 2025)": ["2025-06-13","2025-06-14","2025-06-15","2025-06-16"],
}

all_da_vals = daily_da.values  # descending

for name, dates in event_dates.items():
    if isinstance(dates, str):
        dates = [dates]
    dates = [d for d in dates if d in daily_da.index]
    if not dates:
        print(f"  {name}: no data")
        continue
    event_mape = daily_da.loc[dates].mean()
    # rank = number of days with MAPE >= event_mape (1-based)
    rank = (all_da_vals >= event_mape).sum()
    p_empirical = rank / N
    print(f"  {name}")
    print(f"    DA MAPE = {event_mape:.2f}%   rank = {rank}/{N}   p = {p_empirical:.4f}")

# Bootstrap CI on mean DA MAPE for event days vs. all days
event_day_mapes = []
for dates in event_dates.values():
    if isinstance(dates, str):
        dates = [dates]
    dates = [d for d in dates if d in daily_da.index]
    event_day_mapes.extend(daily_da.loc[dates].values)

rng = np.random.default_rng(42)
n_boot = 10000
baseline_mapes = daily_da.values

# Bootstrap CI for baseline mean
boot_baseline = np.array([
    rng.choice(baseline_mapes, size=len(baseline_mapes), replace=True).mean()
    for _ in range(n_boot)
])
ci_lo, ci_hi = np.percentile(boot_baseline, [2.5, 97.5])
print(f"\n  Baseline DA MAPE: {baseline_mapes.mean():.2f}%  95% CI [{ci_lo:.2f}%, {ci_hi:.2f}%]")

# Mann-Whitney: above-baseline events (Hamas, TP1, Rising Lion) vs all non-event days
above_events = []
for name, dates in event_dates.items():
    if "II" in name:
        continue
    if isinstance(dates, str): dates = [dates]
    dates = [d for d in dates if d in daily_da.index]
    above_events.extend(daily_da.loc[dates].values)

non_event = daily_da.drop([d for dates in event_dates.values()
                            for d in ([dates] if isinstance(dates,str) else dates)
                            if d in daily_da.index]).values

stat, p_mw = stats.mannwhitneyu(above_events, non_event, alternative='greater')
print(f"\n  Mann-Whitney (3 above-baseline events vs. all non-event days):")
print(f"    U = {stat:.0f},  one-tailed p = {p_mw:.4f}")

# ── 2. Renewable penetration → MAPE Spearman ─────────────────────────────────
print("\n=== Spearman ρ: Renewable Penetration Bins vs. SF MAPE ===")
# Use bin data from the paper (Table tab:err-ren)
bin_centres = np.array([2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5])
sf_mape     = np.array([1.19, 0.91, 1.01, 1.20, 1.24, 1.38, 1.64, 2.32, 2.99, 4.31, 5.50, 7.26])
da_mape     = np.array([2.91, 3.35, 3.61, 3.40, 3.16, 3.02, 3.04, 3.48, 3.92, 4.77, 4.94, 3.86])
counts      = np.array([115862, 14537, 9911, 10496, 13377, 16977, 15697, 10581, 6552, 3294, 718, 14])

# Exclude last bin (n=14, unreliable)
mask = counts >= 100
rho_sf, p_sf = stats.spearmanr(bin_centres[mask], sf_mape[mask])
rho_da, p_da = stats.spearmanr(bin_centres[mask], da_mape[mask])
print(f"  SF MAPE vs. bin centre:  ρ = {rho_sf:.3f},  p = {p_sf:.4f}  (n={mask.sum()} bins)")
print(f"  DA MAPE vs. bin centre:  ρ = {rho_da:.3f},  p = {p_da:.4f}  (n={mask.sum()} bins)")

# Weighted Spearman (weight by count)
from scipy.stats import weightedtau
tau_sf, p_tau_sf = weightedtau(bin_centres[mask], sf_mape[mask],
                                weigher=lambda x: counts[mask][x])
print(f"  Weighted Kendall τ (SF): τ = {tau_sf:.3f}")

# ── 3. Year-over-year renewable penetration trend ────────────────────────────
print("\n=== Year-over-year Renewable Trend (Kruskal-Wallis across months) ===")
df_kw = df.copy()
df_kw["ren_share"] = df_kw["ren_share"]  # already computed in load_all_features

by_year = [df_kw[df_kw["year"]==y]["ren_share"].dropna().values for y in [2023, 2024, 2025]]
stat_kw, p_kw = stats.kruskal(*by_year)
print(f"  Kruskal-Wallis (2023 vs 2024 vs 2025 renewable share):")
print(f"    H = {stat_kw:.1f},  p = {p_kw:.2e}")
stat_23_25, p_23_25 = stats.mannwhitneyu(by_year[0], by_year[2], alternative='less')
print(f"  Mann-Whitney (2023 < 2025):  p = {p_23_25:.2e}")

print("\nDone.")
