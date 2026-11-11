#!/usr/bin/env python3
"""Compute persistence baseline MAPE for 5, 15, 30, 60-min horizons."""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))

from forecast_ml_v2 import load_all_features, get_bad_days, add_derived_train_features, mape_arr, mape_mean

df_raw = load_all_features()
bad    = get_bad_days(df_raw)
df     = df_raw[~df_raw["Date"].isin(bad)].copy()

train_all = df[df["year"] == 2024].copy()
test_2023 = df[df["year"] == 2023].copy()
test_2025 = df[df["year"] == 2025].copy()

train_all = add_derived_train_features(train_all, train_all)
test_2023 = add_derived_train_features(test_2023, train_all)
test_2025 = add_derived_train_features(test_2025, train_all)

def allday(d):
    return d[d["Current Demand (MW)"] > 1000].copy()

t23 = allday(test_2023)
t25 = allday(test_2025)

# lag column names for each horizon
LAG_COLS = {
    "5min":  "lag_5min",
    "15min": "lag_15min",
    "30min": "lag_30min",
    "60min": "lag_1h",
}

print("\nPersistence baseline MAPE (all 24 hours, demand > 1000 MW):")
print(f"{'Horizon':<8} {'2023':>8} {'2025':>8} {'Combined':>10}")
print("-" * 38)

for h_name, col in LAG_COLS.items():
    results = []
    all_mape = []
    for yr, test in [("2023", t23), ("2025", t25)]:
        act  = test["Current Demand (MW)"].values
        pred = test[col].values
        m    = mape_arr(act, pred)
        results.append(mape_mean(m))
        all_mape.extend(m[~np.isnan(m)])
    comb = float(np.mean(all_mape))
    print(f"{h_name:<8} {results[0]:>8.2f} {results[1]:>8.2f} {comb:>10.2f}")
