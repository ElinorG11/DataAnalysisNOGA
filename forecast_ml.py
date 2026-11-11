#!/usr/bin/env python3
"""
NOGA-ISO Demand Forecasting — ML Models vs NOGA Baseline
=========================================================
Train: 2024  |  Test: 2023 and 2025

Models compared:
  SM      – Seasonal Mean  (hour × month × offday, from train data)
  GBM-DA  – Gradient Boosting, DA-style (calendar + expected-ren + lag_24h only)
  GBM-SF  – Gradient Boosting, SF-style (+ intraday demand/ren lags)
  LGB-SF  – LightGBM,  SF-style (same features, often more accurate)

  NOGA-DA – NOGA's production day-ahead forecast  (baseline)
  NOGA-SF – NOGA's production short-term forecast (baseline)

Metric: MAPE = |forecast – actual| / actual × 100 %  (08:00–20:00, daytime)
Features include: hour, day-of-week, month, season, Israeli weekend (Fri-Sat),
                  Israeli public holidays, expected renewable share.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import holidays as hol_lib

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent

# ── Constants ─────────────────────────────────────────────────────────────────
HOUR_FROM, HOUR_TO   = 8, 20
DA_MIN, DA_MAX       = 2000, 25000
FROZEN_THR           = 0.10
IL_WEEKEND_DAYS      = {4, 5}   # Python weekday(): Friday=4, Saturday=5

SEASON_MAP = {12:"Winter",1:"Winter",2:"Winter",
              3:"Spring", 4:"Spring", 5:"Spring",
              6:"Summer", 7:"Summer", 8:"Summer",
              9:"Autumn", 10:"Autumn",11:"Autumn"}
SEASON_ORDER = ["Winter","Spring","Summer","Autumn"]

SECURITY_EVENTS = {
    "Hamas day-after":   ["2023-10-08"],
    "True Promise I":    ["2024-04-13","2024-04-14"],
    "True Promise II":   ["2024-10-01"],
    "Rising Lion":       ["2025-06-13","2025-06-14","2025-06-15","2025-06-16"],
}
ALL_EVENT_DATES = {d for ds in SECURITY_EVENTS.values() for d in ds}

MODEL_COLORS = {
    "NOGA-DA":  "#c0392b",
    "NOGA-SF":  "#e74c3c",
    "SM":       "#7f8c8d",
    "GBM-DA":   "#2980b9",
    "GBM-SF":   "#1abc9c",
    "LGB-SF":   "#27ae60",
}
MODEL_ORDER = ["NOGA-DA","NOGA-SF","SM","GBM-DA","GBM-SF","LGB-SF"]

plt.rcParams.update({
    "font.family":"DejaVu Serif","font.size":10,
    "axes.titlesize":11,"axes.labelsize":10,
    "xtick.labelsize":9,"ytick.labelsize":9,
    "figure.dpi":150,"savefig.dpi":300,
    "savefig.bbox":"tight","axes.grid":True,"grid.alpha":0.3,
})

DIVIDER = "=" * 90

# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_holiday_set(years=(2023, 2024, 2025)):
    """Return set of date-strings that are Israeli public holidays."""
    hols = set()
    for y in years:
        for d in hol_lib.Israel(years=y):
            hols.add(str(d))
    return hols


def load_features():
    """Load demand + production mix, engineer all features, compute lag columns."""
    print("Loading and engineering features …")

    # Demand
    dem = pd.read_csv(DATA_DIR / "noga_demand_5min.csv", encoding="utf-8-sig")
    dem.rename(columns={"Interval (1-288)": "Interval"}, inplace=True)
    dem["datetime"] = pd.to_datetime(dem["Date"] + " " + dem["Time"])
    for c in ["Current Demand (MW)",
              "Updated Demand Forecast (MW)",
              "Day-Ahead Demand Forecast (MW)"]:
        dem[c] = pd.to_numeric(dem[c], errors="coerce")

    # Production mix (renewable only)
    prod = pd.read_csv(DATA_DIR / "noga_production_mix_5min.csv", encoding="utf-8-sig")
    prod.rename(columns={"Interval (1-288)": "Interval"}, inplace=True)
    prod["datetime"] = pd.to_datetime(prod["Date"] + " " + prod["Time"])
    prod["Renewables Total (MW)"] = pd.to_numeric(
        prod.get("Renewables Total (MW)", 0), errors="coerce").fillna(0)
    df = dem.merge(prod[["datetime","Renewables Total (MW)"]], on="datetime", how="left")

    # ── Calendar metadata ──────────────────────────────────────────────────
    df["hour"]        = df["datetime"].dt.hour
    df["minute"]      = df["datetime"].dt.minute
    df["month"]       = df["datetime"].dt.month
    df["year"]        = df["datetime"].dt.year
    df["day_of_week"] = df["datetime"].dt.dayofweek   # Mon=0, Fri=4, Sat=5
    df["season"]      = df["month"].map(SEASON_MAP)
    df["is_weekend"]  = df["day_of_week"].isin(IL_WEEKEND_DAYS).astype(int)
    df["time_min"]    = df["hour"] * 60 + df["minute"]
    df["sin_hour"]    = np.sin(2 * np.pi * df["time_min"] / 1440)
    df["cos_hour"]    = np.cos(2 * np.pi * df["time_min"] / 1440)
    df["sin_month"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]   = np.cos(2 * np.pi * df["month"] / 12)

    # ── Holiday flag ───────────────────────────────────────────────────────
    hols = build_holiday_set()
    df["is_holiday"]  = df["Date"].isin(hols).astype(int)
    df["is_offday"]   = np.maximum(df["is_weekend"], df["is_holiday"])

    # ── Renewable share ────────────────────────────────────────────────────
    df["ren_share"] = np.where(df["Current Demand (MW)"] > 500,
                                df["Renewables Total (MW)"].fillna(0) /
                                df["Current Demand (MW)"] * 100,
                                np.nan)

    # ── Lag features (on full sorted series, no leakage) ─────────────────
    df = df.sort_values("datetime").reset_index(drop=True)
    dem_col = "Current Demand (MW)"
    df["lag_1h"]    = df[dem_col].shift(12)    # 1 h  = 12 × 5 min
    df["lag_2h"]    = df[dem_col].shift(24)
    df["lag_6h"]    = df[dem_col].shift(72)
    df["lag_12h"]   = df[dem_col].shift(144)
    df["lag_24h"]   = df[dem_col].shift(288)   # 24 h
    df["lag_48h"]   = df[dem_col].shift(576)   # 48 h
    df["lag_ren_1h"]= df["ren_share"].shift(12)
    df["lag_ren_6h"]= df["ren_share"].shift(72)

    # Mean demand same weekday last 4 weeks (approximate week-lag)
    df["lag_1w"] = df[dem_col].shift(2016)     # 7 days × 288 intervals

    print(f"  {len(df):,} rows  |  holidays found: {len(hols)}")
    return df


def get_bad_days(df):
    bad = set()
    day = df[(df["hour"] >= HOUR_FROM) & (df["hour"] < HOUR_TO)]
    for date, g in day.groupby("Date"):
        vals = g["Current Demand (MW)"].values
        frozen = sum(1 for i in range(1, len(vals))
                     if vals[i] == vals[i-1] and vals[i] > 0)
        if len(vals) > 0 and frozen / len(vals) > FROZEN_THR:
            bad.add(date)
    return bad


def add_expected_renewable(df, train_df):
    """
    Append ren_expected = mean renewable share for (hour, month) from training data.
    This simulates what a DA forecaster knows: the typical solar profile.
    """
    ren_avg = (train_df[train_df["ren_share"].notna()]
               .groupby(["hour","month"])["ren_share"]
               .mean()
               .reset_index()
               .rename(columns={"ren_share": "ren_expected"}))
    if "ren_expected" in df.columns:
        df = df.drop(columns=["ren_expected"])
    return df.merge(ren_avg, on=["hour","month"], how="left")


# ── Feature sets ──────────────────────────────────────────────────────────────
# DA model: only calendar + expected renewable + same-time-yesterday demand
DA_FEATS = ["hour","day_of_week","month","is_weekend","is_holiday","is_offday",
            "sin_hour","cos_hour","sin_month","cos_month",
            "ren_expected","lag_24h","lag_48h"]

# SF model: + intraday demand lags (simulates updated same-day observations)
SF_FEATS = DA_FEATS + ["lag_1h","lag_2h","lag_6h","lag_12h",
                        "lag_ren_1h","lag_ren_6h"]

# ── MAPE helpers ─────────────────────────────────────────────────────────────
def mape_arr(actual, forecast):
    """Element-wise MAPE percent; returns array (nan where invalid)."""
    mask = (actual > 1000) & pd.notna(forecast) & pd.notna(actual)
    out = np.full(len(actual), np.nan)
    out[mask] = np.abs(forecast[mask] - actual[mask]) / actual[mask] * 100
    return out


def mape_mean(arr):
    v = arr[~np.isnan(arr)]
    return float(v.mean()) if len(v) else float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SeasonalMean:
    """Mean demand per (hour, month, is_offday) bucket — trained on 2024."""
    def fit(self, df):
        self._lut = (df[df["Current Demand (MW)"] > 1000]
                     .groupby(["hour","month","is_offday"])["Current Demand (MW)"]
                     .mean())
    def predict(self, df):
        def lookup(row):
            try:
                return self._lut.loc[(row["hour"], row["month"], row["is_offday"])]
            except KeyError:
                return np.nan
        return np.array(df.apply(lookup, axis=1), dtype=float)


def fit_gbm(X_tr, y_tr):
    m = GradientBoostingRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.85, min_samples_leaf=20, random_state=42)
    m.fit(X_tr, y_tr)
    return m


def fit_lgb(X_tr, y_tr):
    m = lgb.LGBMRegressor(
        n_estimators=500, max_depth=7, num_leaves=63,
        learning_rate=0.04, subsample=0.85, colsample_bytree=0.9,
        min_child_samples=20, random_state=42, verbose=-1)
    m.fit(X_tr, y_tr)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    # ── Load & filter ─────────────────────────────────────────────────────
    df_raw  = load_features()
    bad     = get_bad_days(df_raw)
    df_full = df_raw[~df_raw["Date"].isin(bad)].copy()
    print(f"  Clean days: {df_full['Date'].nunique()}  (bad removed: {len(bad)})")

    train_all = df_full[df_full["year"] == 2024].copy()
    test_2023 = df_full[df_full["year"] == 2023].copy()
    test_2025 = df_full[df_full["year"] == 2025].copy()

    # ── Add expected renewable (from training data only) ─────────────────
    train_all = add_expected_renewable(train_all, train_all)
    test_2023 = add_expected_renewable(test_2023, train_all)
    test_2025 = add_expected_renewable(test_2025, train_all)

    # ── Restrict to daytime window ────────────────────────────────────────
    def daytime(d): return d[(d["hour"] >= HOUR_FROM) & (d["hour"] < HOUR_TO) &
                              (d["Current Demand (MW)"] > 1000)].copy()
    tr  = daytime(train_all)
    t23 = daytime(test_2023)
    t25 = daytime(test_2025)
    print(f"\n  Train (2024): {len(tr):,} intervals  |  "
          f"Test 2023: {len(t23):,}  |  Test 2025: {len(t25):,}")

    # ── Fit models ────────────────────────────────────────────────────────
    print("\nFitting models …")

    # Seasonal Mean
    sm = SeasonalMean(); sm.fit(tr)
    print("  ✓ Seasonal Mean")

    # GBM-DA
    da_mask = tr[DA_FEATS].notna().all(axis=1)
    gbm_da  = fit_gbm(tr.loc[da_mask, DA_FEATS], tr.loc[da_mask, "Current Demand (MW)"])
    print(f"  ✓ GBM-DA  (trained on {da_mask.sum():,} intervals)")

    # GBM-SF
    sf_mask = tr[SF_FEATS].notna().all(axis=1)
    gbm_sf  = fit_gbm(tr.loc[sf_mask, SF_FEATS], tr.loc[sf_mask, "Current Demand (MW)"])
    print(f"  ✓ GBM-SF  (trained on {sf_mask.sum():,} intervals)")

    # LGB-SF
    lgb_sf  = fit_lgb(tr.loc[sf_mask, SF_FEATS], tr.loc[sf_mask, "Current Demand (MW)"])
    print("  ✓ LGB-SF")

    # ── Predict & score ──────────────────────────────────────────────────
    all_results = {}

    for yr, test in [("2023", t23), ("2025", t25)]:
        test = test.copy()
        actual = test["Current Demand (MW)"].values

        # NOGA forecasts
        noga_da = test["Day-Ahead Demand Forecast (MW)"].where(
            test["Day-Ahead Demand Forecast (MW)"].between(DA_MIN, DA_MAX)).values
        noga_sf = test["Updated Demand Forecast (MW)"].where(
            test["Updated Demand Forecast (MW)"].between(DA_MIN, DA_MAX)).values

        # Our models
        sm_pred = sm.predict(test)

        da_ok = test[DA_FEATS].notna().all(axis=1)
        gbm_da_pred = np.full(len(test), np.nan)
        gbm_da_pred[da_ok] = gbm_da.predict(test.loc[da_ok, DA_FEATS])

        sf_ok = test[SF_FEATS].notna().all(axis=1)
        gbm_sf_pred = np.full(len(test), np.nan)
        lgb_sf_pred = np.full(len(test), np.nan)
        gbm_sf_pred[sf_ok] = gbm_sf.predict(test.loc[sf_ok, SF_FEATS])
        lgb_sf_pred[sf_ok] = lgb_sf.predict(test.loc[sf_ok, SF_FEATS])

        # Build per-interval MAPE arrays
        test["mape_noga_da"] = mape_arr(actual, noga_da)
        test["mape_noga_sf"] = mape_arr(actual, noga_sf)
        test["mape_sm"]      = mape_arr(actual, sm_pred)
        test["mape_gbm_da"]  = mape_arr(actual, gbm_da_pred)
        test["mape_gbm_sf"]  = mape_arr(actual, gbm_sf_pred)
        test["mape_lgb_sf"]  = mape_arr(actual, lgb_sf_pred)

        all_results[yr] = test

    # ═══════════════════════════════════════════════════════════════════════
    # PRINT TABLES
    # ═══════════════════════════════════════════════════════════════════════

    MAPE_COLS = {
        "NOGA-DA": "mape_noga_da", "NOGA-SF": "mape_noga_sf",
        "SM":      "mape_sm",      "GBM-DA":  "mape_gbm_da",
        "GBM-SF":  "mape_gbm_sf",  "LGB-SF":  "mape_lgb_sf",
    }

    # ── TABLE 1: Overall MAPE ────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("TABLE 1 — Overall MAPE (%) — daytime 08:00–20:00")
    print(DIVIDER)
    hdr = f"{'Model':<12} {'2023':>9} {'2025':>9} {'2023+2025':>12}"
    print(hdr); print("─"*len(hdr))
    combined_table = {}
    for mname, col in MAPE_COLS.items():
        v23 = mape_mean(all_results["2023"][col].values)
        v25 = mape_mean(all_results["2025"][col].values)
        both = mape_mean(np.concatenate([all_results["2023"][col].values,
                                          all_results["2025"][col].values]))
        combined_table[mname] = (v23, v25, both)
        print(f"{mname:<12} {v23:>9.2f} {v25:>9.2f} {both:>12.2f}")

    # ── TABLE 2: MAPE by Season ──────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("TABLE 2 — MAPE (%) by Season")
    print(DIVIDER)
    for yr in ["2023","2025"]:
        print(f"\n  Year {yr}:")
        test = all_results[yr]
        hdr2 = f"  {'Model':<12}" + "".join(f"  {s:>8}" for s in SEASON_ORDER)
        print(hdr2); print("  " + "─"*(len(hdr2)-2))
        for mname, col in MAPE_COLS.items():
            row = f"  {mname:<12}"
            for sea in SEASON_ORDER:
                sub = test.loc[test["season"] == sea, col].values
                row += f"  {mape_mean(sub):>8.2f}"
            print(row)

    # ── TABLE 3: MAPE by Hour ────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("TABLE 3 — MAPE (%) by Hour of Day (both years combined)")
    print(DIVIDER)
    test_all = pd.concat([all_results["2023"], all_results["2025"]])
    hdr3 = f"{'Hour':<6}" + "".join(f"  {m:>8}" for m in MODEL_ORDER)
    print(hdr3); print("─"*len(hdr3))
    for h in range(HOUR_FROM, HOUR_TO, 2):
        sub = test_all[test_all["hour"].isin([h, h+1])]
        row = f"{h:02d}:00"
        for mname in MODEL_ORDER:
            col = MAPE_COLS[mname]
            row += f"  {mape_mean(sub[col].values):>8.2f}"
        print(row)

    # ── TABLE 4: Security Events ─────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("TABLE 4 — MAPE (%) on Security Event Days vs Baseline")
    print(DIVIDER)
    hdr4 = f"{'Event':<22}" + "".join(f"  {m:>8}" for m in MODEL_ORDER)
    print(hdr4); print("─"*len(hdr4))

    for ev_name, dates in SECURITY_EVENTS.items():
        ev_yr = dates[0][:4]
        if ev_yr not in all_results: continue
        ev_rows = all_results[ev_yr][all_results[ev_yr]["Date"].isin(dates)]
        if ev_rows.empty: continue
        row = f"{ev_name:<22}"
        for mname in MODEL_ORDER:
            row += f"  {mape_mean(ev_rows[MAPE_COLS[mname]].values):>8.2f}"
        print(row)

    # Baseline row
    row = f"{'Baseline (all days)':<22}"
    for mname in MODEL_ORDER:
        col = MAPE_COLS[mname]
        v = mape_mean(test_all[col].values)
        row += f"  {v:>8.2f}"
    print(row)

    # ── TABLE 5: Feature importance ──────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("TABLE 5 — LGB-SF Feature Importance (gain)")
    print(DIVIDER)
    fi = pd.Series(lgb_sf.booster_.feature_importance(importance_type="gain"),
                   index=SF_FEATS).sort_values(ascending=False)
    total = fi.sum()
    print(f"{'Feature':<16} {'Importance':>12} {'Share%':>8}")
    print("─"*40)
    for feat, val in fi.items():
        print(f"{feat:<16} {val:>12.0f} {val/total*100:>7.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════════════

    # ── Fig 1: Overall MAPE comparison ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, yr in zip(axes, ["2023","2025"]):
        vals  = [combined_table[m][0 if yr=="2023" else 1] for m in MODEL_ORDER]
        cols_ = [MODEL_COLORS[m] for m in MODEL_ORDER]
        bars  = ax.bar(MODEL_ORDER, vals, color=cols_, edgecolor="black", lw=0.6, zorder=3)
        ax.axhline(combined_table["NOGA-DA"][0 if yr=="2023" else 1],
                   color="#c0392b", ls=":", lw=1.2, alpha=0.7, label="NOGA-DA ref")
        ax.axhline(combined_table["NOGA-SF"][0 if yr=="2023" else 1],
                   color="#e74c3c", ls="--", lw=1.0, alpha=0.7, label="NOGA-SF ref")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.05,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"Overall MAPE — Test Year {yr}")
        ax.set_ylabel("MAPE (%)"); ax.legend(fontsize=8)
        ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, max(vals)*1.25)

    fig.suptitle("ML Models vs NOGA Baseline — Overall MAPE Comparison\n"
                 "(Train: 2024 | Test: 2023 and 2025 | 08:00–20:00)", fontsize=11)
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig10_ml_overall.png")
    plt.close(fig)
    print("\n  → Saved fig10_ml_overall.png")

    # ── Fig 2: MAPE by Season ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, yr in zip(axes, ["2023","2025"]):
        test = all_results[yr]
        x_ = np.arange(len(SEASON_ORDER))
        w  = 0.13
        for i, mname in enumerate(MODEL_ORDER):
            col = MAPE_COLS[mname]
            sea_vals = [mape_mean(test.loc[test["season"]==s, col].values)
                        for s in SEASON_ORDER]
            off = (i - len(MODEL_ORDER)/2 + 0.5) * w
            ax.bar(x_+off, sea_vals, w, color=MODEL_COLORS[mname],
                   label=mname, edgecolor="black", lw=0.4)
        ax.set_xticks(x_); ax.set_xticklabels(SEASON_ORDER, fontsize=9)
        ax.set_ylabel("MAPE (%)"); ax.set_title(f"MAPE by Season — Test {yr}")
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig11_ml_seasonal.png")
    plt.close(fig)
    print("  → Saved fig11_ml_seasonal.png")

    # ── Fig 3: MAPE by Hour ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    test_all = pd.concat([all_results["2023"], all_results["2025"]])
    hrs = list(range(HOUR_FROM, HOUR_TO))
    for mname in MODEL_ORDER:
        col = MAPE_COLS[mname]
        h_vals = [mape_mean(test_all.loc[test_all["hour"]==h, col].values) for h in hrs]
        ls = "--" if "NOGA" in mname else "-"
        lw = 1.5 if "LGB" in mname else 1.2
        ax.plot(hrs, h_vals, color=MODEL_COLORS[mname], lw=lw, ls=ls,
                marker="o", ms=4, label=mname)
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE by Hour of Day — Both Test Years Combined")
    ax.set_xticks(hrs); ax.set_xticklabels([f"{h:02d}:00" for h in hrs], rotation=45, fontsize=8)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig12_ml_hourly.png")
    plt.close(fig)
    print("  → Saved fig12_ml_hourly.png")

    # ── Fig 4: Security Events ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    ev_names_plot = [n for n,ds in SECURITY_EVENTS.items()
                     if ds[0][:4] in all_results]
    ev_names_plot.append("Baseline")

    x_ = np.arange(len(ev_names_plot))
    w  = 0.13
    for i, mname in enumerate(MODEL_ORDER):
        col = MAPE_COLS[mname]
        vals_ = []
        for ev_name in ev_names_plot:
            if ev_name == "Baseline":
                v = mape_mean(test_all[col].values)
            else:
                dates = SECURITY_EVENTS[ev_name]
                yr    = dates[0][:4]
                sub   = all_results[yr][all_results[yr]["Date"].isin(dates)]
                v     = mape_mean(sub[col].values)
            vals_.append(v)
        off = (i - len(MODEL_ORDER)/2 + 0.5) * w
        ax.bar(x_+off, vals_, w, color=MODEL_COLORS[mname],
               label=mname, edgecolor="black", lw=0.4)

    ax.set_xticks(x_)
    ax.set_xticklabels(ev_names_plot, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("MAPE (%)"); ax.set_title("MAPE on Security Event Days — All Models")
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig13_ml_events.png")
    plt.close(fig)
    print("  → Saved fig13_ml_events.png")

    # ── Fig 5: Daily MAPE time series (2025, highlight events) ──────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    for ax, yr in zip(axes, ["2023","2025"]):
        test = all_results[yr]
        daily = (test.groupby("Date")[list(MAPE_COLS.values())].mean()
                 .rename(columns={v:k for k,v in MAPE_COLS.items()})
                 .reset_index())
        daily["Date"] = pd.to_datetime(daily["Date"])
        for mname in ["NOGA-DA","GBM-DA","LGB-SF","NOGA-SF"]:
            ls = "--" if "NOGA" in mname else "-"
            ax.plot(daily["Date"], daily[mname],
                    color=MODEL_COLORS[mname], lw=0.9, ls=ls,
                    alpha=0.85, label=mname)
        # Shade security events
        for ev_name, dates in SECURITY_EVENTS.items():
            if dates[0][:4] == yr:
                for d in dates:
                    ax.axvline(pd.Timestamp(d), color="red", lw=1.5, alpha=0.6)
                ax.text(pd.Timestamp(dates[0]), ax.get_ylim()[1]*0.85 if ax.get_ylim()[1]>0 else 8,
                        ev_name[:10], fontsize=7, color="red", rotation=90)
        ax.set_ylabel("Daily MAPE (%)")
        ax.set_title(f"Daily MAPE — {yr}")
        ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig14_ml_timeseries.png")
    plt.close(fig)
    print("  → Saved fig14_ml_timeseries.png")

    # ── Fig 6: Feature importance ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fi_plot = fi.head(12)
    cols_  = ["#27ae60"] * len(fi_plot)
    ax.barh(fi_plot.index[::-1], fi_plot.values[::-1],
            color=cols_, edgecolor="black", lw=0.5)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("LGB-SF Feature Importance (Top 12)")
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig15_ml_feature_importance.png")
    plt.close(fig)
    print("  → Saved fig15_ml_feature_importance.png")

    print(f"\n{DIVIDER}")
    print("ALL FORECASTING ANALYSES COMPLETE")
    print(f"  Figures: fig10–fig15 saved in {DATA_DIR}")
    print(DIVIDER)

    return all_results, combined_table


if __name__ == "__main__":
    run()
