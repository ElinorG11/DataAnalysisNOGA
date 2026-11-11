#!/usr/bin/env python3
"""
Enhanced NOGA-ISO Demand Forecasting — v2
==========================================
Improvements over v1:
  1. Temperature + irradiance from open-meteo historical archive (Tel Aviv)
  2. Physics-informed PV model:  PV_pred = max(0, GHI × k × (1 − γ × (T−25)))
  3. Short lags: 5-min, 15-min, 30-min for SF model
  4. Temperature anomaly feature: actual_T − typical_T(hour, month)
  5. Delta features: MW/min rate-of-change

Models compared:
  v1 (re-run for fair baseline)
    GBM-DA   – calendar + expected-ren + lag_24h
    LGB-SF   – calendar + 1h/6h/24h lags

  v2 new
    LGB-DA-W  – DA  with weather forecast (T, GHI, pv_model, temp_anomaly)
    LGB-SF-S  – SF  with short lags only (5 / 15 / 30 min) — no weather
    LGB-SF-W  – SF  with short lags + weather

  NOGA-DA, NOGA-SF — production baseline

Train: 2024  |  Test: 2023 and 2025
Metric: daytime MAPE (08:00–20:00)

NOTE: weather is the open-meteo historical archive (ECMWF IFS + ERA5/ERA5-Land
reanalysis), not real-time station observations. The short-term models use
weather as known at the forecast origin t - h only; the day-ahead model uses
weather at the target time, i.e. a perfect-forecast proxy. The evaluation is
retrospective: reanalysis is published with a delay of several days.
"""

import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import holidays as hol_lib

warnings.filterwarnings("ignore")

DATA_DIR    = Path(__file__).parent
WEATHER_CSV = DATA_DIR / "weather_tel_aviv_hourly.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
HOUR_FROM, HOUR_TO   = 8, 20
DA_MIN, DA_MAX       = 2000, 25000
FROZEN_THR           = 0.10
IL_WEEKEND_DAYS      = {4, 5}          # Friday=4, Saturday=5

SEASON_MAP = {12:"Winter",1:"Winter",2:"Winter",
              3:"Spring", 4:"Spring", 5:"Spring",
              6:"Summer", 7:"Summer", 8:"Summer",
              9:"Autumn", 10:"Autumn",11:"Autumn"}
SEASON_ORDER = ["Winter","Spring","Summer","Autumn"]

SECURITY_EVENTS = {
    "Hamas (2023)":      ["2023-10-08"],
    "True Promise I":    ["2024-04-13","2024-04-14"],
    "True Promise II":   ["2024-10-01"],
    "Rising Lion (2025)":["2025-06-13","2025-06-14","2025-06-15","2025-06-16"],
}

# PV temperature derating coefficient (standard crystalline silicon)
PV_GAMMA = 0.004   # fraction/°C above 25 °C

plt.rcParams.update({
    "font.family":"DejaVu Serif","font.size":10,"axes.titlesize":11,
    "axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
    "axes.grid":True,"grid.alpha":0.3,
})

MODEL_COLORS = {
    "NOGA-DA": "#c0392b", "NOGA-SF": "#e74c3c",
    "GBM-DA":  "#7f8c8d", "LGB-SF":  "#95a5a6",
    "LGB-DA-W":"#2980b9", "LGB-SF-W-1h":"#8e44ad",
    "LGB-SF-S":"#1abc9c", "LGB-SF-W-5m":"#27ae60",
    "LGB-SF-W-15m":"#16a085", "LGB-SF-W-30m":"#8e44ad",
}
MODEL_ORDER = ["NOGA-DA","GBM-DA","LGB-DA-W","NOGA-SF","LGB-SF",
               "LGB-SF-W-1h","LGB-SF-W-30m","LGB-SF-W-15m","LGB-SF-S","LGB-SF-W-5m"]
HORIZON_ORDER = ["LGB-SF-W-1h","LGB-SF-W-30m","LGB-SF-W-15m","LGB-SF-W-5m"]

DIVIDER = "=" * 90

# ═══════════════════════════════════════════════════════════════════════════════
# 1 — WEATHER DATA
# ═══════════════════════════════════════════════════════════════════════════════

WX_COLS = ["temp_2m","apparent_temp","cloud_cover","shortwave_rad","direct_rad"]

# The cached weather CSV was requested on 2025-06-12 with timezone=Asia/Jerusalem.
# Open-Meteo returns ISO timestamps with the UTC offset valid *at request time*
# and ignores DST changes inside the range, so the whole file is on fixed UTC+3
# (one hour ahead of Israel local time in winter). Verified against NOGA solar
# output: after conversion, GHI aligns with solar in every season.
WEATHER_CSV_UTC_OFFSET_H = 3

# Horizons (minutes) for which weather "as of the forecast origin" is built
WX_ORIGIN_SHIFTS = (5, 10, 15, 30, 60)


def fetch_weather(force=False):
    """
    Hourly weather for Tel Aviv (32.08N, 34.78E) from the open-meteo Historical
    Weather API (default 'best match': ECMWF IFS + ERA5/ERA5-Land reanalysis).
    Returns a frame indexed by UTC time.
    """
    if WEATHER_CSV.exists() and not force:
        print(f"  Weather CSV exists → loading {WEATHER_CSV.name}")
        w = pd.read_csv(WEATHER_CSV)
        if "datetime_utc" in w.columns:
            w["utc"] = pd.to_datetime(w["datetime_utc"])
        else:
            w["utc"] = pd.to_datetime(w["datetime"]) - pd.Timedelta(hours=WEATHER_CSV_UTC_OFFSET_H)
        return w.set_index("utc")[WX_COLS].sort_index()

    print("  Fetching weather from open-meteo archive API …")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 32.08, "longitude": 34.78,
        "start_date": "2023-04-01", "end_date": "2025-12-31",
        "hourly": ",".join(["temperature_2m","apparent_temperature",
                             "cloud_cover","shortwave_radiation","direct_radiation"]),
        "timezone": "GMT",   # request UTC; converted to local time (with DST) below
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "datetime_utc":   pd.to_datetime(h["time"]),
        "temp_2m":        h["temperature_2m"],
        "apparent_temp":  h["apparent_temperature"],
        "cloud_cover":    h["cloud_cover"],
        "shortwave_rad":  h["shortwave_radiation"],
        "direct_rad":     h["direct_radiation"],
    })
    df.to_csv(WEATHER_CSV, index=False)
    print(f"  Saved {len(df):,} rows → {WEATHER_CSV.name}")
    return df.set_index("datetime_utc")[WX_COLS].sort_index()


def weather_at_shifts(weather_utc, shifts_min=(0,) + WX_ORIGIN_SHIFTS):
    """
    5-min weather in Israel local time (with DST), as last known at t - s.

    Hourly values are only carried forward (no interpolation): an hourly value
    is used from its timestamp onwards, never before. Radiation is the mean of
    the preceding hour, so it is known at its timestamp. Shifts are applied on
    a continuous UTC grid, so they are real time offsets even across Saturday
    gaps in the NOGA data and DST changes.
    Returns {s: frame indexed by naive local time}.
    """
    grid = pd.date_range(weather_utc.index.min(),
                         weather_utc.index.max() + pd.Timedelta(minutes=55), freq="5min")
    w5 = weather_utc.reindex(grid).ffill()
    out = {}
    for s in shifts_min:
        ws = w5.shift(freq=f"{s}min") if s else w5
        local = ws.index.tz_localize("UTC").tz_convert("Asia/Jerusalem").tz_localize(None)
        ws = ws.set_axis(local)
        out[s] = ws[~ws.index.duplicated(keep="first")]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — LOAD AND ENGINEER FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def build_holiday_set(years=(2023,2024,2025)):
    return {str(d) for y in years for d in hol_lib.Israel(years=y)}


def load_all_features():
    print("Loading demand + production mix …")
    dem = pd.read_csv(DATA_DIR/"noga_demand_5min.csv", encoding="utf-8-sig")
    dem.rename(columns={"Interval (1-288)":"Interval"}, inplace=True)
    dem["datetime"] = pd.to_datetime(dem["Date"]+" "+dem["Time"])
    for c in ["Current Demand (MW)","Updated Demand Forecast (MW)",
              "Day-Ahead Demand Forecast (MW)"]:
        dem[c] = pd.to_numeric(dem[c], errors="coerce")

    prod = pd.read_csv(DATA_DIR/"noga_production_mix_5min.csv", encoding="utf-8-sig")
    prod.rename(columns={"Interval (1-288)":"Interval"}, inplace=True)
    prod["datetime"] = pd.to_datetime(prod["Date"]+" "+prod["Time"])
    for c in ["Renewables Total (MW)","PV (MW)","Photovoltaic (MW)","Thermo-Solar (MW)"]:
        if c in prod.columns:
            prod[c] = pd.to_numeric(prod[c], errors="coerce").fillna(0)
    prod["Solar (MW)"] = (prod.get("PV (MW)",0) + prod.get("Photovoltaic (MW)",0)
                          + prod.get("Thermo-Solar (MW)",0))

    df = dem.merge(prod[["datetime","Renewables Total (MW)","Solar (MW)"]].rename(
        columns={"Renewables Total (MW)":"Renewables (MW)"}), on="datetime", how="left")

    # ── Calendar ──────────────────────────────────────────────────────────
    df["hour"]        = df["datetime"].dt.hour
    df["minute"]      = df["datetime"].dt.minute
    df["month"]       = df["datetime"].dt.month
    df["year"]        = df["datetime"].dt.year
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["season"]      = df["month"].map(SEASON_MAP)
    df["is_weekend"]  = df["day_of_week"].isin(IL_WEEKEND_DAYS).astype(int)
    df["time_min"]    = df["hour"]*60 + df["minute"]
    df["sin_hour"]    = np.sin(2*np.pi*df["time_min"]/1440)
    df["cos_hour"]    = np.cos(2*np.pi*df["time_min"]/1440)
    df["sin_month"]   = np.sin(2*np.pi*df["month"]/12)
    df["cos_month"]   = np.cos(2*np.pi*df["month"]/12)
    hols = build_holiday_set()
    df["is_holiday"]  = df["Date"].isin(hols).astype(int)
    df["is_offday"]   = np.maximum(df["is_weekend"], df["is_holiday"])

    # ── Renewable share ───────────────────────────────────────────────────
    df["ren_share"] = np.where(df["Current Demand (MW)"]>500,
                                df["Renewables (MW)"].fillna(0)/
                                df["Current Demand (MW)"]*100, np.nan)

    # ── Sort → lag features (entire series, no leakage) ──────────────────
    df = df.sort_values("datetime").reset_index(drop=True)
    D = "Current Demand (MW)"
    # Long lags (v1)
    df["lag_24h"]    = df[D].shift(288)
    df["lag_48h"]    = df[D].shift(576)
    df["lag_6h"]     = df[D].shift(72)
    df["lag_2h"]     = df[D].shift(24)
    df["lag_1h"]     = df[D].shift(12)
    df["lag_ren_1h"] = df["ren_share"].shift(12)
    df["lag_ren_6h"] = df["ren_share"].shift(72)
    # Short lags (v2)
    df["lag_30min"]  = df[D].shift(6)
    df["lag_15min"]  = df[D].shift(3)
    df["lag_5min"]   = df[D].shift(1)
    # Solar lags
    S = "Solar (MW)"
    df["lag_5min_solar"]  = df[S].shift(1)
    df["lag_15min_solar"] = df[S].shift(3)
    df["lag_30min_solar"] = df[S].shift(6)
    df["lag_1h_solar"]    = df[S].shift(12)
    # Rate-of-change features, measured at the forecast origin (t-5min) so that
    # no target-time observation enters a feature (d_t must not appear).
    df["delta_5min"]   = df[D].shift(1) - df[D].shift(2)
    df["delta_30min"]  = df[D].shift(1) - df[D].shift(7)
    df["solar_delta_5min"] = df[S].shift(1) - df[S].shift(2)

    # ── Weather features ───────────────────────────────────────────────────
    print("Loading / downloading weather …")
    weather_h = fetch_weather()
    wx = weather_at_shifts(weather_h)
    # Weather at the target time t (used only by the day-ahead model and the
    # PV-model fit figure) ...
    for c in WX_COLS:
        df[c] = wx[0][c].reindex(df["datetime"]).values
    # ... and as known at the forecast origin t - h, for the short-term models
    for s in WX_ORIGIN_SHIFTS:
        for c in WX_COLS:
            df[f"{c}_o{s}"] = wx[s][c].reindex(df["datetime"]).values

    # Lagged weather for SF (real time offsets, from the continuous grid)
    df["lag_1h_temp"]    = df["temp_2m_o60"]
    df["lag_5min_ghi"]   = df["shortwave_rad_o5"]
    df["lag_15min_ghi"]  = df["shortwave_rad_o15"]
    df["solar_ghi_delta"]= df["shortwave_rad_o5"] - df["shortwave_rad_o10"]

    print(f"  Dataset: {len(df):,} rows  |  date range: "
          f"{df['Date'].min()} → {df['Date'].max()}")
    return df


def get_bad_days(df):
    bad = set()
    day = df[(df["hour"]>=HOUR_FROM) & (df["hour"]<HOUR_TO)]
    for date, g in day.groupby("Date"):
        vals = g["Current Demand (MW)"].values
        frozen = sum(1 for i in range(1,len(vals))
                     if vals[i]==vals[i-1] and vals[i]>0)
        if len(vals)>0 and frozen/len(vals)>FROZEN_THR:
            bad.add(date)
    return bad


def add_derived_train_features(df, train_df):
    """
    Features computed from training data:
      ren_expected : mean renewable share for (hour, month)
      temp_expected: mean temperature for (hour, month)  → for temp_anomaly
      pv_coeff     : calibrated GHI→PV coefficient (MW per W/m²)
    """
    # Expected renewable
    ren_avg = (train_df[train_df["ren_share"].notna()]
               .groupby(["hour","month"])["ren_share"].mean()
               .reset_index().rename(columns={"ren_share":"ren_expected"}))
    if "ren_expected" in df.columns: df = df.drop(columns=["ren_expected"])
    df = df.merge(ren_avg, on=["hour","month"], how="left")

    # Expected temperature & anomaly
    temp_avg = (train_df[train_df["temp_2m"].notna()]
                .groupby(["hour","month"])["temp_2m"].mean()
                .reset_index().rename(columns={"temp_2m":"temp_expected"}))
    if "temp_expected" in df.columns: df = df.drop(columns=["temp_expected"])
    df = df.merge(temp_avg, on=["hour","month"], how="left")
    df["temp_anomaly"] = df["temp_2m"] - df["temp_expected"]

    # PV physics model: calibrate GHI→Solar coefficient from training data
    day_tr = train_df[(train_df["shortwave_rad"]>50) &
                      (train_df["Solar (MW)"]>100)].copy()
    if len(day_tr) > 100:
        # Coefficient per month (accounts for seasonal efficiency variation)
        monthly_k = (day_tr.groupby("month")
                     .apply(lambda g: np.median(g["Solar (MW)"] /
                                                 g["shortwave_rad"].clip(lower=50)))
                     .reset_index().rename(columns={0:"pv_k"}))
        if "pv_k" in df.columns: df = df.drop(columns=["pv_k"])
        df = df.merge(monthly_k, on="month", how="left")
        pv_k_global = float(np.median(day_tr["Solar (MW)"] /
                                       day_tr["shortwave_rad"].clip(lower=50)))
        df["pv_k"] = df["pv_k"].fillna(pv_k_global)
    else:
        df["pv_k"] = 4.5   # default fallback MW per W/m²

    # pv_model = max(0, GHI × k × (1 − γ × max(0, T−25)))
    # Built from weather at t (suffix "") and from weather known at each
    # forecast origin t - h (suffix "_o<h>"); k is a fixed training constant.
    for sfx in [""] + [f"_o{s}" for s in WX_ORIGIN_SHIFTS]:
        ghi    = df[f"shortwave_rad{sfx}"].fillna(0).clip(lower=0)
        t_cor  = (1 - PV_GAMMA * np.maximum(0, df[f"temp_2m{sfx}"].fillna(25) - 25))
        df[f"pv_model{sfx}"] = np.maximum(0, ghi * df["pv_k"] * t_cor)
        if sfx:
            df[f"temp_anomaly{sfx}"] = df[f"temp_2m{sfx}"] - df["temp_expected"]

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — FEATURE SETS
# ═══════════════════════════════════════════════════════════════════════════════

_CAL = ["hour","day_of_week","month","is_weekend","is_holiday","is_offday",
        "sin_hour","cos_hour","sin_month","cos_month","ren_expected"]

_LAG_LONG = ["lag_24h","lag_48h"]
_LAG_MED  = ["lag_1h","lag_2h","lag_6h","lag_ren_1h","lag_ren_6h"]
_LAG_SHORT= ["lag_5min","lag_15min","lag_30min","lag_5min_solar",
              "lag_15min_solar","lag_30min_solar","lag_1h_solar",
              "delta_5min","delta_30min","solar_delta_5min"]
# Weather at the target time t: day-ahead model only (a perfect-forecast proxy)
_WX       = ["temp_2m","apparent_temp","cloud_cover","shortwave_rad",
             "direct_rad","pv_model","temp_anomaly"]
_WX_LAG   = ["lag_1h_temp","lag_5min_ghi","lag_15min_ghi","solar_ghi_delta"]

def _wx_origin(h):
    """Weather features as known at the forecast origin t - h (minutes)."""
    return [f"{c}_o{h}" for c in _WX]

# v1 feature sets (reproduced for fair comparison)
DA_FEATS    = _CAL + _LAG_LONG
SF_FEATS    = _CAL + _LAG_LONG + _LAG_MED

# v2 feature sets
DA_W_FEATS   = _CAL + _LAG_LONG + _WX
SF_W1H_FEATS = _CAL + _LAG_LONG + _LAG_MED + _wx_origin(60)            # 1h lag + weather at origin
SF_S_FEATS   = _CAL + _LAG_LONG + _LAG_SHORT                          # short lags, no weather
SF_W_FEATS   = _CAL + _LAG_LONG + _LAG_SHORT + _wx_origin(5) + _WX_LAG # full (5-min horizon)

# Horizon variants for LGB-SF-W (lags ≥ horizon only)
_LAG_30M_H = ["lag_30min","lag_1h","lag_2h","lag_6h","lag_ren_1h","lag_ren_6h",
               "lag_30min_solar","lag_1h_solar"]
_LAG_15M_H = ["lag_15min","lag_30min","lag_1h","lag_2h","lag_6h","lag_ren_1h","lag_ren_6h",
               "lag_15min_solar","lag_30min_solar","lag_1h_solar"]
SF_W_30M_FEATS = _CAL + _LAG_LONG + _LAG_30M_H + _wx_origin(30) + ["lag_1h_temp"]
SF_W_15M_FEATS = _CAL + _LAG_LONG + _LAG_15M_H + _wx_origin(15) + ["lag_1h_temp","lag_15min_ghi"]


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — MAPE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def mape_arr(actual, forecast):
    mask = (actual > 1000) & pd.notna(forecast) & pd.notna(actual)
    out  = np.full(len(actual), np.nan)
    out[mask] = np.abs(forecast[mask]-actual[mask]) / actual[mask] * 100
    return out

def mape_mean(arr):
    v = arr[~np.isnan(arr)]
    return float(v.mean()) if len(v) else float("nan")

def fit_lgb(X, y, feats, n=600, depth=7, lr=0.03, leaves=63):
    ok = X[feats].notna().all(axis=1)
    m  = lgb.LGBMRegressor(n_estimators=n, max_depth=depth, num_leaves=leaves,
                            learning_rate=lr, subsample=0.85, colsample_bytree=0.9,
                            min_child_samples=20, random_state=42, verbose=-1)
    m.fit(X.loc[ok, feats], y[ok])
    return m, ok.sum()

def fit_gbm(X, y, feats):
    ok = X[feats].notna().all(axis=1)
    m  = GradientBoostingRegressor(n_estimators=400, max_depth=5, learning_rate=0.04,
                                    subsample=0.85, min_samples_leaf=20, random_state=42)
    m.fit(X.loc[ok, feats], y[ok])
    return m, ok.sum()

def predict(model, X, feats):
    ok  = X[feats].notna().all(axis=1)
    out = np.full(len(X), np.nan)
    out[ok] = model.predict(X.loc[ok, feats])
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    df_raw = load_all_features()
    bad    = get_bad_days(df_raw)
    df     = df_raw[~df_raw["Date"].isin(bad)].copy()

    train_all  = df[df["year"]==2024].copy()
    test_2023  = df[df["year"]==2023].copy()
    test_2025  = df[df["year"]==2025].copy()

    for d in [train_all, test_2023, test_2025]:
        d.__class__ = pd.DataFrame   # suppress chained-assignment warning

    # Derived features from training stats only
    print("Computing train-derived features (ren_expected, pv_model, temp_anomaly) …")
    train_all  = add_derived_train_features(train_all, train_all)
    test_2023  = add_derived_train_features(test_2023, train_all)
    test_2025  = add_derived_train_features(test_2025, train_all)

    def allday(d):  return d[d["Current Demand (MW)"]>1000].copy()
    def daytime(d): return d[(d["hour"]>=HOUR_FROM) & (d["hour"]<HOUR_TO) &
                              (d["Current Demand (MW)"]>1000)].copy()
    tr  = allday(train_all)
    t23 = allday(test_2023)
    t25 = allday(test_2025)
    print(f"  Train (all hours): {len(tr):,}  |  Test-2023: {len(t23):,}  |  Test-2025: {len(t25):,}")

    # ── Fit models ────────────────────────────────────────────────────────
    print("\nFitting models …")
    y_tr = tr["Current Demand (MW)"]

    gbm_da,   n1 = fit_gbm(tr, y_tr, DA_FEATS);       print(f"  ✓ GBM-DA     ({n1:,})")
    lgb_sf,   n2 = fit_lgb(tr, y_tr, SF_FEATS);       print(f"  ✓ LGB-SF     ({n2:,})")
    lgb_daw,  n3 = fit_lgb(tr, y_tr, DA_W_FEATS);     print(f"  ✓ LGB-DA-W   ({n3:,})  ← weather")
    lgb_sfw1, n4 = fit_lgb(tr, y_tr, SF_W1H_FEATS);   print(f"  ✓ LGB-SF-W-1h  ({n4:,})  ← 1h lag + weather")
    lgb_sfs,  n5 = fit_lgb(tr, y_tr, SF_S_FEATS);     print(f"  ✓ LGB-SF-S   ({n5:,})  ← short lags")
    lgb_sfw,  n6 = fit_lgb(tr, y_tr, SF_W_FEATS);     print(f"  ✓ LGB-SF-W-5m  ({n6:,})  ← short + weather")
    lgb_sfw30, n7 = fit_lgb(tr, y_tr, SF_W_30M_FEATS); print(f"  ✓ LGB-SF-W-30m ({n7:,})  ← 30-min horizon")
    lgb_sfw15, n8 = fit_lgb(tr, y_tr, SF_W_15M_FEATS); print(f"  ✓ LGB-SF-W-15m ({n8:,})  ← 15-min horizon")

    # ── Score ──────────────────────────────────────────────────────────────
    MAPE_COLS = {
        "NOGA-DA":      "mape_noga_da",
        "NOGA-SF":      "mape_noga_sf",
        "GBM-DA":       "mape_gbm_da",
        "LGB-SF":       "mape_lgb_sf",
        "LGB-DA-W":     "mape_lgb_daw",
        "LGB-SF-W-1h":  "mape_lgb_sfw1",
        "LGB-SF-W-30m": "mape_lgb_sfw30",
        "LGB-SF-W-15m": "mape_lgb_sfw15",
        "LGB-SF-S":     "mape_lgb_sfs",
        "LGB-SF-W-5m":  "mape_lgb_sfw",
    }

    results = {}
    for yr, test in [("2023", t23), ("2025", t25)]:
        test = test.copy()
        act  = test["Current Demand (MW)"].values
        nda  = test["Day-Ahead Demand Forecast (MW)"].where(
               test["Day-Ahead Demand Forecast (MW)"].between(DA_MIN,DA_MAX)).values
        nsf  = test["Updated Demand Forecast (MW)"].where(
               test["Updated Demand Forecast (MW)"].between(DA_MIN,DA_MAX)).values

        test["mape_noga_da"]   = mape_arr(act, nda)
        test["mape_noga_sf"]   = mape_arr(act, nsf)
        test["mape_gbm_da"]    = mape_arr(act, predict(gbm_da,   test, DA_FEATS))
        test["mape_lgb_sf"]    = mape_arr(act, predict(lgb_sf,   test, SF_FEATS))
        test["mape_lgb_daw"]   = mape_arr(act, predict(lgb_daw,  test, DA_W_FEATS))
        test["mape_lgb_sfw1"]  = mape_arr(act, predict(lgb_sfw1, test, SF_W1H_FEATS))
        test["mape_lgb_sfw30"] = mape_arr(act, predict(lgb_sfw30, test, SF_W_30M_FEATS))
        test["mape_lgb_sfw15"] = mape_arr(act, predict(lgb_sfw15, test, SF_W_15M_FEATS))
        test["mape_lgb_sfs"]   = mape_arr(act, predict(lgb_sfs,  test, SF_S_FEATS))
        test["mape_lgb_sfw"]   = mape_arr(act, predict(lgb_sfw,  test, SF_W_FEATS))

        results[yr] = test

    # ═══════════════════════════════════════════════════════════════════════
    # PRINT TABLES
    # ═══════════════════════════════════════════════════════════════════════

    # ── TABLE 1: Overall MAPE ─────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("TABLE 1 — Overall MAPE (%) — All 24 hours")
    print(DIVIDER)
    hdr = (f"{'Model':<12} {'Type':<9} {'2023':>7} {'2025':>7} "
           f"{'Comb.':>7} {'vs NOGA-DA':>11} {'vs NOGA-SF':>11}")
    print(hdr); print("─"*len(hdr))
    summary = {}
    noga_da_c = mape_mean(np.concatenate([results["2023"]["mape_noga_da"].values,
                                           results["2025"]["mape_noga_da"].values]))
    noga_sf_c = mape_mean(np.concatenate([results["2023"]["mape_noga_sf"].values,
                                           results["2025"]["mape_noga_sf"].values]))
    TYPE_MAP = {"NOGA-DA":"DA-base","NOGA-SF":"SF-base","GBM-DA":"DA-v1",
                "LGB-SF":"SF-v1","LGB-DA-W":"DA-v2","LGB-SF-W-1h":"SF-v2",
                "LGB-SF-W-30m":"SF-v2","LGB-SF-W-15m":"SF-v2",
                "LGB-SF-S":"SF-v2","LGB-SF-W-5m":"SF-v2"}
    for mname, col in MAPE_COLS.items():
        v23  = mape_mean(results["2023"][col].values)
        v25  = mape_mean(results["2025"][col].values)
        both = mape_mean(np.concatenate([results["2023"][col].values,
                                          results["2025"][col].values]))
        summary[mname] = (v23, v25, both)
        mtype = TYPE_MAP[mname]
        print(f"{mname:<12} {mtype:<9} {v23:>7.2f} {v25:>7.2f} {both:>7.2f} "
              f"{both-noga_da_c:>+10.2f} {both-noga_sf_c:>+10.2f}")

    # ── TABLE 1b: Horizon Comparison ─────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("TABLE 1b — LGB-SF-W: MAPE (%) by Forecast Horizon (All 24 hours)")
    print(DIVIDER)
    HORIZON_LABELS = {
        "LGB-SF-W-1h":  "~60 min",
        "LGB-SF-W-30m": "~30 min",
        "LGB-SF-W-15m": "~15 min",
        "LGB-SF-W-5m":  "~5 min",
    }
    hdr_h = f"{'Model':<16} {'Horizon':>9} {'2023':>7} {'2025':>7} {'Comb.':>7}"
    print(hdr_h); print("─"*len(hdr_h))
    for mname in HORIZON_ORDER:
        col = MAPE_COLS[mname]
        v23  = mape_mean(results["2023"][col].values)
        v25  = mape_mean(results["2025"][col].values)
        both = mape_mean(np.concatenate([results["2023"][col].values,
                                          results["2025"][col].values]))
        print(f"{mname:<16} {HORIZON_LABELS[mname]:>9} {v23:>7.2f} {v25:>7.2f} {both:>7.2f}")
    # Reference: NOGA-SF
    col_nsf = MAPE_COLS["NOGA-SF"]
    v23_nsf  = mape_mean(results["2023"][col_nsf].values)
    v25_nsf  = mape_mean(results["2025"][col_nsf].values)
    both_nsf = mape_mean(np.concatenate([results["2023"][col_nsf].values,
                                          results["2025"][col_nsf].values]))
    print("─"*len(hdr_h))
    print(f"{'NOGA-SF (ref)':<16} {'assumed 5m':>9} {v23_nsf:>7.2f} {v25_nsf:>7.2f} {both_nsf:>7.2f}")

    # ── TABLE 2: By Season ────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("TABLE 2 — MAPE (%) by Season")
    print(DIVIDER)
    for yr in ["2023","2025"]:
        test = results[yr]
        print(f"\n  Year {yr}:")
        hdr2 = f"  {'Model':<12}" + "".join(f"  {s:>8}" for s in SEASON_ORDER)
        print(hdr2); print("  "+"─"*(len(hdr2)-2))
        for mname, col in MAPE_COLS.items():
            row = f"  {mname:<12}"
            for sea in SEASON_ORDER:
                row += f"  {mape_mean(test.loc[test['season']==sea,col].values):>8.2f}"
            print(row)

    # ── TABLE 3: Security Events (daytime 08:00-20:00 only) ──────────────
    test_all_day = pd.concat([daytime(results["2023"]), daytime(results["2025"])])
    print(f"\n{DIVIDER}")
    print("TABLE 3 — MAPE (%) on Security Event Days (08:00–20:00 only) vs Baseline")
    print(DIVIDER)
    hdr3 = f"{'Event':<24}" + "".join(f"  {m:>9}" for m in MODEL_ORDER)
    print(hdr3); print("─"*len(hdr3))
    for ev_name, dates in SECURITY_EVENTS.items():
        yr = dates[0][:4]
        if yr not in results: continue
        sub = daytime(results[yr][results[yr]["Date"].isin(dates)])
        if sub.empty: continue
        row = f"{ev_name:<24}"
        for mname in MODEL_ORDER:
            row += f"  {mape_mean(sub[MAPE_COLS[mname]].values):>9.2f}"
        print(row)
    row_bl = f"{'Baseline (all days)':<24}"
    for mname in MODEL_ORDER:
        row_bl += f"  {mape_mean(test_all_day[MAPE_COLS[mname]].values):>9.2f}"
    print(row_bl)

    # ── TABLE 4: Feature importance ───────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("TABLE 4 — Feature Importance: LGB-SF-W (gain, top 15)")
    print(DIVIDER)
    fi = pd.Series(lgb_sfw.booster_.feature_importance("gain"),
                   index=SF_W_FEATS).sort_values(ascending=False)
    total = fi.sum()
    for feat, val in fi.head(15).items():
        print(f"  {feat:<20} {val/total*100:>6.1f}%")

    print(f"\n  Feature importance: LGB-SF-W-1h (gain, top 15)")
    fi_w1 = pd.Series(lgb_sfw1.booster_.feature_importance("gain"),
                      index=SF_W1H_FEATS).sort_values(ascending=False)
    total_w1 = fi_w1.sum()
    for feat, val in fi_w1.head(15).items():
        print(f"  {feat:<20} {val/total_w1*100:>6.1f}%")

    print(f"\n  Feature importance: LGB-SF-W-5m (gain, top 15)")
    fi_5m = pd.Series(lgb_sfw.booster_.feature_importance("gain"),
                      index=SF_W_FEATS).sort_values(ascending=False)
    total_5m = fi_5m.sum()
    for feat, val in fi_5m.head(15).items():
        print(f"  {feat:<20} {val/total_5m*100:>6.1f}%")

    print(f"\n  Feature importance: LGB-DA-W (top 10)")
    fi_da = pd.Series(lgb_daw.booster_.feature_importance("gain"),
                      index=DA_W_FEATS).sort_values(ascending=False)
    total_da = fi_da.sum()
    for feat, val in fi_da.head(10).items():
        print(f"  {feat:<20} {val/total_da*100:>6.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════════════

    DA_MODELS = ["NOGA-DA","GBM-DA","LGB-DA-W"]
    SF_MODELS = ["NOGA-SF","LGB-SF","LGB-SF-W-1h","LGB-SF-S","LGB-SF-W-5m"]

    # ── Fig 16: DA comparison before/after weather ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, yr in zip(axes, ["2023","2025"]):
        vals  = [summary[m][0 if yr=="2023" else 1] for m in DA_MODELS]
        cols_ = [MODEL_COLORS[m] for m in DA_MODELS]
        bars  = ax.bar(DA_MODELS, vals, color=cols_, edgecolor="black", lw=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.05,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"DA Models — Test {yr}")
        ax.set_ylabel("MAPE (%)")
        ax.set_ylim(0, max(vals)*1.35)
        ax.set_xticklabels(DA_MODELS, fontsize=10)
    fig.suptitle("DA Forecast: Calendar Only vs +Weather+PV Model\n"
                 "(Train 2024 | Tel Aviv temperature + irradiance)", fontsize=11)
    plt.tight_layout()
    fig.savefig(DATA_DIR/"fig16_da_weather_comparison.png")
    plt.close(fig)
    print("\n  → Saved fig16_da_weather_comparison.png")

    # ── Fig 17: SF comparison — lag length effect ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, yr in zip(axes, ["2023","2025"]):
        vals  = [summary[m][0 if yr=="2023" else 1] for m in SF_MODELS]
        cols_ = [MODEL_COLORS[m] for m in SF_MODELS]
        bars  = ax.bar(SF_MODELS, vals, color=cols_, edgecolor="black", lw=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.02,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"SF Models — Test {yr}")
        ax.set_ylabel("MAPE (%)")
        ax.set_ylim(0, max(vals)*1.4)
        ax.set_xticklabels(SF_MODELS, fontsize=9, rotation=10)
    fig.suptitle("SF Forecast: 1-hour Lag vs 5-min Lag vs +Weather\n"
                 "(shorter lag = fundamentally better tracking)", fontsize=11)
    plt.tight_layout()
    fig.savefig(DATA_DIR/"fig17_sf_lag_comparison.png")
    plt.close(fig)
    print("  → Saved fig17_sf_lag_comparison.png")

    # ── Fig 17b: Horizon comparison — LGB-SF-W at 4 horizons ───────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    horizon_models = HORIZON_ORDER
    horizon_labels = ["1h","30min","15min","5min"]
    horizon_colors = ["#8e44ad","#2980b9","#16a085","#27ae60"]
    for ax, yr in zip(axes, ["2023","2025"]):
        vals_h = [summary[m][0 if yr=="2023" else 1] for m in horizon_models]
        bars   = ax.bar(horizon_labels, vals_h, color=horizon_colors,
                        edgecolor="black", lw=0.6, zorder=3)
        # NOGA-SF reference
        nsf_v = summary["NOGA-SF"][0 if yr=="2023" else 1]
        ax.axhline(nsf_v, color="#e74c3c", ls="--", lw=1.5, label=f"NOGA-SF {nsf_v:.2f}%")
        for bar, v in zip(bars, vals_h):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"LGB-SF-W Horizons — Test {yr}")
        ax.set_ylabel("MAPE (%)")
        ax.set_xlabel("Forecast Horizon")
        ax.set_ylim(0, max(vals_h + [nsf_v])*1.4)
        ax.legend(fontsize=9)
    fig.suptitle("LGB-SF-W: MAPE vs Forecast Horizon\n"
                 "(shorter horizon = more recent demand info = lower MAPE)", fontsize=11)
    plt.tight_layout()
    fig.savefig(DATA_DIR/"fig17b_horizon_comparison.png")
    plt.close(fig)
    print("  → Saved fig17b_horizon_comparison.png")

    # ── Fig 18: MAPE by hour — all 7 models ─────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    hrs = list(range(HOUR_FROM, HOUR_TO))
    for ax, model_group, title in [
            (axes[0], DA_MODELS + ["NOGA-DA"], "DA Models by Hour"),
            (axes[1], SF_MODELS + ["NOGA-SF"], "SF Models by Hour")]:
        seen = set()
        for mname in dict.fromkeys(model_group):  # deduplicate, keep order
            if mname in seen: continue
            seen.add(mname)
            col  = MAPE_COLS[mname]
            h_v  = [mape_mean(pd.concat(list(results.values())).loc[pd.concat(list(results.values()))["hour"]==h, col].values) for h in hrs]
            ls   = "--" if "NOGA" in mname else "-"
            lw   = 1.6 if "W" in mname or "NOGA" in mname else 1.1
            ax.plot(hrs, h_v, color=MODEL_COLORS[mname], lw=lw, ls=ls,
                    marker="o", ms=4, label=mname)
        ax.set_title(title); ax.set_ylabel("MAPE (%)")
        ax.set_xticks(hrs)
        ax.set_xticklabels([f"{h:02d}:00" for h in hrs], rotation=45, fontsize=8)
        ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    fig.savefig(DATA_DIR/"fig18_mape_by_hour.png")
    plt.close(fig)
    print("  → Saved fig18_mape_by_hour.png")

    # ── Fig 19: PV model fit example (Summer 2024 week) ─────────────────
    week = train_all[(train_all["Date"]>="2024-07-15") &
                     (train_all["Date"]<="2024-07-21")].copy()
    if len(week)>0:
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(week["datetime"], week["Solar (MW)"],  "b-",  lw=1.5, label="Actual Solar (MW)")
        ax.plot(week["datetime"], week["pv_model"],    "r--", lw=1.5, label="PV Physics Model")
        ax.fill_between(week["datetime"], week["Solar (MW)"], week["pv_model"],
                        alpha=0.15, color="gray")
        ax.set_xlabel("DateTime"); ax.set_ylabel("MW")
        ax.set_title("Physics-Based PV Model Fit — Sample Week (Jul 15–21, 2024)\n"
                     r"$\mathrm{PV_{pred}} = \max(0,\; \mathrm{GHI} \times k_m "
                     r"\times [1 - \gamma(T-25)])$")
        ax.legend(); ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        fig.savefig(DATA_DIR/"fig19_pv_model_fit.png")
        plt.close(fig)
        print("  → Saved fig19_pv_model_fit.png")

    # ── Fig 20: Full comparison bar chart — all 7 models, 2023 & 2025 ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, yr in zip(axes, ["2023","2025"]):
        vals  = [summary[m][0 if yr=="2023" else 1] for m in MODEL_ORDER]
        cols_ = [MODEL_COLORS[m] for m in MODEL_ORDER]
        bars  = ax.bar(MODEL_ORDER, vals, color=cols_, edgecolor="black", lw=0.6, zorder=3)
        # Horizontal reference lines
        noga_da_v = summary["NOGA-DA"][0 if yr=="2023" else 1]
        noga_sf_v = summary["NOGA-SF"][0 if yr=="2023" else 1]
        ax.axhline(noga_da_v, color="#c0392b", ls=":", lw=1.2, alpha=0.7)
        ax.axhline(noga_sf_v, color="#e74c3c", ls="--", lw=1.0, alpha=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.03,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"All Models — Test {yr}")
        ax.set_ylabel("MAPE (%)")
        ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, max(vals)*1.35)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0],color="#c0392b",ls=":",lw=1.5,label="NOGA-DA ref"),
        Line2D([0],[0],color="#e74c3c",ls="--",lw=1.5,label="NOGA-SF ref"),
    ]
    axes[1].legend(handles=legend_elements, fontsize=9)
    fig.suptitle("Full Model Comparison: v1 (calendar) vs v2 (weather + short lags)\n"
                 "Train: 2024  |  08:00–20:00  |  Red dots = NOGA reference", fontsize=11)
    plt.tight_layout()
    fig.savefig(DATA_DIR/"fig20_full_comparison.png")
    plt.close(fig)
    print("  → Saved fig20_full_comparison.png")

    print(f"\n{DIVIDER}")
    print("ALL V2 ANALYSES COMPLETE  —  figures fig16–fig20 saved")
    print(DIVIDER)

    return results, summary


if __name__ == "__main__":
    run()
