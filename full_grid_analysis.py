#!/usr/bin/env python3
"""
Full NOGA-ISO Israeli Grid Analysis
=====================================
Analyses (all hours unless stated; 08-20 for forecast error):
  A. Forecast error — security events only (MAPE %, daily + 30-min)
  B. Renewable penetration trend  (monthly solar+wind / demand)
  C. Duck curve + evening ramp analysis  (seasonal daily profiles)
  D. CO2 intensity trend  (monthly + hourly)
  E. Forecast error vs renewable penetration  (binned)
  F. BESS dispatch patterns  (hourly by season)
  G. Security event impact on production mix
  H. Forecast error by solar phase  (morning-ramp / solar-peak / decline / evening)

Outputs:  9 PNG figures + printed tables  (same folder as script)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent
DEMAND_CSV = DATA_DIR / "noga_demand_5min.csv"
PROD_CSV   = DATA_DIR / "noga_production_mix_5min.csv"
CO2_CSV    = DATA_DIR / "noga_co2_5min.csv"

# ── Security events (weather events excluded — handled well by NOGA model) ────
SECURITY_EVENTS = [
    ("Hamas day-after",   "Oct 8, 2023",          ["2023-10-08"]),
    ("True Promise I",    "Apr 13–14, 2024",       ["2024-04-13", "2024-04-14"]),
    ("True Promise II",   "Oct 1, 2024",           ["2024-10-01"]),
    ("Rising Lion",       "Jun 13–16, 2025",       ["2025-06-13", "2025-06-14",
                                                    "2025-06-15", "2025-06-16"]),
]
ALL_EVENT_DATES = [d for _, _, ds in SECURITY_EVENTS for d in ds]

# ── Constants ─────────────────────────────────────────────────────────────────
HOUR_FROM, HOUR_TO = 8, 20
DA_MIN, DA_MAX     = 2000, 25000
FROZEN_THR         = 0.10
TOP_N              = 10
BIN_MIN            = 30    # minutes per MAPE bin

SEASON_MAP = {12:"Winter",1:"Winter",2:"Winter",
              3:"Spring", 4:"Spring", 5:"Spring",
              6:"Summer", 7:"Summer", 8:"Summer",
              9:"Autumn", 10:"Autumn",11:"Autumn"}
SEASON_ORDER  = ["Winter","Spring","Summer","Autumn"]
SEASON_COLORS = {"Winter":"#1f77b4","Spring":"#2ca02c",
                 "Summer":"#d62728","Autumn":"#ff7f0e"}
YEAR_COLORS   = {2023:"#1f77b4", 2024:"#ff7f0e", 2025:"#2ca02c"}

# Contextual explanation for worst-MAPE days
DAY_NOTES = {
    "2023-12-01": "Early-winter cold snap; DA model calibrated on warmer months",
    "2025-11-19": "Nov 2025 SF anomaly cluster; DA unaffected",
    "2025-11-20": "Nov 2025 SF anomaly cluster; DA unaffected",
    "2025-11-21": "Nov 2025 SF anomaly cluster; DA unaffected",
    "2025-11-22": "Nov 2025 SF anomaly cluster; DA unaffected",
    "2025-11-23": "Nov 2025 SF anomaly cluster; DA unaffected",
}

def auto_note(date_str):
    if date_str in DAY_NOTES:
        return DAY_NOTES[date_str]
    month = int(date_str[5:7])
    if month in (12, 1, 2):
        return "Winter cold-season; DA model underestimates morning heating demand"
    if month in (3, 4):
        return "Spring transition; rapid demand-temperature swing"
    if month in (10, 11):
        return "Autumn transition; early cold snap or seasonal onset"
    return "Summer anomaly; unusual demand or renewable surge"

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3,
})

DIVIDER = "=" * 90

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & MERGING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all():
    print("Loading CSVs …")

    # ── Demand ──────────────────────────────────────────────────────────────
    dem = pd.read_csv(DEMAND_CSV, encoding="utf-8-sig")
    dem.rename(columns={"Interval (1-288)": "Interval"}, inplace=True)
    dem["datetime"] = pd.to_datetime(dem["Date"] + " " + dem["Time"])
    for c in ["Current Demand (MW)", "Updated Demand Forecast (MW)",
              "Day-Ahead Demand Forecast (MW)"]:
        dem[c] = pd.to_numeric(dem[c], errors="coerce")

    # ── Production mix ───────────────────────────────────────────────────────
    prod = pd.read_csv(PROD_CSV, encoding="utf-8-sig")
    prod.rename(columns={"Interval (1-288)": "Interval"}, inplace=True)
    prod["datetime"] = pd.to_datetime(prod["Date"] + " " + prod["Time"])
    for c in prod.columns:
        if c not in ("Date", "Time", "datetime", "Interval"):
            prod[c] = pd.to_numeric(prod[c], errors="coerce").fillna(0)

    # ── CO2 ─────────────────────────────────────────────────────────────────
    co2 = pd.read_csv(CO2_CSV, encoding="utf-8-sig")
    co2.rename(columns={"Interval (1-288)": "Interval"}, inplace=True)
    co2["datetime"] = pd.to_datetime(co2["Date"] + " " + co2["Time"])
    for c in co2.columns:
        if c not in ("Date", "Time", "datetime", "Interval"):
            co2[c] = pd.to_numeric(co2[c], errors="coerce")

    # ── Merge ────────────────────────────────────────────────────────────────
    prod_cols = ["datetime", "Coal (MW)", "Gas (MW)", "Diesel/Soler (MW)", "Mazut (MW)",
                 "Battery Storage (MW)", "Photovoltaic (MW)", "PV (MW)",
                 "Thermo-Solar (MW)", "Wind (MW)", "Bio-Gas (MW)",
                 "Renewables Total (MW)", "Actual Demand (MW)", "Net Demand (MW)"]
    co2_cols  = ["datetime", "Total Emissions (mTCO2/h)", "Emission Ratio (tCO2/MWh)",
                 "Coal Emissions (mTCO2/h)", "Gas Emissions (mTCO2/h)"]

    df = dem.merge(prod[prod_cols], on="datetime", how="left")
    df = df.merge(co2[co2_cols],  on="datetime", how="left")

    # ── Derived columns ──────────────────────────────────────────────────────
    df["hour"]   = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["month"]  = df["datetime"].dt.month
    df["year"]   = df["datetime"].dt.year
    df["season"] = df["month"].map(SEASON_MAP)
    df["ym"]     = df["datetime"].dt.to_period("M")

    df["Solar (MW)"]   = (df["PV (MW)"].fillna(0)
                         + df["Photovoltaic (MW)"].fillna(0)
                         + df["Thermo-Solar (MW)"].fillna(0))
    df["Fossil (MW)"]  = (df["Coal (MW)"].fillna(0)
                         + df["Gas (MW)"].fillna(0)
                         + df["Diesel/Soler (MW)"].fillna(0)
                         + df["Mazut (MW)"].fillna(0))
    ren   = df["Renewables Total (MW)"].fillna(0)
    adm   = df["Actual Demand (MW)"]
    df["Renewable Share (%)"] = np.where(adm > 0, ren / adm * 100, np.nan)
    df["Solar Share (%)"]     = np.where(adm > 0,
                                          df["Solar (MW)"] / adm * 100, np.nan)

    # MAPE
    act = df["Current Demand (MW)"]
    da  = df["Day-Ahead Demand Forecast (MW)"]
    sf  = df["Updated Demand Forecast (MW)"]
    da_ok = da.between(DA_MIN, DA_MAX) & (act > 1000)
    sf_ok = sf.between(DA_MIN, DA_MAX) & (act > 1000)
    df["DA MAPE"] = np.where(da_ok, (da - act).abs() / act * 100, np.nan)
    df["SF MAPE"] = np.where(sf_ok, (sf - act).abs() / act * 100, np.nan)

    # 30-min bin slot label
    df["slot"] = (df["hour"] * 60 + df["minute"]) // BIN_MIN * BIN_MIN
    df["slot_label"] = df["slot"].apply(lambda s: f"{s//60:02d}:{s%60:02d}")

    print(f"  {len(df):,} rows  |  {df['Date'].nunique()} calendar dates  |  "
          f"{df['year'].min()}–{df['year'].max()}")
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


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS A — Forecast Error: Security Events Only
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_A(df):
    print(f"\n{DIVIDER}")
    print("ANALYSIS A — FORECAST ERROR (security events only, 08:00–20:00)")
    print(DIVIDER)

    win     = df[(df["hour"] >= HOUR_FROM) & (df["hour"] < HOUR_TO)].copy()  # 08-20 for security events
    win_all = df[df["Current Demand (MW)"] > 1000].copy()                    # all hours for top-10 / baseline

    # Daily MAPE (08-20) — for security events table and fig8
    daily_da_ev = win.groupby("Date")["DA MAPE"].mean().dropna()
    daily_sf_ev = win.groupby("Date")["SF MAPE"].mean().dropna()
    bl_da_ev = float(np.nanmean(sorted(daily_da_ev.values, reverse=True)))
    bl_sf_ev = float(np.nanmean(sorted(daily_sf_ev.values, reverse=True)))

    # Daily MAPE (all hours) — for top-10 table and fig9 and baseline
    daily_da  = win_all.groupby("Date")["DA MAPE"].mean().dropna()
    daily_sf  = win_all.groupby("Date")["SF MAPE"].mean().dropna()
    s_da = sorted(daily_da.values, reverse=True)
    s_sf = sorted(daily_sf.values, reverse=True)
    n    = len(daily_da)
    bl_da = float(np.nanmean(s_da))
    bl_sf = float(np.nanmean(s_sf))
    print(f"  Clean days: {n}  |  Baseline MAPE (all hours)  DA={bl_da:.2f}%  SF={bl_sf:.2f}%")

    # 30-min bins — keep 08-20 for security event worst-window analysis
    bin_da = win.groupby(["Date","slot_label"])["DA MAPE"].mean().dropna()
    bin_sf = win.groupby(["Date","slot_label"])["SF MAPE"].mean().dropna()
    s_bda  = sorted(bin_da.values, reverse=True)
    s_bsf  = sorted(bin_sf.values, reverse=True)
    nb = len(bin_da)
    bl_bda = float(np.nanmean(s_bda)); bl_bsf = float(np.nanmean(s_bsf))

    def rk(val, srt): return sum(1 for v in srt if v >= val)

    # ── TABLE A1: Events daily ───────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print("TABLE A1 — Security events: daily MAPE (08:00–20:00)")
    print(f"{'─'*90}")
    hdr = (f"{'Event':<22} {'Date(s)':<17} {'DA%':>7} {'SF%':>7} "
           f"{'ΔDA%':>7} {'ΔSF%':>7} {'DArank':>9} {'SFrank':>9}")
    print(hdr); print("─"*len(hdr))
    s_da_ev = sorted(daily_da_ev.values, reverse=True)
    s_sf_ev = sorted(daily_sf_ev.values, reverse=True)
    n_ev = len(daily_da_ev)
    ev_rows = []
    for name, dates_str, dates in SECURITY_EVENTS:
        da_e  = [daily_da_ev[d] for d in dates if d in daily_da_ev]
        sf_e  = [daily_sf_ev[d] for d in dates if d in daily_sf_ev]
        avg_da = float(np.mean(da_e)) if da_e else float("nan")
        avg_sf = float(np.mean(sf_e)) if sf_e else float("nan")
        rda    = rk(avg_da, s_da_ev) if not np.isnan(avg_da) else "—"
        rsf    = rk(avg_sf, s_sf_ev) if not np.isnan(avg_sf) else "—"
        dda = avg_da - bl_da_ev; dsf = avg_sf - bl_sf_ev
        print(f"{name:<22} {dates_str:<17} {avg_da:>7.2f} {avg_sf:>7.2f} "
              f"{dda:>+7.2f} {dsf:>+7.2f} {str(rda):>7}/{n_ev}  {str(rsf):>7}/{n_ev}")
        ev_rows.append((name, dates_str, avg_da, avg_sf, dda, dsf, rda, rsf, n_ev))
    print(f"{'Baseline (all days)':<22} {'—':<17} {bl_da_ev:>7.2f} {bl_sf_ev:>7.2f}")

    # ── TABLE A2: Top N days by DA MAPE ─────────────────────────────────────
    print(f"\nTABLE A2 — Top {TOP_N} days by daily DA MAPE (with season and context)")
    print(f"{'─'*90}")
    print(f"{'Rank':<5} {'Date':<12} {'Season':<8} {'DA%':>7} {'SF%':>7} "
          f"{'DA rk':>8} {'SF rk':>8}  Context")
    print("─"*90)
    for rank, (date, val) in enumerate(
            sorted(daily_da.items(), key=lambda x: -x[1])[:TOP_N], 1):
        m   = int(date[5:7])
        sea = SEASON_MAP[m]
        sf_v = daily_sf.get(date, float("nan"))
        rksf = rk(sf_v, s_sf) if not np.isnan(sf_v) else "—"
        note = auto_note(date)
        print(f"{rank:<5} {date:<12} {sea:<8} {val:>7.2f} {sf_v:>7.2f} "
              f"{rank:>6}/{n}  {str(rksf):>6}/{n}  {note}")

    # ── TABLE A3: Top N days by SF MAPE ─────────────────────────────────────
    print(f"\nTABLE A3 — Top {TOP_N} days by daily SF MAPE (with season and context)")
    print(f"{'─'*90}")
    print(f"{'Rank':<5} {'Date':<12} {'Season':<8} {'SF%':>7} {'DA%':>7} "
          f"{'SF rk':>8} {'DA rk':>8}  Context")
    print("─"*90)
    for rank, (date, val) in enumerate(
            sorted(daily_sf.items(), key=lambda x: -x[1])[:TOP_N], 1):
        m   = int(date[5:7])
        sea = SEASON_MAP[m]
        da_v = daily_da.get(date, float("nan"))
        rkda = rk(da_v, s_da) if not np.isnan(da_v) else "—"
        note = auto_note(date)
        print(f"{rank:<5} {date:<12} {sea:<8} {val:>7.2f} {da_v:>7.2f} "
              f"{rank:>6}/{n}  {str(rkda):>6}/{n}  {note}")

    # ── TABLE A4: Events worst 30-min window ─────────────────────────────────
    print(f"\nTABLE A4 — Security events: worst 30-min window")
    print(f"{'─'*90}")
    hdr2 = (f"{'Event':<22} {'DA time':>12} {'DA%':>7} {'ΔDA%':>7} {'DA rk':>10}"
            f"  {'SF time':>12} {'SF%':>7} {'ΔSF%':>7} {'SF rk':>10}")
    print(hdr2); print("─"*len(hdr2))
    for name, dates_str, dates in SECURITY_EVENTS:
        ev_bda = {(d,s): bin_da.get((d,s), np.nan) for d in dates
                  for s in df["slot_label"].unique() if not np.isnan(bin_da.get((d,s), np.nan))}
        ev_bsf = {(d,s): bin_sf.get((d,s), np.nan) for d in dates
                  for s in df["slot_label"].unique() if not np.isnan(bin_sf.get((d,s), np.nan))}
        if not ev_bda or not ev_bsf: continue
        wdk, wdv = max(ev_bda.items(), key=lambda x: x[1])
        wsk, wsv = max(ev_bsf.items(), key=lambda x: x[1])
        da_lbl = f"{wdk[0][5:]} {wdk[1]}"
        sf_lbl = f"{wsk[0][5:]} {wsk[1]}"
        print(f"{name:<22} {da_lbl:>12} {wdv:>7.2f} {wdv-bl_bda:>+7.2f} "
              f"{rk(wdv,s_bda):>8}/{nb}  "
              f"{sf_lbl:>12} {wsv:>7.2f} {wsv-bl_bsf:>+7.2f} "
              f"{rk(wsv,s_bsf):>8}/{nb}")

    # ── FIGURE A: Security events bar chart ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names  = [row[0] for row in ev_rows]
    da_v   = [row[2] for row in ev_rows]
    sf_v   = [row[3] for row in ev_rows]
    x      = np.arange(len(names))
    w      = 0.35

    for ax, vals, metric, bl in zip(axes, [da_v, sf_v], ["DA", "SF"], [bl_da_ev, bl_sf_ev]):
        bars = ax.bar(x, vals, w, color=["#c0392b" if v > bl else "#2980b9" for v in vals],
                      edgecolor="black", linewidth=0.6, zorder=3)
        ax.axhline(bl, color="black", linestyle="--", linewidth=1.2,
                   label=f"Baseline {bl:.2f}%")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9, rotation=12, ha="right")
        ax.set_ylabel("MAPE (%)")
        ax.set_title(f"{metric} Forecast Error — Security Events vs Baseline")
        ax.legend(fontsize=9); ax.set_ylim(0, max(vals+[bl])*1.25)

    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig8_security_events_mape.png")
    plt.close(fig)
    print("\n  → Saved fig8_security_events_mape.png")

    # ── FIGURE A2: Top-10 days ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, daily, label, s_vals, bl in [
            (axes[0], daily_da, "DA MAPE (%)", s_da, bl_da),
            (axes[1], daily_sf, "SF MAPE (%)", s_sf, bl_sf)]:
        top = sorted(daily.items(), key=lambda x: -x[1])[:TOP_N]
        dates_lbl = [d for d,_ in top]
        vals_     = [v for _,v in top]
        months_   = [int(d[5:7]) for d in dates_lbl]
        colors_   = [SEASON_COLORS[SEASON_MAP[m]] for m in months_]
        y_ = np.arange(len(dates_lbl))
        ax.barh(y_, vals_, color=colors_, edgecolor="black", linewidth=0.5)
        ax.axvline(bl, color="black", linestyle="--", linewidth=1.1, label=f"Baseline {bl:.2f}%")
        ax.set_yticks(y_); ax.set_yticklabels(dates_lbl, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(label); ax.set_title(f"Top {TOP_N} Days — {label}")
        ax.legend(fontsize=9)
        from matplotlib.patches import Patch
        patches = [Patch(color=c, label=s) for s,c in SEASON_COLORS.items()]
        ax.legend(handles=patches+[plt.Line2D([0],[0],color="black",ls="--",
                                              label=f"Baseline {bl:.2f}%")],
                  fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig9_top10_days.png")
    plt.close(fig)
    print("  → Saved fig9_top10_days.png")

    return bl_da, bl_sf, ev_rows, daily_da, daily_sf


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS B — Renewable Penetration Trend
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_B(df):
    print(f"\n{DIVIDER}")
    print("ANALYSIS B — RENEWABLE PENETRATION TREND")
    print(DIVIDER)

    # Monthly averages (all 24h)
    mon = df.groupby("ym").agg(
        ren_share=("Renewable Share (%)", "mean"),
        solar_share=("Solar Share (%)", "mean"),
        wind_mw=("Wind (MW)", "mean"),
        solar_mw=("Solar (MW)", "mean"),
        demand_mw=("Current Demand (MW)", "mean"),
    ).reset_index()
    mon["ym_str"] = mon["ym"].astype(str)
    mon = mon.sort_values("ym")

    # Annual statistics
    df_an = df.copy()
    ann = df_an.groupby("year").agg(
        ren_share=("Renewable Share (%)", "mean"),
        solar_share=("Solar Share (%)", "mean"),
        wind_avg=("Wind (MW)", "mean"),
        solar_avg=("Solar (MW)", "mean"),
    )
    print("\nTABLE B1 — Annual renewable penetration statistics")
    print(f"{'─'*60}")
    print(f"{'Year':<6} {'Ren% avg':>10} {'Solar% avg':>12} {'Solar MW':>10} {'Wind MW':>9}")
    print("─"*60)
    for yr, row in ann.iterrows():
        print(f"{yr:<6} {row.ren_share:>10.1f} {row.solar_share:>12.1f} "
              f"{row.solar_avg:>10.0f} {row.wind_avg:>9.0f}")

    # Seasonal breakdown
    sea_ann = df_an.groupby(["year","season"])["Renewable Share (%)"].mean().unstack("season")
    print("\nTABLE B2 — Renewable share (%) by year and season")
    print(f"{'─'*60}")
    print(f"{'Year':<6}", end="")
    for s in SEASON_ORDER:
        if s in sea_ann.columns: print(f"  {s:>8}", end="")
    print()
    print("─"*60)
    for yr, row in sea_ann.iterrows():
        print(f"{yr:<6}", end="")
        for s in SEASON_ORDER:
            if s in row.index:
                v = row[s]
                print(f"  {v:>8.1f}" if not np.isnan(v) else f"  {'—':>8}", end="")
        print()

    # ── FIGURE B ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Renewable share %
    ax = axes[0]
    ax.fill_between(range(len(mon)), mon["ren_share"].values,
                    alpha=0.3, color="#2ca02c", label="Renewables Total")
    ax.plot(range(len(mon)), mon["ren_share"].values, color="#2ca02c", lw=1.5)
    ax.fill_between(range(len(mon)), mon["solar_share"].values,
                    alpha=0.3, color="#ff7f0e", label="Solar only")
    ax.plot(range(len(mon)), mon["solar_share"].values, color="#ff7f0e", lw=1.5, ls="--")
    ax.set_ylabel("Share of Demand (%)")
    ax.set_title("Monthly Renewable Penetration — Israel Grid (Apr 2023 – Dec 2025)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # Panel 2: Renewables Total MW vs Total Demand MW (daytime 06:00–20:00)
    df_day = df[df["hour"].between(6, 19)]
    mon_day = df_day.groupby("ym").agg(
        wind_mw=("Wind (MW)", "mean"),
        solar_mw=("Solar (MW)", "mean"),
        demand_mw=("Current Demand (MW)", "mean"),
    ).reset_index().sort_values("ym")
    ax2 = axes[1]
    ren_total = mon_day["solar_mw"].values + mon_day["wind_mw"].values
    ax2.fill_between(range(len(mon_day)), ren_total, alpha=0.35,
                     color="#2ca02c", label="Renewables Total (MW)")
    ax2.plot(range(len(mon_day)), ren_total, color="#2ca02c", lw=1.5)
    ax2.plot(range(len(mon_day)), mon_day["demand_mw"].values,
             color="#d62728", lw=2, ls="--", label="Total Demand (MW)")
    ax2.set_ylabel("Average MW (06:00–20:00)")
    ax2.set_title("Monthly Renewables Total vs Total Demand (MW, 06:00–20:00)")
    ax2.legend(loc="upper left")

    ticks = list(range(0, len(mon), 3))
    lbels = [mon["ym_str"].iloc[i] for i in ticks]
    for ax in axes:
        ax.set_xticks(ticks); ax.set_xticklabels(lbels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig1_renewable_trend.png")
    plt.close(fig)
    print("\n  → Saved fig1_renewable_trend.png")

    return mon, ann


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS C — Duck Curve + Evening Ramp
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_C(df):
    print(f"\n{DIVIDER}")
    print("ANALYSIS C — DUCK CURVE & EVENING RAMP ANALYSIS")
    print(DIVIDER)

    # Average hourly profiles by season (full day)
    hourly = df.groupby(["season", "hour"]).agg(
        demand=("Current Demand (MW)", "mean"),
        solar=("Solar (MW)", "mean"),
        net_demand=("Net Demand (MW)", "mean"),
    ).reset_index()

    # Evening ramp: max MW/h rate from 15:00 to 20:00
    ramp_stats = []
    for (season, year), g in df.groupby(["season", "year"]):
        eve = g[(g["hour"] >= 15) & (g["hour"] < 20)]
        if eve.empty: continue
        # MW change over the window (20:00 demand - 15:00 demand)
        h15 = g[g["hour"] == 15]["Current Demand (MW)"].mean()
        h19 = g[g["hour"] == 19]["Current Demand (MW)"].mean()
        ramp_mw = h19 - h15
        ramp_stats.append({"season": season, "year": year, "ramp_mw": ramp_mw,
                           "h15_mw": h15, "h19_mw": h19})
    ramp_df = pd.DataFrame(ramp_stats)

    print("\nTABLE C1 — Evening ramp (MW) 15:00→20:00 by season and year")
    print(f"{'─'*55}")
    piv = ramp_df.pivot_table(index="year", columns="season", values="ramp_mw", aggfunc="mean")
    print(f"{'Year':<6}", end="")
    for s in SEASON_ORDER:
        if s in piv.columns: print(f"  {s:>8}", end="")
    print(); print("─"*55)
    for yr, row in piv.iterrows():
        print(f"{yr:<6}", end="")
        for s in SEASON_ORDER:
            if s in piv.columns:
                v = row.get(s, np.nan)
                print(f"  {v:>+8.0f}" if not np.isnan(v) else f"  {'—':>8}", end="")
        print()

    # Midday solar overgeneration (solar > demand / 2 indicator)
    solar_peak = df[(df["hour"] >= 10) & (df["hour"] <= 14)].copy()
    solar_peak["solar_frac"] = np.where(
        solar_peak["Current Demand (MW)"] > 500,
        solar_peak["Solar (MW)"] / solar_peak["Current Demand (MW)"],
        np.nan)
    sp_mon = solar_peak.groupby(["year","month"])["solar_frac"].mean()
    print("\nTABLE C2 — Solar fraction of demand at midday (10:00–14:00) by month/year")
    print(f"{'─'*55}")
    sp_piv = sp_mon.unstack("year")
    print(f"{'Month':<8}", end="")
    for yr in sorted(sp_piv.columns): print(f"  {yr:>6}", end="")
    print(); print("─"*55)
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in range(1, 13):
        if m not in sp_piv.index: continue
        row = sp_piv.loc[m]
        print(f"{MONTHS[m-1]:<8}", end="")
        for yr in sorted(sp_piv.columns):
            v = row.get(yr, np.nan)
            print(f"  {v:>6.2f}" if not np.isnan(v) else f"  {'—':>6}", end="")
        print()

    # ── FIGURE C ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    hrs = list(range(24))

    for i, season in enumerate(SEASON_ORDER):
        ax  = axes[i]
        sub = hourly[hourly["season"] == season]
        ax.plot(sub["hour"], sub["demand"], "k-",  lw=2.0, label="Demand (MW)", zorder=4)
        ax.fill_between(sub["hour"], sub["solar"], alpha=0.5,
                        color="#ff7f0e", label="Solar (MW)")
        ax.plot(sub["hour"], sub["solar"],  color="#ff7f0e", lw=1.5, zorder=3)
        # Net demand
        ax.plot(sub["hour"], sub["demand"]-sub["solar"], color="#1f77b4",
                lw=1.5, ls="--", label="Net Demand (MW)", alpha=0.8)
        ax.set_title(f"{season}", fontsize=12, fontweight="bold",
                     color=SEASON_COLORS[season])
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("MW")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=8, rotation=30)
        ax.legend(fontsize=8)

    fig.suptitle("Israeli Grid Duck Curve — Average Daily Profiles by Season\n"
                 "(Apr 2023 – Dec 2025)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig2_duck_curve.png")
    plt.close(fig)
    print("\n  → Saved fig2_duck_curve.png")

    return ramp_df


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS D — CO2 Intensity
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_D(df):
    print(f"\n{DIVIDER}")
    print("ANALYSIS D — CO2 INTENSITY TREND  (from Jun 2023)")
    print(DIVIDER)

    # Filter to physically plausible range (coal ≈0.9, gas CCGT ≈0.4, mix 0.1–1.2)
    co2 = df[df["Emission Ratio (tCO2/MWh)"].notna() &
             df["Emission Ratio (tCO2/MWh)"].between(0.05, 1.5)].copy()

    # Monthly average CO2 intensity
    mon_co2 = co2.groupby("ym").agg(
        intensity=("Emission Ratio (tCO2/MWh)", "mean"),
        total_em=("Total Emissions (mTCO2/h)", "mean"),
        demand=("Current Demand (MW)", "mean"),
    ).reset_index()
    mon_co2["ym_str"] = mon_co2["ym"].astype(str)
    mon_co2 = mon_co2.sort_values("ym")

    # Hourly average
    hourly_co2 = co2.groupby("hour")["Emission Ratio (tCO2/MWh)"].mean()

    # Annual
    ann_co2 = co2.groupby("year")["Emission Ratio (tCO2/MWh)"].mean()
    print("\nTABLE D1 — Annual average CO2 intensity (tCO2/MWh)")
    print(f"{'─'*40}")
    for yr, v in ann_co2.items():
        print(f"  {yr}: {v:.4f} tCO2/MWh")

    print("\nTABLE D2 — CO2 intensity by hour of day (avg all years)")
    print(f"{'─'*40}")
    print(f"{'Hour':<6} {'tCO2/MWh':>12}")
    print("─"*40)
    for h, v in hourly_co2.items():
        bar = "█" * int(v * 50)
        print(f"{h:02d}:00  {v:>10.4f}  {bar}")

    # ── FIGURE D ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: monthly trend
    ax = axes[0]
    x_ = range(len(mon_co2))
    ax.plot(x_, mon_co2["intensity"].values, "o-", color="#8e44ad", lw=2, ms=5)
    ax.fill_between(x_, mon_co2["intensity"].values, alpha=0.2, color="#8e44ad")
    ticks = list(range(0, len(mon_co2), 2))
    ax.set_xticks(ticks)
    ax.set_xticklabels([mon_co2["ym_str"].iloc[i] for i in ticks],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("CO₂ Intensity (tCO₂/MWh)")
    ax.set_title("Monthly Average Grid CO₂ Intensity\n(Jun 2023 – Dec 2025)")

    # Panel 2: hourly profile
    ax2 = axes[1]
    hrs_ = hourly_co2.index.tolist()
    ax2.bar(hrs_, hourly_co2.values, color="#8e44ad", alpha=0.7, edgecolor="black", lw=0.4)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("CO₂ Intensity (tCO₂/MWh)")
    ax2.set_title("Average CO₂ Intensity by Hour of Day\n(lower = more renewable on grid)")
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig3_co2_intensity.png")
    plt.close(fig)
    print("\n  → Saved fig3_co2_intensity.png")

    return mon_co2


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS E — Forecast Error vs Renewable Penetration
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_E(df):
    print(f"\n{DIVIDER}")
    print("ANALYSIS E — FORECAST ERROR vs RENEWABLE PENETRATION")
    print(DIVIDER)

    win = df[df["DA MAPE"].notna() & df["Renewable Share (%)"].notna()].copy()

    # Bin renewable share in 5% buckets
    bins   = list(range(0, 80, 5)) + [100]
    labels = [f"{b}–{b+5}%" for b in bins[:-1]]
    win["ren_bin"] = pd.cut(win["Renewable Share (%)"], bins=bins, labels=labels, right=False)

    grp = win.groupby("ren_bin", observed=True).agg(
        da_mape=("DA MAPE", "mean"),
        sf_mape=("SF MAPE", "mean"),
        count=("DA MAPE", "count"),
    ).dropna()

    print("\nTABLE E1 — MAPE by renewable penetration bin (all 24 hours)")
    print(f"{'─'*55}")
    print(f"{'Ren. share bin':<16} {'DA MAPE%':>10} {'SF MAPE%':>10} {'Count':>8}")
    print("─"*55)
    for idx, row in grp.iterrows():
        print(f"{str(idx):<16} {row.da_mape:>10.2f} {row.sf_mape:>10.2f} {int(row['count']):>8}")

    # ── FIGURE E ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x_  = range(len(grp))
    ax.plot(x_, grp["da_mape"].values, "o-", color="#c0392b", lw=2, ms=6, label="DA MAPE %")
    ax.plot(x_, grp["sf_mape"].values, "s-", color="#2980b9", lw=2, ms=6, label="SF MAPE %")
    ax2b = ax.twinx()
    ax2b.bar(x_, grp["count"].values, alpha=0.2, color="gray", label="Count")
    ax2b.set_ylabel("Interval Count", color="gray")
    ax.set_xticks(x_); ax.set_xticklabels(grp.index.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Renewable Share (% of demand)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Forecast Error vs Renewable Penetration Level\n(all 24 hours, all clean days)")
    lines1, labs1 = ax.get_legend_handles_labels()
    ax.legend(lines1, labs1, fontsize=9)
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig4_forecast_vs_renewable.png")
    plt.close(fig)
    print("\n  → Saved fig4_forecast_vs_renewable.png")

    return grp


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS F — BESS Dispatch Patterns
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_F(df):
    print(f"\n{DIVIDER}")
    print("ANALYSIS F — BESS DISPATCH PATTERNS")
    print(DIVIDER)

    bess = df[df["Battery Storage (MW)"] != 0].copy()
    print(f"  BESS non-zero intervals: {len(bess):,}  "
          f"(charge min={df['Battery Storage (MW)'].min():.1f} MW, "
          f"discharge max={df['Battery Storage (MW)'].max():.1f} MW)")

    # Hourly mean by season (includes zeros)
    hourly_bess = df.groupby(["season","hour"])["Battery Storage (MW)"].mean().reset_index()

    # Annual stats
    ann_bess = df.groupby("year").agg(
        charge_avg=("Battery Storage (MW)", lambda x: x[x<0].mean()),
        discharge_avg=("Battery Storage (MW)", lambda x: x[x>0].mean()),
        charge_hrs=("Battery Storage (MW)", lambda x: (x<0).sum() / 12),
        discharge_hrs=("Battery Storage (MW)", lambda x: (x>0).sum() / 12),
    )
    print("\nTABLE F1 — BESS annual dispatch statistics")
    print(f"{'─'*65}")
    print(f"{'Year':<6} {'Avg Chg MW':>12} {'Avg Dis MW':>12} {'Chg h/yr':>10} {'Dis h/yr':>10}")
    print("─"*65)
    for yr, row in ann_bess.iterrows():
        print(f"{yr:<6} {row.charge_avg:>12.2f} {row.discharge_avg:>12.2f} "
              f"{row.charge_hrs:>10.0f} {row.discharge_hrs:>10.0f}")

    # Correlation with solar (discharge peaks at evening when solar drops)
    corr_da = df[df["Battery Storage (MW)"] > 0.1]["hour"].value_counts().sort_index()
    print("\nTABLE F2 — BESS discharge hour distribution (intervals per hour)")
    print(f"{'─'*45}")
    for h, cnt in corr_da.items():
        bar = "█" * (cnt // 50)
        print(f"  {h:02d}:00  {cnt:>6}  {bar}")

    # ── FIGURE F ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    # Panel 1: average BESS by hour per season
    for season in SEASON_ORDER:
        sub = hourly_bess[hourly_bess["season"] == season]
        ax1.plot(sub["hour"], sub["Battery Storage (MW)"],
                 color=SEASON_COLORS[season], lw=1.8, label=season, marker="o", ms=4)
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.fill_between(range(24), 0, color="none")
    ax1.set_xlabel("Hour of Day"); ax1.set_ylabel("BESS MW (+ = discharge)")
    ax1.set_title("Average BESS Dispatch by Hour and Season")
    ax1.legend(fontsize=9)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)], fontsize=8)

    # Panel 2: scatter solar vs BESS
    sample = df[df["Battery Storage (MW)"].abs() > 0.05].sample(min(5000, len(df)), random_state=42)
    sc = ax2.scatter(sample["Solar (MW)"], sample["Battery Storage (MW)"],
                     c=sample["hour"], cmap="viridis", s=5, alpha=0.5)
    plt.colorbar(sc, ax=ax2, label="Hour of Day")
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.set_xlabel("Solar Generation (MW)")
    ax2.set_ylabel("BESS MW (+ = discharge)")
    ax2.set_title("BESS Dispatch vs Solar Generation\n(coloured by hour)")

    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig5_bess_dispatch.png")
    plt.close(fig)
    print("\n  → Saved fig5_bess_dispatch.png")

    return hourly_bess


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS G — Security Event Impact on Production Mix
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_G(df):
    print(f"\n{DIVIDER}")
    print("ANALYSIS G — SECURITY EVENT IMPACT ON PRODUCTION MIX")
    print(DIVIDER)

    sources = ["Coal (MW)", "Gas (MW)", "Solar (MW)", "Wind (MW)",
               "Bio-Gas (MW)", "Battery Storage (MW)"]
    src_labels = ["Coal", "Gas", "Solar", "Wind", "Bio-Gas", "BESS"]

    # Baseline per calendar month (exclude ±7 days around events)
    def season_baseline(event_dates):
        months = list({int(d[5:7]) for d in event_dates})
        excl   = set()
        for d in event_dates:
            dt = pd.Timestamp(d)
            excl |= {str((dt + pd.Timedelta(days=i)).date())
                     for i in range(-7, 8)}
        mask = df["month"].isin(months) & ~df["Date"].isin(excl)
        return df[mask][sources + ["Current Demand (MW)"]].mean()

    rows_g = []
    for name, dates_str, dates in SECURITY_EVENTS:
        ev_df  = df[df["Date"].isin(dates)]
        ev_avg = ev_df[sources + ["Current Demand (MW)"]].mean()
        bl_avg = season_baseline(dates)
        row = {"event": name, "dates": dates_str}
        for src, lbl in zip(sources, src_labels):
            ev_v = ev_avg.get(src, 0); bl_v = bl_avg.get(src, 1)
            row[lbl]      = ev_v
            row[lbl+"_bl"]  = bl_v
            row[lbl+"_d"]   = ev_v - bl_v
            row[lbl+"_pct"] = (ev_v - bl_v) / bl_v * 100 if abs(bl_v) > 0.1 else 0
        # demand delta
        ev_dem = ev_avg.get("Current Demand (MW)", float("nan"))
        bl_dem = bl_avg.get("Current Demand (MW)", float("nan"))
        row["dem_d"]   = ev_dem - bl_dem if not (np.isnan(ev_dem) or np.isnan(bl_dem)) else float("nan")
        row["dem_pct"] = (ev_dem - bl_dem) / bl_dem * 100 if (not np.isnan(bl_dem) and bl_dem > 0) else float("nan")
        rows_g.append(row)

    print("\nTABLE G1 — Production mix deviation during security events (MW vs seasonal baseline)")
    print(f"{'─'*90}")
    hdr = f"{'Event':<22} " + "".join(f"{lbl:>10}" for lbl in src_labels) + "  Demand"
    print(hdr); print("─"*90)
    for r in rows_g:
        line = f"{r['event']:<22} "
        for lbl in src_labels:
            d = r[lbl+"_d"]
            line += f"  {d:>+8.0f}"
        dd = r.get("dem_d", float("nan"))
        line += f"  {dd:>+8.0f}" if not np.isnan(dd) else "       —"
        print(line)

    src_labels_pct = [l for l in src_labels if l != "BESS"]  # BESS baseline ≈0, % meaningless
    print("\nTABLE G2 — Production mix deviation (% vs seasonal baseline)")
    print(f"{'─'*80}")
    hdr2 = f"{'Event':<22} " + "".join(f"{lbl:>10}" for lbl in src_labels_pct) + "  Demand%"
    print(hdr2); print("─"*80)
    for r in rows_g:
        line = f"{r['event']:<22} "
        for lbl in src_labels_pct:
            p = r[lbl+"_pct"]
            line += f"  {p:>+8.1f}%"
        dp = r.get("dem_pct", float("nan"))
        line += f"  {dp:>+7.1f}%" if not np.isnan(dp) else "       —"
        print(line)

    # ── FIGURE G ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    n_events = len(rows_g)
    n_src    = len(src_labels)
    x        = np.arange(n_events)
    width    = 0.13
    colors   = ["#2c3e50","#e67e22","#f1c40f","#1abc9c","#9b59b6","#e74c3c"]

    for i, (lbl, col) in enumerate(zip(src_labels, colors)):
        pcts = [r[lbl+"_pct"] for r in rows_g]
        off  = (i - n_src/2 + 0.5) * width
        bars = ax.bar(x + off, pcts, width, label=lbl, color=col,
                      edgecolor="black", linewidth=0.4)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["event"] for r in rows_g], fontsize=10, rotation=10)
    ax.set_ylabel("Change vs Seasonal Baseline (%)")
    ax.set_title("Production Mix Change During Security Events\n(deviation from same-month seasonal baseline)")
    ax.legend(fontsize=9, ncol=3)
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig6_event_production_mix.png")
    plt.close(fig)
    print("\n  → Saved fig6_event_production_mix.png")

    return rows_g


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS H — Forecast Error by Solar Phase
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_H(df):
    print(f"\n{DIVIDER}")
    print("ANALYSIS H — FORECAST ERROR BY SOLAR PHASE")
    print(DIVIDER)

    # Define solar phases (daytime 08–20)
    def solar_phase(h):
        if  8 <= h <  9: return "Pre-ramp\n(08–09)"
        if  9 <= h < 13: return "Solar Peak\n(09–13)"
        if 13 <= h < 17: return "Solar Decline\n(13–17)"
        if 17 <= h < 20: return "Evening Ramp\n(17–20)"
        return None

    win = df[(df["hour"] >= HOUR_FROM) & (df["hour"] < HOUR_TO)].copy()
    win["phase"] = win["hour"].apply(solar_phase)
    win = win[win["phase"].notna()]

    PHASE_ORDER = ["Pre-ramp\n(08–09)", "Solar Peak\n(09–13)",
                   "Solar Decline\n(13–17)", "Evening Ramp\n(17–20)"]

    # MAPE by phase and year
    grp = win.groupby(["phase","year"]).agg(
        da=("DA MAPE","mean"), sf=("SF MAPE","mean"),
    ).reset_index()

    print("\nTABLE H1 — DA MAPE (%) by solar phase and year")
    print(f"{'─'*55}")
    piv = grp.pivot_table(index="year", columns="phase", values="da")
    piv.columns.name = None
    print(f"{'Year':<6}", end="")
    for p in PHASE_ORDER:
        lbl = p.replace("\n"," ")
        print(f"  {lbl:>20}", end="")
    print(); print("─"*55)
    for yr, row in piv.iterrows():
        print(f"{yr:<6}", end="")
        for p in PHASE_ORDER:
            v = row.get(p, np.nan)
            print(f"  {v:>20.2f}" if not np.isnan(v) else f"  {'—':>20}", end="")
        print()

    print("\nTABLE H2 — SF MAPE (%) by solar phase and year")
    print(f"{'─'*55}")
    piv_sf = grp.pivot_table(index="year", columns="phase", values="sf")
    piv_sf.columns.name = None
    print(f"{'Year':<6}", end="")
    for p in PHASE_ORDER:
        lbl = p.replace("\n"," ")
        print(f"  {lbl:>20}", end="")
    print(); print("─"*55)
    for yr, row in piv_sf.iterrows():
        print(f"{yr:<6}", end="")
        for p in PHASE_ORDER:
            v = row.get(p, np.nan)
            print(f"  {v:>20.2f}" if not np.isnan(v) else f"  {'—':>20}", end="")
        print()

    # ── FIGURE H ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, metric, title in [
            (axes[0], "da", "DA Forecast Error by Solar Phase"),
            (axes[1], "sf", "SF Forecast Error by Solar Phase")]:
        sub   = grp.pivot_table(index="phase", columns="year", values=metric)
        sub   = sub.reindex(PHASE_ORDER)
        years = sorted(grp["year"].unique())
        x_    = np.arange(len(PHASE_ORDER))
        w_    = 0.25
        for j, yr in enumerate(years):
            vals_ = [sub.loc[p, yr] if p in sub.index and yr in sub.columns
                     else np.nan for p in PHASE_ORDER]
            off_  = (j - len(years)/2 + 0.5) * w_
            ax.bar(x_ + off_, vals_, w_, color=YEAR_COLORS.get(yr,"gray"),
                   label=str(yr), edgecolor="black", linewidth=0.4)
        ax.set_xticks(x_)
        ax.set_xticklabels([p.replace("\n"," ") for p in PHASE_ORDER], fontsize=9, rotation=15)
        ax.set_ylabel("MAPE (%)"); ax.set_title(title)
        ax.legend(title="Year", fontsize=9)

    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig7_solar_phase_mape.png")
    plt.close(fig)
    print("\n  → Saved fig7_solar_phase_mape.png")

    return grp


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    df_raw = load_all()
    bad    = get_bad_days(df_raw)
    df     = df_raw[~df_raw["Date"].isin(bad)].copy()
    print(f"  After quality filter: {df['Date'].nunique()} clean days "
          f"({len(bad)} removed)")

    # Run all analyses
    bl_da, bl_sf, ev_rows, daily_da, daily_sf = analysis_A(df)
    mon_ren, ann_ren   = analysis_B(df)
    ramp_df            = analysis_C(df)
    mon_co2            = analysis_D(df)
    grp_ren            = analysis_E(df)
    hourly_bess        = analysis_F(df)
    rows_g             = analysis_G(df)
    grp_phase          = analysis_H(df)

    print(f"\n{DIVIDER}")
    print("ALL ANALYSES COMPLETE")
    print(f"  Figures saved: fig1–fig9 in {DATA_DIR}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
