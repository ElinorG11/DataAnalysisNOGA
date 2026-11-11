"""
Forecast Error Analysis — NOGA-ISO Demand Data
===============================================
Six tables, all MAPE-normalized (%), hours 08:00–20:00:

  TABLE 1 — Events: daily MAPE (DA + SF)
  TABLE 2 — Top N days by daily DA MAPE
  TABLE 3 — Top N days by daily SF MAPE
  TABLE 4 — Events: worst 30-min window (DA + SF)
  TABLE 5 — Top N thirty-min windows by DA MAPE
  TABLE 6 — Top N thirty-min windows by SF MAPE

Customize at the top:  EVENTS, HOUR_FROM, HOUR_TO, TOP_N, CSV_PATH
"""

from pathlib import Path
from collections import defaultdict
import csv

# ── Configuration ─────────────────────────────────────────────────────────────

CSV_PATH  = Path(__file__).parent / "noga_demand_5min.csv"
HOUR_FROM = 8    # inclusive
HOUR_TO   = 20   # exclusive
TOP_N     = 10   # rows in tables 2, 3, 5, 6
BIN_MIN   = 30   # bin width in minutes (6 samples per bin at 5-min resolution)

EVENTS = [
    ("Jun 2 2023 — Load shedding",     "Weather",  ["2023-06-02"]),
    ("Aug 13 2023 — Heatwave record",  "Weather",  ["2023-08-13"]),
    ("Oct 8 2023 — Hamas day-after",   "Security", ["2023-10-08"]),
    ("Apr 13-14 2024 — True Promise",  "Security", ["2024-04-13", "2024-04-14"]),
    ("Oct 1 2024 — True Promise II",   "Security", ["2024-10-01"]),
    ("Jun 13-16 2025 — Rising Lion",   "Security", ["2025-06-13", "2025-06-14",
                                                     "2025-06-15", "2025-06-16"]),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def hour_of(t): return int(t[:2])

def bin_label(t, bin_min):
    h, m = int(t[:2]), int(t[3:5])
    return f"{h:02d}:{(m // bin_min) * bin_min:02d}"

def mape(v): return sum(v) / len(v) if v else float("nan")

def rank_of(val, sorted_desc):
    return sum(1 for v in sorted_desc if v >= val)

def fmt(val, width=9):
    return f"{val:{width}.2f}" if val == val else f"{'N/A':>{width}}"

def find_bad_days(rows_by_date, hour_from, hour_to, max_frozen=0.10):
    bad = set()
    for date, rows in rows_by_date.items():
        day = [r for r in rows if hour_from <= hour_of(r["Time"]) < hour_to]
        if not day: continue
        frozen = sum(
            1 for i in range(1, len(day))
            if day[i]["Current Demand (MW)"] == day[i-1]["Current Demand (MW)"]
            and float(day[i]["Current Demand (MW)"] or 0) > 0
        )
        if frozen / len(day) > max_frozen:
            bad.add(date)
    return bad


def load_data(path, hour_from, hour_to, bad_days, bin_min):
    """
    Returns:
      daily_da, daily_upd  — {date: avg_mape}
      bin_da,   bin_upd    — {(date, slot): avg_mape}
    """
    da_day  = defaultdict(list)
    upd_day = defaultdict(list)
    da_bin  = defaultdict(list)
    upd_bin = defaultdict(list)

    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            date = row["Date"]
            if date in bad_days: continue
            h = hour_of(row["Time"])
            if not (hour_from <= h < hour_to): continue
            actual = float(row["Current Demand (MW)"] or 0)
            if actual < 1000: continue
            slot = bin_label(row["Time"], bin_min)
            da_v  = row.get("Day-Ahead Demand Forecast (MW)", "")
            upd_v = row.get("Updated Demand Forecast (MW)", "")
            if da_v and da_v != "0":
                v = float(da_v)
                if 2000 <= v <= 25000:
                    pct = abs(v - actual) / actual * 100
                    da_day[date].append(pct)
                    da_bin[(date, slot)].append(pct)
            if upd_v and upd_v != "0":
                v = float(upd_v)
                if 2000 <= v <= 25000:
                    pct = abs(v - actual) / actual * 100
                    upd_day[date].append(pct)
                    upd_bin[(date, slot)].append(pct)

    min_day_samples = 50
    min_bin_samples = bin_min // 5 // 2   # at least half the expected samples

    daily_da  = {d: mape(v) for d, v in da_day.items()  if len(v) >= min_day_samples}
    daily_upd = {d: mape(v) for d, v in upd_day.items() if len(v) >= min_day_samples}
    bin_da    = {k: mape(v) for k, v in da_bin.items()  if len(v) >= min_bin_samples}
    bin_upd   = {k: mape(v) for k, v in upd_bin.items() if len(v) >= min_bin_samples}

    return daily_da, daily_upd, bin_da, bin_upd


# ── Main ──────────────────────────────────────────────────────────────────────

def run(csv_path=CSV_PATH, hour_from=HOUR_FROM, hour_to=HOUR_TO,
        events=EVENTS, top_n=TOP_N, bin_min=BIN_MIN):

    print(f"\nLoading : {csv_path}")

    rows_by_date = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows_by_date[row["Date"]].append(row)

    bad_days = find_bad_days(rows_by_date, hour_from, hour_to)
    daily_da, daily_upd, bin_da, bin_upd = load_data(
        csv_path, hour_from, hour_to, bad_days, bin_min)

    # Sorted lists for ranking
    s_day_da  = sorted(daily_da.values(),  reverse=True)
    s_day_upd = sorted(daily_upd.values(), reverse=True)
    s_bin_da  = sorted(bin_da.values(),    reverse=True)
    s_bin_upd = sorted(bin_upd.values(),   reverse=True)

    n_days = len(daily_da)
    n_bins = len(bin_da)

    bl_day_da  = mape(s_day_da)
    bl_day_upd = mape(s_day_upd)
    bl_bin_da  = mape(s_bin_da)
    bl_bin_upd = mape(s_bin_upd)

    print(f"Hour window : {hour_from:02d}:00 – {hour_to:02d}:00")
    print(f"Clean days  : {n_days}  |  30-min bins: {n_bins}")
    print(f"Baseline MAPE (daily)  — DA: {bl_day_da:.2f}%  |  SF: {bl_day_upd:.2f}%")
    print(f"Baseline MAPE (30-min) — DA: {bl_bin_da:.2f}%  |  SF: {bl_bin_upd:.2f}%")

    # ── TABLE 1: Events daily ──────────────────────────────────────────────────
    _print_events_daily(events, daily_da, daily_upd, s_day_da, s_day_upd,
                        bl_day_da, bl_day_upd, n_days)

    # ── TABLE 2: Top N days by daily DA ──────────────────────────────────────
    _print_top_daily(daily_da, daily_upd, s_day_upd, n_days, top_n,
                     sort_by="DA", label="TABLE 2 — Top days by daily DA MAPE")

    # ── TABLE 3: Top N days by daily SF ──────────────────────────────────────
    _print_top_daily(daily_upd, daily_da, s_day_da, n_days, top_n,
                     sort_by="SF", label="TABLE 3 — Top days by daily SF MAPE")

    # ── TABLE 4: Events worst 30-min window ───────────────────────────────────
    _print_events_30min(events, bin_da, bin_upd, s_bin_da, s_bin_upd,
                        bl_bin_da, bl_bin_upd, n_bins)

    # ── TABLE 5: Top N 30-min windows by DA ──────────────────────────────────
    _print_top_bins(bin_da, bin_upd, s_bin_upd, n_bins, top_n,
                    sort_by="DA", label="TABLE 5 — Top 30-min windows by DA MAPE")

    # ── TABLE 6: Top N 30-min windows by SF ──────────────────────────────────
    _print_top_bins(bin_upd, bin_da, s_bin_da, n_bins, top_n,
                    sort_by="SF", label="TABLE 6 — Top 30-min windows by SF MAPE")

    print("\nRank 1 = highest (worst) error  |  Positive Δ = worse than baseline")


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_events_daily(events, daily_da, daily_upd, s_da, s_upd,
                        bl_da, bl_upd, n_days):
    print(f"\n{'═'*100}")
    print("TABLE 1 — Events: daily MAPE (08:00–20:00)")
    print(f"{'═'*100}")
    h = (f"{'Event':<36} {'Type':<9} {'DA MAPE%':>9} {'SF MAPE%':>9} "
         f"{'ΔDA%':>8} {'ΔSF%':>8} {'DA Rank':>10} {'SF Rank':>10}")
    print(h); print("─"*len(h))
    for name, etype, dates in events:
        da_e  = [e for d in dates if d in daily_da  for e in [daily_da[d]]]
        upd_e = [e for d in dates if d in daily_upd for e in [daily_upd[d]]]
        avg_da  = mape(da_e);   avg_upd = mape(upd_e)
        rk_da   = rank_of(avg_da, s_da); rk_upd = rank_of(avg_upd, s_upd)
        print(f"{name:<36} {etype:<9} {fmt(avg_da)} {fmt(avg_upd)} "
              f"{avg_da-bl_da:>+8.2f} {avg_upd-bl_upd:>+8.2f} "
              f"{rk_da:>8}/{n_days}  {rk_upd:>8}/{n_days}")


def _print_top_daily(primary, secondary, s_sec, n_days, top_n, sort_by, label):
    print(f"\n{'═'*72}")
    print(label)
    print(f"{'═'*72}")
    a, b = ("DA MAPE%", "SF MAPE%") if sort_by == "DA" else ("SF MAPE%", "DA MAPE%")
    h = f"{'Rank':<5} {'Date':<13} {a:>9} {b:>9} {'Primary Rank':>13} {'Secondary Rank':>15}"
    print(h); print("─"*len(h))
    for rank, (date, pm) in enumerate(
            sorted(primary.items(), key=lambda x: -x[1])[:top_n], 1):
        sm   = secondary.get(date, float("nan"))
        rk_s = rank_of(sm, s_sec) if sm == sm else "N/A"
        print(f"{rank:<5} {date:<13} {fmt(pm)} {fmt(sm)} "
              f"{rank:>11}/{n_days}  {rk_s:>13}/{n_days}")


def _print_events_30min(events, bin_da, bin_upd, s_da, s_upd,
                        bl_da, bl_upd, n_bins):
    print(f"\n{'═'*114}")
    print("TABLE 4 — Events: worst 30-min window")
    print(f"{'═'*114}")
    h = (f"{'Event':<36} {'Type':<9}"
         f" {'DA Time':>10} {'DA MAPE%':>9} {'ΔDA%':>8} {'DA Rank':>12}"
         f"  {'SF Time':>10} {'SF MAPE%':>9} {'ΔSF%':>8} {'SF Rank':>12}")
    print(h); print("─"*len(h))
    for name, etype, dates in events:
        ev_da  = {(d,s): bin_da[d,s]  for d in dates
                  for s in {k[1] for k in bin_da  if k[0]==d} if (d,s) in bin_da}
        ev_upd = {(d,s): bin_upd[d,s] for d in dates
                  for s in {k[1] for k in bin_upd if k[0]==d} if (d,s) in bin_upd}
        if not ev_da or not ev_upd: continue
        wdk, wuk = max(ev_da, key=ev_da.get), max(ev_upd, key=ev_upd.get)
        wd, wu = ev_da[wdk], ev_upd[wuk]
        da_lbl  = f"{wdk[0][5:]} {wdk[1]}"
        upd_lbl = f"{wuk[0][5:]} {wuk[1]}"
        rda = rank_of(wd, s_da); rup = rank_of(wu, s_upd)
        print(f"{name:<36} {etype:<9}"
              f" {da_lbl:>10} {fmt(wd)} {wd-bl_da:>+8.2f} {rda:>10}/{n_bins}"
              f"  {upd_lbl:>10} {fmt(wu)} {wu-bl_upd:>+8.2f} {rup:>10}/{n_bins}")


def _print_top_bins(primary, secondary, s_sec, n_bins, top_n, sort_by, label):
    print(f"\n{'═'*80}")
    print(label)
    print(f"{'═'*80}")
    a, b = ("DA MAPE%", "SF MAPE%") if sort_by == "DA" else ("SF MAPE%", "DA MAPE%")
    h = (f"{'Rank':<5} {'Date':<12} {'Slot':>6} {a:>9} {b:>10} "
         f"{'Primary Rank':>13} {'Secondary Rank':>15}")
    print(h); print("─"*len(h))
    for rank, ((date, slot), pm) in enumerate(
            sorted(primary.items(), key=lambda x: -x[1])[:top_n], 1):
        sm   = secondary.get((date, slot), float("nan"))
        rk_s = rank_of(sm, s_sec) if sm == sm else "N/A"
        print(f"{rank:<5} {date:<12} {slot:>6} {fmt(pm)} {fmt(sm, 10)} "
              f"{rank:>11}/{n_bins}  {rk_s:>13}/{n_bins}")


if __name__ == "__main__":
    run()
