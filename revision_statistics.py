"""Revision statistics (Scientific Reports, reviewer 2): event tests (all four events, day and
event level), block bootstrap of the baseline, renewable-share checks, CO2/renewable averaging
conventions and coverage, and the Battery Storage records."""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import full_grid_analysis as fga

df_raw = fga.load_all()
bad = fga.get_bad_days(df_raw)
df = df_raw[~df_raw.Date.isin(bad)].copy()
win = df[(df.hour >= 8) & (df.hour < 20)]
dda = win.groupby("Date")["DA MAPE"].mean().dropna()
dsf = win.groupby("Date")["SF MAPE"].mean().reindex(dda.index)
dates = pd.to_datetime(dda.index)
EV = {name: [d for d in ds if d in dda.index] for name, _, ds in fga.SECURITY_EVENTS}
ev_days = [d for v in EV.values() for d in v]
print("event days used:", EV)
non = dda.drop(ev_days)

# ---- Q4a: day-level, all event days vs non-event days
u, p = stats.mannwhitneyu(dda[ev_days], non, alternative="greater")
print(f"Q4 day-level MW (all {len(ev_days)} event days vs {len(non)}): U={u:.0f} p={p:.4f}")
# ---- Q4b: event-level permutation test (multi-day event = one unit)
obs = np.mean([dda[v].mean() for v in EV.values()])
lens = [len(v) for v in EV.values()]
idx = list(non.index); pos = {d: i for i, d in enumerate(idx)}
rng = np.random.default_rng(42); N = 20000; null = np.empty(N)
for b in range(N):
    used = set(); vals = []
    for L in lens:
        while True:
            s = rng.integers(0, len(idx) - L + 1)
            blk = idx[s:s + L]
            if not used.intersection(blk):
                used.update(blk); vals.append(non[blk].mean()); break
    null[b] = np.mean(vals)
p_perm = (np.sum(null >= obs) + 1) / (N + 1)
print(f"Q4 event-level permutation: observed mean of 4 event means = {obs:.2f}%, null mean {null.mean():.2f}%, "
      f"95th pct {np.percentile(null,95):.2f}%, p = {p_perm:.4f}")
for name, v in EV.items():
    m = dda[v].mean()
    print(f"   {name:16s} DA {m:.2f}%  percentile among ordinary days: {100*(non < m).mean():.1f}  "
          f"SF {dsf[v].mean():.2f}% (SF pct {100*(dsf.drop(ev_days) < dsf[v].mean()).mean():.1f})")

# ---- Q5: baseline mean CI, iid vs moving-block bootstrap; ordinary-day distribution
x = dda.values
iid = [rng.choice(x, len(x)).mean() for _ in range(10000)]
B = 7; nb = int(np.ceil(len(x) / B))
blk = [np.concatenate([x[s:s + B] for s in rng.integers(0, len(x) - B + 1, nb)])[:len(x)].mean() for _ in range(10000)]
print(f"Q5 baseline DA mean {x.mean():.2f}%: iid CI [{np.percentile(iid,2.5):.2f}, {np.percentile(iid,97.5):.2f}]  "
      f"7-day block CI [{np.percentile(blk,2.5):.2f}, {np.percentile(blk,97.5):.2f}]")
print(f"   ordinary-day DA percentiles: 50th {np.percentile(non,50):.2f}, 90th {np.percentile(non,90):.2f}, 95th {np.percentile(non,95):.2f}")
print(f"   SF baseline {dsf.mean():.2f}; ordinary-day SF 90th {np.percentile(dsf.drop(ev_days).dropna(),90):.2f}, 95th {np.percentile(dsf.drop(ev_days).dropna(),95):.2f}")

# ---- Q6: renewable association checks
e = df[df["DA MAPE"].notna() & df["Renewable Share (%)"].notna()].copy()
e["rb"] = pd.cut(e["Renewable Share (%)"], list(range(0, 65, 5)), right=False)
e["sf_abs"] = (e["Updated Demand Forecast (MW)"] - e["Current Demand (MW)"]).abs()
e["da_abs"] = (e["Day-Ahead Demand Forecast (MW)"] - e["Current Demand (MW)"]).abs()
g = e.groupby("rb", observed=True).agg(n=("SF MAPE", "size"), sf=("SF MAPE", "mean"), da=("DA MAPE", "mean"),
                                       sf_mw=("sf_abs", "mean"), da_mw=("da_abs", "mean"), dem=("Current Demand (MW)", "mean"))
print("Q6 all hours by bin:\n", g.round(2).to_string())
mid = e[(e.hour >= 10) & (e.hour < 14)].groupby("rb", observed=True).agg(n=("SF MAPE", "size"), sf=("SF MAPE", "mean"), sf_mw=("sf_abs", "mean"), da=("DA MAPE","mean"))
print("Q6 10:00-14:00 only:\n", mid.round(2).to_string())
# within hour-of-day and month: rank correlation of SF APE with renewable share, averaged over strata
rs = []
for (h, mo), gg in e[(e.hour >= 8) & (e.hour < 18) & e["SF MAPE"].notna()].groupby(["hour", "month"]):
    if len(gg) > 200 and gg["Renewable Share (%)"].std() > 1:
        rs.append(stats.spearmanr(gg["Renewable Share (%)"], gg["SF MAPE"])[0])
print(f"Q6 within (hour, month) strata 08-18h: median Spearman rho(ren share, SF APE) = {np.median(rs):.3f}, "
      f"positive in {np.mean(np.array(rs) > 0)*100:.0f}% of {len(rs)} strata")
ex = e[~((e.year == 2025) & e.month.isin([10, 11]))]
print("Q6 excl Oct-Nov 2025 SF by bin:", ex.groupby("rb", observed=True)["SF MAPE"].mean().round(2).tolist())

# ---- Q13: averaging conventions + coverage
co = df[df["Emission Ratio (tCO2/MWh)"].between(0.05, 1.5)]
ta = co.groupby("year")["Emission Ratio (tCO2/MWh)"].mean()
ew = co.groupby("year").apply(lambda g: g["Total Emissions (mTCO2/h)"].sum() / g["Current Demand (MW)"].sum())
ta_jd = co[co.month >= 6].groupby("year")["Emission Ratio (tCO2/MWh)"].mean()
ew_jd = co[co.month >= 6].groupby("year").apply(lambda g: g["Total Emissions (mTCO2/h)"].sum() / g["Current Demand (MW)"].sum())
print("Q13 CO2 time-avg:", ta.round(3).to_dict(), " energy-weighted:", ew.round(3).to_dict())
print("    Jun-Dec only time-avg:", ta_jd.round(3).to_dict(), " energy-weighted:", ew_jd.round(3).to_dict())
co_raw = df_raw[df_raw["Emission Ratio (tCO2/MWh)"].between(0.05, 1.5)]
print("    time-avg without frozen-day filter:", co_raw.groupby("year")["Emission Ratio (tCO2/MWh)"].mean().round(3).to_dict())
print("    CO2 coverage days per year:", co.groupby("year").Date.nunique().to_dict(), " months:", co.groupby("year").month.min().to_dict())
r = df[df["Renewable Share (%)"].notna()]
print("Q13 ren share time-avg:", r.groupby("year")["Renewable Share (%)"].mean().round(1).to_dict(),
      " energy-weighted:", r.groupby("year").apply(lambda g: 100 * g["Renewables Total (MW)"].sum() / g["Actual Demand (MW)"].sum()).round(1).to_dict())
print("    demand/production coverage days per year:", df.groupby("year").Date.nunique().to_dict())
# conditional intensity vs ren share endpoints (Fig 15)
co2b = co.copy(); co2b["rb"] = pd.cut(co2b["Renewable Share (%)"], list(range(0, 65, 5)), right=False)
print("Q12 intensity by bin:", co2b.groupby("rb", observed=True)["Emission Ratio (tCO2/MWh)"].mean().round(3).tolist())

# ---- Q11: BESS records
b = df["Battery Storage (MW)"]
for y in (2023, 2024, 2025):
    s = df[df.year == y]["Battery Storage (MW)"]
    nz = s[s.notna() & (s != 0)]
    print(f"Q11 BESS {y}: non-zero records {len(nz):,}  (days {df[(df.year==y) & b.notna() & (b!=0)].Date.nunique()}), "
          f"min {nz.min() if len(nz) else 0:.1f} max {nz.max() if len(nz) else 0:.1f} MW, mean |x| {nz.abs().mean() if len(nz) else 0:.2f}")
print("   first date with sustained non-zero (>=50 records/day):",
      df[b.notna() & (b != 0)].groupby("Date").size().loc[lambda s: s >= 50].index.min())
