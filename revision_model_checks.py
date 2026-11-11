"""Revision model checks (Scientific Reports, reviewer 2): hyperparameter check using validation
within 2024 only, and demand-lags-only / +calendar / full feature comparison per horizon."""
import sys, warnings, itertools
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import forecast_ml_v2 as m

df_raw = m.load_all_features()
df = df_raw[~df_raw["Date"].isin(m.get_bad_days(df_raw))].copy()
tr_all = m.add_derived_train_features(df[df.year == 2024].copy(), df[df.year == 2024].copy())
t23 = m.add_derived_train_features(df[df.year == 2023].copy(), tr_all)
t25 = m.add_derived_train_features(df[df.year == 2025].copy(), tr_all)
f = lambda d: d[d["Current Demand (MW)"] > 1000].copy()
tr_all, t23, t25 = f(tr_all), f(t23), f(t25)
H = {"5": m.SF_W_FEATS, "15": m.SF_W_15M_FEATS, "30": m.SF_W_30M_FEATS, "60": m.SF_W1H_FEATS}

def mp(mod, d, feats):
    return m.mape_mean(m.mape_arr(d["Current Demand (MW)"].values, m.predict(mod, d, feats)))

# ---- Q8: train Jan-Oct 2024, validate Nov-Dec 2024 (no 2023/2025 data used)
fit_part = tr_all[tr_all.month <= 10]; val = tr_all[tr_all.month >= 11]
grid = list(itertools.product([300, 600, 1000], [5, 7, 9], [0.03, 0.1]))
print(f"Q8 validation: fit {len(fit_part):,} rows (Jan-Oct 2024), validate {len(val):,} (Nov-Dec 2024), {len(grid)} configs")
for h, feats in H.items():
    res = []
    for n, dpt, lr in grid:
        mod, _ = m.fit_lgb(fit_part, fit_part["Current Demand (MW)"], feats, n=n, depth=dpt, lr=lr)
        res.append(((n, dpt, lr), mp(mod, val, feats)))
    res.sort(key=lambda r: r[1])
    used = dict(res)[(600, 7, 0.03)]
    rank = [r[0] for r in res].index((600, 7, 0.03)) + 1
    best_cfg, best_val = res[0]
    # test with validation-selected config, trained on full 2024
    mod, _ = m.fit_lgb(tr_all, tr_all["Current Demand (MW)"], feats, n=best_cfg[0], depth=best_cfg[1], lr=best_cfg[2])
    a = m.mape_arr(t23["Current Demand (MW)"].values, m.predict(mod, t23, feats))
    b = m.mape_arr(t25["Current Demand (MW)"].values, m.predict(mod, t25, feats))
    print(f"  {h:>2} min: used (600,7,0.03) val {used:.3f} (rank {rank}/{len(grid)}); best {best_cfg} val {best_val:.3f}; "
          f"worst val {res[-1][1]:.3f}; test with val-selected config: comb {m.mape_mean(np.concatenate([a,b])):.3f}")

# ---- Q9: demand-lags-only ablation
LAGS = {"5": ["lag_5min", "lag_15min", "lag_30min", "lag_24h", "lag_48h", "delta_5min", "delta_30min"],
        "15": ["lag_15min", "lag_30min", "lag_1h", "lag_2h", "lag_6h", "lag_24h", "lag_48h"],
        "30": ["lag_30min", "lag_1h", "lag_2h", "lag_6h", "lag_24h", "lag_48h"],
        "60": ["lag_1h", "lag_2h", "lag_6h", "lag_24h", "lag_48h"]}
print("Q9 ablation (combined 2023+2025 MAPE %):")
for h in H:
    out = {}
    for name, feats in [("lags only", LAGS[h]), ("lags + calendar", LAGS[h] + m._CAL), ("full LGB-SF-W", H[h])]:
        mod, _ = m.fit_lgb(tr_all, tr_all["Current Demand (MW)"], feats)
        a = m.mape_arr(t23["Current Demand (MW)"].values, m.predict(mod, t23, feats))
        b = m.mape_arr(t25["Current Demand (MW)"].values, m.predict(mod, t25, feats))
        out[name] = (m.mape_mean(a), m.mape_mean(b), m.mape_mean(np.concatenate([a, b])))
    print(f"  {h:>2} min: " + " | ".join(f"{k}: {v[0]:.2f}/{v[1]:.2f}/{v[2]:.3f}" for k, v in out.items()))
