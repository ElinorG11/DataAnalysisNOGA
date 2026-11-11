#!/usr/bin/env python3
"""
forecast_dl.py — Deep-Learning Demand Forecasting
==================================================
Models
  LSTM-DA    : 2-layer LSTM, 48-h context, day-ahead multi-step (30-min res)
  TRANS-DA   : Vanilla Transformer encoder, same I/O as LSTM-DA
  LSTM-SF    : 2-layer LSTM, 6-h context (newest obs t-1h), 1h-ahead (5-min res)
  TRANS-SF   : Vanilla Transformer encoder, same I/O as LSTM-SF

Note on Informer vs vanilla Transformer:
  Informer's ProbSparse attention gives O(L log L) vs O(L²).
  For L=96 (DA input), vanilla needs ~9 k attention operations;
  Informer ~672.  At this scale the difference is <1 ms on CPU and
  accuracy is equivalent or worse (sparser attention can miss local
  patterns).  We use vanilla Transformer for all DL models.

Train: Jan–Oct 2024  |  Val (early stop): Nov–Dec 2024
Test:  2023 and 2025 (clean days only)
Metric: daytime MAPE  08:00–20:00

DA comparison vs: NOGA-DA 3.24%  |  LGB-DA-W 3.96%
SF comparison vs: NOGA-SF 1.48%  |  LGB-SF    1.80%
"""

import sys, math, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR    = Path(__file__).parent
WEATHER_CSV = DATA_DIR / "weather_tel_aviv_hourly.csv"

HOUR_FROM, HOUR_TO = 8, 20
FROZEN_THR = 0.10

# ── Previous results (reference lines on plots) ───────────────────────────────
PREV = {
    "NOGA-DA":  {"2023": 3.22, "2025": 3.25, "Comb.": 3.24},
    "LGB-DA-W": {"2023": 4.69, "2025": 3.49, "Comb.": 3.96},
    "NOGA-SF":  {"2023": 1.03, "2025": 1.77, "Comb.": 1.48},
    "LGB-SF":   {"2023": 1.87, "2025": 1.74, "Comb.": 1.80},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

# Training hyper-parameters
DA_EPOCHS     = 100
SF_EPOCHS     = 120
BATCH_DA      = 64
BATCH_SF      = 256         # smaller batch → better gradient estimates for SF
LR_DA         = 3e-4
LR_SF         = 5e-5        # SF needs gentler LR; sequence → scalar is very noisy
PATIENCE      = 20          # early-stopping patience (epochs)

# Sequence lengths (30-min for DA, 5-min for SF)
DA_IN_STEPS   = 96          # 48 h at 30-min
DA_OUT_STEPS  = 48          # 24 h at 30-min
SF_IN_STEPS   = 72          # 6 h at 5-min (newest obs = t-1h = shift 12)
SF_MIN_LAG    = 12          # minimum lag in 5-min steps = 1 h


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_5min():
    print("  Loading 5-min demand + production mix …")
    dem = pd.read_csv(DATA_DIR/"noga_demand_5min.csv", encoding="utf-8-sig")
    dem.rename(columns={"Interval (1-288)": "Interval"}, inplace=True)
    dem["datetime"] = pd.to_datetime(dem["Date"] + " " + dem["Time"])
    for c in ["Current Demand (MW)", "Updated Demand Forecast (MW)",
              "Day-Ahead Demand Forecast (MW)"]:
        dem[c] = pd.to_numeric(dem[c], errors="coerce")

    prod = pd.read_csv(DATA_DIR/"noga_production_mix_5min.csv", encoding="utf-8-sig")
    prod.rename(columns={"Interval (1-288)": "Interval"}, inplace=True)
    prod["datetime"] = pd.to_datetime(prod["Date"] + " " + prod["Time"])
    for c in ["PV (MW)", "Photovoltaic (MW)", "Thermo-Solar (MW)"]:
        if c in prod.columns:
            prod[c] = pd.to_numeric(prod[c], errors="coerce").fillna(0)
    prod["Solar (MW)"] = (prod.get("PV (MW)", 0)
                          + prod.get("Photovoltaic (MW)", 0)
                          + prod.get("Thermo-Solar (MW)", 0))

    df = dem.merge(prod[["datetime", "Solar (MW)"]], on="datetime", how="left")

    # Attach hourly weather (forward-filled to 5-min)
    if WEATHER_CSV.exists():
        wh = pd.read_csv(WEATHER_CSV, parse_dates=["datetime"])
        wh = wh.set_index("datetime").sort_index()
        df = df.set_index("datetime").sort_index()
        df = df.join(wh[["temp_2m", "shortwave_rad"]], how="left")
        # forward-fill then back-fill so no leading/trailing NaN survives
        df[["temp_2m", "shortwave_rad"]] = (
            df[["temp_2m", "shortwave_rad"]].ffill().bfill().fillna(0))
        df = df.reset_index()
    else:
        df["temp_2m"] = 20.0
        df["shortwave_rad"] = 0.0

    df["hour"]    = df["datetime"].dt.hour
    df["minute"]  = df["datetime"].dt.minute
    df["month"]   = df["datetime"].dt.month
    df["year"]    = df["datetime"].dt.year
    df["dow"]     = df["datetime"].dt.dayofweek
    df["is_offday"] = df["dow"].isin({4, 5}).astype(float)

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def bad_days(df):
    bad = set()
    day = df[(df["hour"] >= HOUR_FROM) & (df["hour"] < HOUR_TO)]
    for date, g in day.groupby("Date"):
        v = g["Current Demand (MW)"].values
        fr = sum(1 for i in range(1, len(v)) if v[i] == v[i-1] and v[i] > 0)
        if len(v) > 0 and fr / len(v) > FROZEN_THR:
            bad.add(date)
    return bad


def resample_30min(df_5min):
    """Aggregate 5-min data to 30-min for DA models."""
    df = df_5min.copy()
    df = df.set_index("datetime")
    agg = df[["Current Demand (MW)", "Solar (MW)", "temp_2m",
              "shortwave_rad", "is_offday"]].resample("30min").mean()
    agg["Date"]   = agg.index.strftime("%Y-%m-%d")
    agg["hour"]   = agg.index.hour
    agg["month"]  = agg.index.month
    agg["year"]   = agg.index.year
    agg["dow"]    = agg.index.dayofweek
    agg = agg.reset_index()
    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

class Scaler:
    """Z-score scaler fitted on training data."""
    def __init__(self):
        self.mean = {}
        self.std  = {}

    def fit(self, df, cols):
        for c in cols:
            v = df[c].dropna()
            self.mean[c] = float(v.mean())
            self.std[c]  = float(v.std()) or 1.0

    def transform(self, df, cols):
        out = df.copy()
        for c in cols:
            out[c] = (df[c] - self.mean[c]) / self.std[c]
        return out

    def inverse(self, arr, col):
        return arr * self.std[col] + self.mean[col]


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

def time_encode(hour, minute, month):
    """Return (sin_hour, cos_hour, sin_month, cos_month) for a timestamp."""
    frac = (hour * 60 + minute) / 1440.0
    return (math.sin(2 * math.pi * frac),
            math.cos(2 * math.pi * frac),
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12))


class DADataset(Dataset):
    """
    Day-ahead dataset at 30-min resolution.
    Each sample: (history_seq, context_vec) → target_seq
      history_seq : [DA_IN_STEPS,  n_seq_feats]  — past 48 h
      context_vec : [n_ctx_feats]                — target-day static context
      target_seq  : [DA_OUT_STEPS]               — normalised demand next 24 h
    """
    SEQ_FEATS = ["Current Demand (MW)", "Solar (MW)", "temp_2m",
                 "shortwave_rad", "is_offday"]
    CTX_FEATS_SIZE = 6   # sh, ch, sm, cm, temp_mean, ghi_mean

    def __init__(self, df_30, scaler, dates):
        self.scaler = scaler
        self.df = scaler.transform(
            df_30, ["Current Demand (MW)", "Solar (MW)", "temp_2m", "shortwave_rad"])
        self.dates = sorted(dates)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        target_date = self.dates[idx]
        df = self.df

        # Target day rows
        tgt = df[df["Date"] == target_date].sort_values("datetime")
        # History: the 48 h immediately before midnight of target_date
        cutoff = pd.Timestamp(target_date)
        hist = df[df["datetime"] < cutoff].tail(DA_IN_STEPS)

        if len(hist) < DA_IN_STEPS or len(tgt) < DA_OUT_STEPS:
            return self._zeros()

        seq = torch.tensor(
            hist[self.SEQ_FEATS].fillna(0).values[-DA_IN_STEPS:],
            dtype=torch.float32)
        tgt_demand = torch.tensor(
            tgt["Current Demand (MW)"].fillna(0).values[:DA_OUT_STEPS],
            dtype=torch.float32)

        # Context vector for the target day: time encoding + avg weather
        sh, ch, sm, cm = time_encode(0, 0, tgt["month"].iloc[0])
        ctx = torch.tensor([
            sh, ch, sm, cm,
            float(tgt["temp_2m"].fillna(0).mean()),
            float(tgt["shortwave_rad"].fillna(0).mean()),
        ], dtype=torch.float32)

        return seq, ctx, tgt_demand

    def _zeros(self):
        return (torch.zeros(DA_IN_STEPS, len(self.SEQ_FEATS)),
                torch.zeros(self.CTX_FEATS_SIZE),
                torch.zeros(DA_OUT_STEPS))


class SFDataset(Dataset):
    """
    Short-forecast dataset at 5-min resolution (1-h minimum lag).
    Input:  SF_IN_STEPS (72) steps ending at t - SF_MIN_LAG (t-1h)
    Target: demand at t
    """
    SEQ_FEATS = ["dem_n", "sol_n", "temp_n", "ghi_n",
                 "sin_h", "cos_h", "sin_m", "cos_m"]

    def __init__(self, arr, valid_mask):
        # arr: numpy [N, n_feats] normalised, valid_mask: bool [N]
        self.arr   = arr.astype(np.float32)
        self.valid = np.where(valid_mask)[0]
        # only indices with enough context
        lo = SF_IN_STEPS + SF_MIN_LAG - 1
        self.valid = self.valid[self.valid >= lo]

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        i = self.valid[idx]
        # input: steps [i - SF_MIN_LAG - SF_IN_STEPS + 1  ..  i - SF_MIN_LAG]
        start = i - SF_MIN_LAG - SF_IN_STEPS + 1
        end   = i - SF_MIN_LAG + 1          # exclusive
        seq   = torch.tensor(self.arr[start:end], dtype=torch.float32)
        target = torch.tensor(self.arr[i, 0], dtype=torch.float32)
        return seq, target


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    """2-layer LSTM encoder + MLP head."""
    def __init__(self, in_dim, hidden, n_layers, ctx_dim, out_dim, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + ctx_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, seq, ctx=None):
        _, (h, _) = self.lstm(seq)
        h_last = h[-1]
        inp = torch.cat([h_last, ctx], dim=1) if ctx is not None else h_last
        return self.head(inp)


class TransformerModel(nn.Module):
    """
    Vanilla Transformer encoder (multi-head self-attention) + mean pooling
    + MLP head.

    Note: For sequence lengths used here (DA: 96 steps, SF: 72 steps) vanilla
    attention (O(L²)) is negligibly different from Informer's ProbSparse
    (O(L log L)) in wall-clock time.  Informer becomes beneficial only for
    L > 512.
    """
    def __init__(self, in_dim, d_model, nhead, n_layers,
                 ctx_dim, out_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        # Learnable positional encoding
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, batch_first=True,
            norm_first=True)           # Pre-LN for stability
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm     = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model + ctx_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, seq, ctx=None):
        B, T, _ = seq.shape
        pos = torch.arange(T, device=seq.device).unsqueeze(0).expand(B, -1)
        x = self.input_proj(seq) + self.pos_emb(pos)
        x = self.encoder(x)
        x = self.norm(x).mean(dim=1)   # mean pooling across time
        inp = torch.cat([x, ctx], dim=1) if ctx is not None else x
        return self.head(inp)


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, criterion, task):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        opt.zero_grad()
        if task == "da":
            seq, ctx, tgt = [b.to(DEVICE) for b in batch]
            pred = model(seq, ctx)
        else:
            seq, tgt = [b.to(DEVICE) for b in batch]
            pred = model(seq, None)
        loss = criterion(pred, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(tgt)
        n     += len(tgt)
    return total / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, task):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        if task == "da":
            seq, ctx, tgt = [b.to(DEVICE) for b in batch]
            pred = model(seq, ctx)
        else:
            seq, tgt = [b.to(DEVICE) for b in batch]
            pred = model(seq, None)
        loss = criterion(pred, tgt)
        total += loss.item() * len(tgt)
        n     += len(tgt)
    return total / n


def fit(model, tr_loader, va_loader, epochs, lr, task, label):
    model.to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    criterion = nn.MSELoss()
    best_val, wait, best_state = float("inf"), 0, None

    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, tr_loader, opt, criterion, task)
        va_loss = eval_epoch(model, va_loader, criterion, task)
        sched.step()
        if math.isnan(va_loss) or math.isinf(va_loss):
            pass   # don't save, don't count as no-improvement
        elif va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
        if ep % 10 == 0 or ep == epochs:
            print(f"    {label}  ep {ep:3d}  tr={tr_loss:.6f}  "
                  f"va={va_loss:.6f}  best={best_val:.6f}")

    if best_state is None:          # no finite val loss ever — use final weights
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — DA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_da(df_5min, bad):
    print("\n── DA Task (30-min resolution) ──────────────────────────────────")
    df30_all  = resample_30min(df_5min)          # full series for history lookup
    df30_clean = df30_all[~df30_all["Date"].isin(bad)].copy()

    # Date sets — targets are clean days only; history uses df30_all
    fit_dates = sorted(df30_clean[
        (df30_clean["year"] == 2024) &
        (df30_clean["Date"] < "2024-11-01")]["Date"].unique())
    val_dates = sorted(df30_clean[
        (df30_clean["year"] == 2024) &
        (df30_clean["Date"] >= "2024-11-01")]["Date"].unique())
    te23      = sorted(df30_clean[df30_clean["year"] == 2023]["Date"].unique())
    te25      = sorted(df30_clean[df30_clean["year"] == 2025]["Date"].unique())

    scaler = Scaler()
    scaler.fit(df30_all[df30_all["Date"].isin(fit_dates)],
               ["Current Demand (MW)", "Solar (MW)", "temp_2m", "shortwave_rad"])

    # Pass the full (unfiltered) df30 so history queries never hit gaps
    ds_fit = DADataset(df30_all, scaler, fit_dates)
    ds_val = DADataset(df30_all, scaler, val_dates)
    ds_t23 = DADataset(df30_all, scaler, te23)
    ds_t25 = DADataset(df30_all, scaler, te25)

    ld_fit = DataLoader(ds_fit, batch_size=BATCH_DA, shuffle=True,  num_workers=0)
    ld_val = DataLoader(ds_val, batch_size=BATCH_DA, shuffle=False, num_workers=0)
    ld_t23 = DataLoader(ds_t23, batch_size=BATCH_DA, shuffle=False, num_workers=0)
    ld_t25 = DataLoader(ds_t25, batch_size=BATCH_DA, shuffle=False, num_workers=0)

    print(f"  Fit days: {len(fit_dates)}  Val: {len(val_dates)}  "
          f"Test23: {len(te23)}  Test25: {len(te25)}")

    in_dim, ctx_dim = len(DADataset.SEQ_FEATS), DADataset.CTX_FEATS_SIZE
    results = {}

    for name, model in [
        ("LSTM-DA",  LSTMModel(in_dim, 128, 2, ctx_dim, DA_OUT_STEPS, 0.2)),
        ("TRANS-DA", TransformerModel(in_dim, 64, 4, 2, ctx_dim, DA_OUT_STEPS, DA_IN_STEPS)),
    ]:
        print(f"\n  Training {name} …")
        model = fit(model, ld_fit, ld_val, DA_EPOCHS, LR_DA, "da", name)
        mapes = {}
        for yr, loader, dates in [("2023", ld_t23, te23), ("2025", ld_t25, te25)]:
            mapes[yr] = _da_mape(model, loader, scaler, df30_all, dates)
        mapes["Comb."] = np.mean([mapes["2023"], mapes["2025"]])
        results[name] = mapes
        print(f"  {name}: 2023={mapes['2023']:.2f}%  "
              f"2025={mapes['2025']:.2f}%  Comb.={mapes['Comb.']:.2f}%")

    return results


@torch.no_grad()
def _da_mape(model, loader, scaler, df30, dates):
    """Compute daytime MAPE for DA task."""
    model.eval()
    all_actual, all_pred = [], []
    for batch in loader:
        seq, ctx, tgt = [b.to(DEVICE) for b in batch]
        pred = model(seq, ctx).cpu().numpy()
        tgt  = tgt.cpu().numpy()
        # de-normalise
        mu, sg = scaler.mean["Current Demand (MW)"], scaler.std["Current Demand (MW)"]
        pred = pred * sg + mu
        tgt  = tgt  * sg + mu
        all_actual.append(tgt)
        all_pred.append(pred)

    actual = np.concatenate(all_actual)  # [N_days, 48]
    pred   = np.concatenate(all_pred)

    # Daytime: 30-min steps 16..39 (08:00-20:00)
    dt_idx = list(range(16, 40))
    a_dt = actual[:, dt_idx].ravel()
    p_dt = pred[:,   dt_idx].ravel()
    mask = a_dt > 1000
    return float(np.mean(np.abs(p_dt[mask] - a_dt[mask]) / a_dt[mask] * 100)) if mask.any() else float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# 7 — SF PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_sf(df_5min, bad):
    print("\n── SF Task (5-min resolution, 1-h minimum lag) ──────────────────")
    df = df_5min.copy()
    df = df[~df["Date"].isin(bad)].sort_values("datetime").reset_index(drop=True)
    df["year"] = df["datetime"].dt.year

    # Compute time-of-day encodings
    frac = (df["hour"] * 60 + df["minute"]) / 1440.0
    df["sin_h"] = np.sin(2 * np.pi * frac)
    df["cos_h"] = np.cos(2 * np.pi * frac)
    df["sin_m"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_m"] = np.cos(2 * np.pi * df["month"] / 12)

    # Scaler from 2024 training (Jan-Oct) data
    tr_mask = (df["year"] == 2024) & (df["datetime"] < pd.Timestamp("2024-11-01"))
    scaler = Scaler()
    scaler.fit(df[tr_mask], ["Current Demand (MW)", "Solar (MW)",
                              "temp_2m", "shortwave_rad"])
    df_n = scaler.transform(df, ["Current Demand (MW)", "Solar (MW)",
                                  "temp_2m", "shortwave_rad"])
    df_n = df_n.rename(columns={
        "Current Demand (MW)": "dem_n", "Solar (MW)": "sol_n",
        "temp_2m": "temp_n", "shortwave_rad": "ghi_n"})

    feat_cols = SFDataset.SEQ_FEATS
    arr = df_n[feat_cols].fillna(0).values.astype(np.float32)

    # Validity mask — daytime only and not bad dates
    daytime = ((df["hour"] >= HOUR_FROM) & (df["hour"] < HOUR_TO) &
                (df["Current Demand (MW)"] > 1000))

    def make_loader(year_mask, shuffle):
        vm = daytime & year_mask
        ds = SFDataset(arr, vm.values)
        return DataLoader(ds, batch_size=BATCH_SF, shuffle=shuffle, num_workers=0)

    tr_mask_f = (df["year"] == 2024) & (df["datetime"] < pd.Timestamp("2024-11-01"))
    va_mask   = (df["year"] == 2024) & (df["datetime"] >= pd.Timestamp("2024-11-01"))
    te_mask23 = (df["year"] == 2023)
    te_mask25 = (df["year"] == 2025)

    ld_fit = make_loader(tr_mask_f, True)
    ld_val = make_loader(va_mask,   False)
    ld_t23 = make_loader(te_mask23, False)
    ld_t25 = make_loader(te_mask25, False)

    print(f"  Train samples: {len(SFDataset(arr, (daytime & tr_mask_f).values)):,}  "
          f" Val: {len(SFDataset(arr, (daytime & va_mask).values)):,}")

    in_dim = len(SFDataset.SEQ_FEATS)
    results = {}

    for name, model in [
        ("LSTM-SF",  LSTMModel(in_dim, 128, 2, 0, 1, 0.1)),
        ("TRANS-SF", TransformerModel(in_dim, 64, 4, 2, 0, 1, SF_IN_STEPS)),
    ]:
        print(f"\n  Training {name} …")
        model = fit(model, ld_fit, ld_val, SF_EPOCHS, LR_SF, "sf", name)
        mapes = {}
        for yr, loader in [("2023", ld_t23), ("2025", ld_t25)]:
            mapes[yr] = _sf_mape(model, loader, scaler)
        mapes["Comb."] = np.mean([mapes["2023"], mapes["2025"]])
        results[name] = mapes
        print(f"  {name}: 2023={mapes['2023']:.2f}%  "
              f"2025={mapes['2025']:.2f}%  Comb.={mapes['Comb.']:.2f}%")

    return results


@torch.no_grad()
def _sf_mape(model, loader, scaler):
    model.eval()
    errors = []
    mu, sg = scaler.mean["Current Demand (MW)"], scaler.std["Current Demand (MW)"]
    for seq, tgt in loader:
        seq, tgt = seq.to(DEVICE), tgt.to(DEVICE)
        pred = model(seq, None).cpu().numpy().ravel()
        actual = tgt.cpu().numpy().ravel()
        pred   = pred   * sg + mu
        actual = actual * sg + mu
        mask = actual > 1000
        if mask.any():
            e = np.abs(pred[mask] - actual[mask]) / actual[mask] * 100
            errors.extend(e.tolist())
    return float(np.mean(errors)) if errors else float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# 8 — RESULTS + FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})

MODEL_COLORS = {
    "NOGA-DA":  "#c0392b", "LGB-DA-W": "#2980b9",
    "LSTM-DA":  "#8e44ad", "TRANS-DA": "#1abc9c",
    "NOGA-SF":  "#e74c3c", "LGB-SF":   "#95a5a6",
    "LSTM-SF":  "#e67e22", "TRANS-SF": "#27ae60",
}
DIVIDER = "=" * 80


def print_table(da_results, sf_results):
    all_results = {**PREV, **da_results, **sf_results}
    print(f"\n{DIVIDER}")
    print("FINAL COMPARISON — Overall daytime MAPE (%)")
    print(DIVIDER)
    hdr = f"{'Model':<12} {'Type':<10} {'2023':>7} {'2025':>7} {'Comb.':>7}"
    print(hdr); print("─" * len(hdr))
    groups = [
        ("── DA ──────────────────────────────────────────", None),
        ("NOGA-DA",   "reference"),
        ("LGB-DA-W",  "LGB v2"),
        ("LSTM-DA",   "DL v3"),
        ("TRANS-DA",  "DL v3"),
        ("── SF (1-h lag) ───────────────────────────────", None),
        ("NOGA-SF",   "reference"),
        ("LGB-SF",    "LGB v1"),
        ("LSTM-SF",   "DL v3"),
        ("TRANS-SF",  "DL v3"),
    ]
    for name, mtype in groups:
        if mtype is None:
            print(name); continue
        r = all_results.get(name, {})
        if r:
            print(f"{name:<12} {mtype:<10} {r.get('2023', float('nan')):>7.2f} "
                  f"{r.get('2025', float('nan')):>7.2f} "
                  f"{r.get('Comb.', float('nan')):>7.2f}")
    print()


def save_figures(da_results, sf_results):
    # ── Fig 21: DA comparison bar chart ─────────────────────────────────
    da_models = ["NOGA-DA", "LGB-DA-W", "LSTM-DA", "TRANS-DA"]
    da_all = {**PREV, **da_results}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, yr in zip(axes, ["2023", "2025"]):
        vals  = [da_all[m].get(yr, float("nan")) for m in da_models]
        cols_ = [MODEL_COLORS[m] for m in da_models]
        bars  = ax.bar(da_models, vals, color=cols_, edgecolor="black", lw=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.axhline(da_all["NOGA-DA"].get(yr), color="#c0392b",
                   ls=":", lw=1.3, alpha=0.7, label="NOGA-DA ref")
        ax.set_title(f"DA Models — Test {yr}")
        ax.set_ylabel("MAPE (%)")
        ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.35)
        ax.legend(fontsize=8)
    fig.suptitle("Day-Ahead Forecasting: LGB vs LSTM vs Transformer\n"
                 "(Train 2024 | 30-min resolution | 48-h context)", fontsize=11)
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig21_da_dl_comparison.png")
    plt.close(fig)

    # ── Fig 22: SF comparison bar chart ─────────────────────────────────
    sf_models = ["NOGA-SF", "LGB-SF", "LSTM-SF", "TRANS-SF"]
    sf_all = {**PREV, **sf_results}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, yr in zip(axes, ["2023", "2025"]):
        vals  = [sf_all[m].get(yr, float("nan")) for m in sf_models]
        cols_ = [MODEL_COLORS[m] for m in sf_models]
        bars  = ax.bar(sf_models, vals, color=cols_, edgecolor="black", lw=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.axhline(sf_all["NOGA-SF"].get(yr), color="#e74c3c",
                   ls=":", lw=1.3, alpha=0.7, label="NOGA-SF ref")
        ax.set_title(f"SF Models (1-h lag) — Test {yr}")
        ax.set_ylabel("MAPE (%)")
        ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.40)
        ax.legend(fontsize=8)
    fig.suptitle("Short-Forecast (1-h lag): LGB vs LSTM vs Transformer\n"
                 "(Train 2024 | 5-min resolution | 6-h context window)", fontsize=11)
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig22_sf_dl_comparison.png")
    plt.close(fig)

    # ── Fig 23: Combined radar / all models ──────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    all_m = ["NOGA-DA", "LGB-DA-W", "LSTM-DA", "TRANS-DA",
             "NOGA-SF", "LGB-SF",   "LSTM-SF", "TRANS-SF"]
    all_res = {**PREV, **da_results, **sf_results}
    x = np.arange(len(all_m))
    for i, yr in enumerate(["2023", "2025"]):
        vals = [all_res.get(m, {}).get(yr, float("nan")) for m in all_m]
        ax.plot(x, vals, "o-", lw=1.6, ms=7, label=yr,
                alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(all_m, rotation=30, ha="right")
    ax.axvline(3.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.text(1.5, ax.get_ylim()[1]*0.95, "DA models", ha="center",
            fontsize=9, color="gray")
    ax.text(5.5, ax.get_ylim()[1]*0.95, "SF models (1-h lag)", ha="center",
            fontsize=9, color="gray")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("All Models: MAPE by Test Year (2023 vs 2025)\n"
                 "Dashed line separates DA from SF groups")
    ax.legend()
    plt.tight_layout()
    fig.savefig(DATA_DIR / "fig23_all_models_overview.png")
    plt.close(fig)

    print("  → Saved fig21_da_dl_comparison.png")
    print("  → Saved fig22_sf_dl_comparison.png")
    print("  → Saved fig23_all_models_overview.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9 — MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data …")
    df5 = load_5min()
    bad = bad_days(df5)
    print(f"  Bad days filtered: {len(bad)}")

    da_results = run_da(df5, bad)
    sf_results = run_sf(df5, bad)

    print_table(da_results, sf_results)
    save_figures(da_results, sf_results)

    print(f"\n{DIVIDER}")
    print("DONE — figures fig21–fig23 saved")
    print(DIVIDER)

    return da_results, sf_results


if __name__ == "__main__":
    main()
