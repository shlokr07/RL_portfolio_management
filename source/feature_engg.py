# src/feature_engineering.py
import numpy as np
import pandas as pd
import json
from pathlib import Path

# ---------- CONFIG ----------
EPS = 1e-8
INPUT_TRADABLE = Path("data/tradable")
INPUT_MACRO = Path("data/features")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
HDF_PATH = OUT_DIR / "features_weekly.h5"
STATS_JSON = OUT_DIR / "feature_stats.json"

# ---------- IO ----------
def read_csv_folder(folder: Path):
    files = sorted(folder.glob("*.csv"))
    dfs = {}
    for f in files:
        ticker = f.stem
        df = pd.read_csv(f, parse_dates=True, index_col=0)
        df.columns = [c.capitalize() for c in df.columns]
        dfs[ticker] = df
    return dfs

def resample_weekly(df):
    return df.resample("W-SUN").last()

# ---------- numeric helpers ----------
def safe_log(series: pd.Series):
    s = series.astype(float).fillna(0.0)
    return np.log(np.clip(s + EPS, EPS, None))

def safe_div(numer: pd.Series, denom: pd.Series):
    denom = denom.replace(0, np.nan).fillna(EPS)
    return numer / denom

# ---------- feature computation ----------
def compute_features_for_ticker(df_weekly: pd.DataFrame):
    out = pd.DataFrame(index=df_weekly.index)
    close = df_weekly["Close"].astype(float)
    vol = df_weekly["Volume"].astype(float) if "Volume" in df_weekly.columns else pd.Series(0.0, index=df_weekly.index)

    out["close"] = close
    out["logdiff_close"] = safe_log(close).diff()

    for col in ("Open", "High", "Low"):
        if col in df_weekly.columns:
            out[f"logdiff_{col.lower()}"] = safe_log(df_weekly[col].astype(float)).diff()

    # Weekly rupee volatility
    rupee_vol = vol * close
    out["log_diff_weekly_rupeevol"] = safe_log(rupee_vol).diff()

    # Illiquidity (Amihud) computed weekly
    r_pct = close.pct_change().replace([np.inf, -np.inf], np.nan)
    out["illiquidity"] = safe_div(r_pct.abs(), rupee_vol)

    # Only save logdiff moving averages (no raw MA)
    for w in [4, 12, 24, 52]:
        ma = close.rolling(window=w, min_periods=w).mean()
        out[f"logdiff_ma_{w}"] = safe_log(ma).diff()

    # Quarter position
    quarters = out.index.to_period("Q")
    qp = pd.Series(index=out.index, dtype=float)
    for q, idxs in out.groupby(quarters).groups.items():
        idxs = list(idxs)
        L = len(idxs)
        if L == 1:
            qp.loc[idxs] = 1.0
        else:
            for i, idx in enumerate(idxs):
                qp.loc[idx] = i / (L - 1)
    out["quarter_position"] = qp
    return out

# ---------- main pipeline ----------
def postprocess_and_save(tradables_dict: dict, macros_dict: dict):
    trad_feats = {}
    for t, df in tradables_dict.items():
        weekly = resample_weekly(df)
        feats = compute_features_for_ticker(weekly)
        feats = feats.iloc[52:]  # drop first 52 weeks
        feats = feats.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        trad_feats[t] = feats

    macro_feats = {}
    for t, df in macros_dict.items():
        weekly = resample_weekly(df)
        close = weekly["Close"].astype(float)
        dfm = pd.DataFrame(index=weekly.index)
        dfm["close"] = close
        dfm["logdiff_close"] = safe_log(close).diff()
        macro_feats[t] = dfm

    all_start_dates = [df.index[0] for df in list(trad_feats.values()) + list(macro_feats.values())]
    all_end_dates = [df.index[-1] for df in list(trad_feats.values()) + list(macro_feats.values())]
    common_start, common_end = max(all_start_dates), min(all_end_dates)
    print(f"Common date range: {common_start} to {common_end}")

    # Align to common date range
    for t in trad_feats:
        trad_feats[t] = trad_feats[t][(trad_feats[t].index >= common_start) & (trad_feats[t].index <= common_end)]
    for t in macro_feats:
        macro_feats[t] = macro_feats[t][(macro_feats[t].index >= common_start) & (macro_feats[t].index <= common_end)]
        macro_feats[t] = macro_feats[t].replace([np.inf, -np.inf], np.nan).dropna(how="any")

    # Save HDF5
    with pd.HDFStore(HDF_PATH, mode="w") as store:
        for t, df in trad_feats.items():
            store.put(f"tradables/{t}", df, format="table")
        for t, df in macro_feats.items():
            store.put(f"macros/{t}", df, format="table")

    # Stats
    stats = {"tradables": {}, "macros": {}}
    for t, df in trad_feats.items():
        stats["tradables"][t] = {col: {"min": float(df[col].min()),
                                       "max": float(df[col].max()),
                                       "mean": float(df[col].mean()),
                                       "std": float(df[col].std())} for col in df.columns}
    for t, df in macro_feats.items():
        stats["macros"][t] = {col: {"min": float(df[col].min()),
                                    "max": float(df[col].max()),
                                    "mean": float(df[col].mean()),
                                    "std": float(df[col].std())} for col in df.columns}

    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\n✅ Feature statistics saved to {STATS_JSON}")
    print(f"✅ Saved processed HDF5 file: {HDF_PATH}")

def main():
    trad = read_csv_folder(INPUT_TRADABLE)
    macros = read_csv_folder(INPUT_MACRO)
    if not trad:
        raise RuntimeError(f"No tradable CSVs found in {INPUT_TRADABLE}")
    if not macros:
        raise RuntimeError(f"No macro CSVs found in {INPUT_MACRO}")
    postprocess_and_save(trad, macros)

if __name__ == "__main__":
    main()
