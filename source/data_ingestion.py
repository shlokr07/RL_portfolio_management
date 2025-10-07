# src/data_fetch.py
import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

# -------------------------------------------------
# 1. Define tradable equities (NIFTY 50 components)
# -------------------------------------------------
STOCKS = {
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS"],
    "IT": ["INFY.NS", "TCS.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS"],
    "Auto": ["TATAMOTORS.NS", "MARUTI.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS"],
    "Metals": ["TATASTEEL.NS", "JSWSTEEL.NS"],
    "Telecom": ["BHARTIARTL.NS"],
}

# -------------------------------------------------
# 2. Define non-tradable macro/sector ETFs (features only)
# -------------------------------------------------
MACRO_ETFS = [
    "^CNXIT",
    "^CNXINFRA",
    "^CNXFMCG",
    "^CNXMETAL",
    "^CNXREALTY",
    "^CNXPHARMA",
    "^CNXAUTO",
    "^CNXENERGY",
    "^NSEBANK",
    "^NSEI",
]

# -------------------------------------------------
# 3. Combine lists for download
# -------------------------------------------------
TRADABLE_TICKERS = sum(STOCKS.values(), [])
FEATURE_TICKERS = MACRO_ETFS
ALL_TICKERS = TRADABLE_TICKERS + FEATURE_TICKERS

# -------------------------------------------------
# 4. Data fetcher
# -------------------------------------------------
def fetch_ohlcv(tickers, start="2011-07-18", end=None, interval="1d"):
    """Download OHLCV data for each ticker from Yahoo Finance."""
    end = end or datetime.today().strftime("%Y-%m-%d")
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
    )

    ohlcv_dict = {}
    for t in tickers:
        if t in data.columns.levels[0]:
            df = data[t].copy()
            df.dropna(how="any", inplace=True)
            ohlcv_dict[t] = df
    return ohlcv_dict

# -------------------------------------------------
# 5. Save to CSVs
# -------------------------------------------------
def save_to_csv(data_dict, folder_path):
    """Save each ticker’s DataFrame to its own CSV file."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    for ticker, df in data_dict.items():
        path = folder / f"{ticker.replace('^','').replace('.','_')}.csv"
        df.to_csv(path)
    print(f"Saved {len(data_dict)} tickers to {folder_path}")

# -------------------------------------------------
# 6. Run script
# -------------------------------------------------
if __name__ == "__main__":
    ohlcv = fetch_ohlcv(ALL_TICKERS)

    tradable = {t: ohlcv[t] for t in TRADABLE_TICKERS if t in ohlcv}
    features = {t: ohlcv[t] for t in FEATURE_TICKERS if t in ohlcv}

    save_to_csv(tradable, "data/tradable")
    save_to_csv(features, "data/features")
