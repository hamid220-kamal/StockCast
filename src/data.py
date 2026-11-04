"""Dataset creation and loading.

The project must run out-of-the-box without internet access, so the default
data source is a *realistic* synthetic OHLCV series generated with a
Geometric Brownian Motion (GBM) – the same stochastic process behind the
classic Black-Scholes model.  A real Kaggle / Yahoo Finance CSV can be dropped
in at any time (see ``load_real_csv`` and ``fetch_yfinance``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def generate_synthetic(n: int = config.SYNTH_TRADING_DAYS,
                       seed: int = config.SYNTH_SEED) -> pd.DataFrame:
    """Generate ``n`` daily OHLCV bars using Geometric Brownian Motion."""
    rng = np.random.default_rng(seed)

    dt = 1.0 / 252.0
    mu, sigma = config.SYNTH_ANNUAL_DRIFT, config.SYNTH_ANNUAL_VOL
    shocks = rng.normal(loc=0.0, scale=1.0, size=n)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * shocks
    close = config.SYNTH_START_PRICE * np.exp(np.cumsum(log_returns))

    # Derive intraday Open/High/Low/Volume consistently from the close path.
    open_ = np.empty_like(close)
    open_[0] = config.SYNTH_START_PRICE
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.003, size=n - 1))

    intraday_spread = np.abs(rng.normal(0, 0.008, size=n))
    high = np.maximum(open_, close) * (1 + intraday_spread)
    low = np.minimum(open_, close) * (1 - intraday_spread)

    base_volume = 6.5e7
    volume = (base_volume * (1 + 4 * intraday_spread) *
              rng.lognormal(mean=0.0, sigma=0.15, size=n)).round(0)

    # Business-day index so plots and lags behave like real market data.
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": volume},
        index=idx,
    )
    df.index.name = "Date"
    return df.round(4)


# Standard column-name aliases found across Kaggle / Yahoo / NSE-BSE dumps.
_COLUMN_ALIASES = {
    "Date": ["date", "day", "timestamp", "dt"],
    "Open": ["open", "o", "open price"],
    "High": ["high", "h"],
    "Low": ["low", "l"],
    "Close": ["close", "c", "last", "closing price"],
    "Volume": ["volume", "vol", "v", "total volume"],
    "Adj Close": ["adj close", "adjclose", "adjusted close"],
}


def load_real_csv(path: str, date_col: str | None = None) -> pd.DataFrame:
    """Load any Kaggle/CSV OHLC file and normalise it to the canonical schema.

    Handles common column-name variants and both ``Date``-index and
    ``Date``-column layouts.
    """
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}

    rename: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                rename[lower[a]] = canonical
                break
    df = df.rename(columns=rename)

    # Locate a date column if not already the index.
    if "Date" not in df.columns:
        if date_col and date_col in df.columns:
            df = df.rename(columns={date_col: "Date"})
        else:
            df["Date"] = pd.RangeIndex(len(df))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    if "Close" not in df.columns:
        raise ValueError("Could not find a closing-price column in the CSV.")

    # Backfill missing OHLC/Volume so the pipeline never sees NaNs.
    for col in ("Open", "High", "Low", "Volume"):
        if col not in df.columns:
            df[col] = df["Close"]
        df[col] = df[col].fillna(df["Close"])
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"].fillna(df["Close"])
        df = df.drop(columns=["Adj Close"])

    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def fetch_yfinance(ticker: str, period: str = "6y") -> pd.DataFrame:
    """Optional live download via yfinance (requires network + the package)."""
    import yfinance as yf  # imported lazily – optional dependency

    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.title)
    raw.index.name = "Date"
    return raw[["Open", "High", "Low", "Close", "Volume"]].dropna()


def build_dataset(force_synth: bool = False) -> pd.DataFrame:
    """Return the canonical OHLCV frame, caching it to ``config.DATASET_CSV``.

    Order of precedence: an existing cached CSV, else a freshly generated
    synthetic series (deterministic via the configured seed).
    """
    if config.DATASET_CSV.exists() and not force_synth:
        return load_real_csv(config.DATASET_CSV)

    df = generate_synthetic()
    df.to_csv(config.DATASET_CSV)
    return df
