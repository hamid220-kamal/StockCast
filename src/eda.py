"""Exploratory Data Analysis – produces the report figures + a stats summary."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid", context="notebook")
_PALETTE = "#1f77b4"


def _save(fig, name: str) -> str:
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_price_and_volume(df: pd.DataFrame, ticker: str = config.TICKER) -> str:
    """Closing-price line with rolling means + a volume bar subplot."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(df.index, df["Close"], label="Close", color=_PALETTE, lw=1.4)
    ax1.plot(df.index, df["Close"].rolling(20).mean(),
             label="20-day MA", color="crimson", lw=1.1, alpha=0.8)
    ax1.plot(df.index, df["Close"].rolling(50).mean(),
             label="50-day MA", color="darkgreen", lw=1.1, alpha=0.8)
    ax1.set_ylabel("Price")
    ax1.set_title(f"{ticker} – Closing Price & Moving Averages")
    ax1.legend(loc="upper left")

    ax2.bar(df.index, df["Volume"], width=1.0, color="steelblue", alpha=0.7)
    ax2.set_ylabel("Volume")
    ax2.set_title("Trading Volume")
    return _save(fig, "01_price_volume.png")


def plot_return_distribution(df: pd.DataFrame, ticker: str = config.TICKER) -> str:
    """Daily-return histogram and rolling volatility."""
    ret = df["Close"].pct_change().dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(ret, bins=80, kde=True, ax=ax1, color=_PALETTE)
    ax1.axvline(0, color="grey", ls="--", lw=1)
    ax1.set_title(f"{ticker} – Daily Return Distribution")
    ax1.set_xlabel("Daily return")

    vol = ret.rolling(20).std() * np.sqrt(252) * 100
    ax2.plot(vol.index, vol, color="darkorange", lw=1.2)
    ax2.set_title("Rolling Annualised Volatility (%)")
    ax2.set_ylabel("Volatility")
    return _save(fig, "02_returns.png")


def plot_feature_correlation(features: pd.DataFrame) -> str:
    """Heat-map of correlations across engineered features."""
    corr = features.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix")
    return _save(fig, "03_correlation.png")


def summarize(df: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Return a compact, JSON-friendly EDA summary for the report."""
    ret = df["Close"].pct_change().dropna()
    return {
        "n_rows": int(len(df)),
        "date_range": [str(df.index.min().date()), str(df.index.max().date())],
        "close_min": float(df["Close"].min()),
        "close_max": float(df["Close"].max()),
        "close_start": float(df["Close"].iloc[0]),
        "close_end": float(df["Close"].iloc[-1]),
        "total_return_%": float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100),
        "mean_daily_return_%": float(ret.mean() * 100),
        "annualised_volatility_%": float(ret.std() * np.sqrt(252) * 100),
        "n_features": int(features.shape[1]),
        "features": list(features.columns),
    }


def run_eda(df: pd.DataFrame, features: pd.DataFrame,
            ticker: str = config.TICKER) -> dict:
    """Generate every EDA figure and return the summary dictionary."""
    plot_price_and_volume(df, ticker)
    plot_return_distribution(df, ticker)
    plot_feature_correlation(features)
    return summarize(df, features)
