"""Feature engineering, scaling and sequence construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Augment the OHLCV frame with engineered technical features."""
    out = df.copy()
    out["Return_1d"] = out["Close"].pct_change().fillna(0.0)
    out["SMA_5"] = out["Close"].rolling(5, min_periods=1).mean()
    out["SMA_20"] = out["Close"].rolling(20, min_periods=1).mean()
    out["RSI_14"] = rsi(out["Close"], 14)
    out["Volatility_10"] = out["Return_1d"].rolling(10, min_periods=1).std().fillna(0.0)

    # Keep only the configured feature + target columns and drop warm-up NaNs.
    cols = config.FEATURE_COLUMNS
    out = out[[c for c in cols if c in out.columns]].dropna()
    return out


def build_supervised(features: pd.DataFrame, target: pd.Series,
                     lookback: int = config.LOOKBACK):
    """Create overlapping ``(X, y)`` samples for the sequence model.

    ``X[i]`` is the ``lookback``-day window ending at day ``t`` and ``y[i]`` is
    the closing price on day ``t+1`` (the value we forecast).
    """
    values = features.to_numpy(dtype=float)
    tgt = target.to_numpy(dtype=float)

    X, y_price, y_ret, last_price = [], [], [], []
    for i in range(len(values) - lookback):
        prev = tgt[i + lookback - 1]          # price just before the target
        nxt = tgt[i + lookback]               # the close we forecast
        X.append(values[i:i + lookback])
        y_price.append(nxt)
        y_ret.append((nxt - prev) / prev)     # next-day simple return
        last_price.append(prev)
    return (np.asarray(X), np.asarray(y_price),
            np.asarray(y_ret), np.asarray(last_price))


def reconstruct_price(last_price: np.ndarray, y_ret: np.ndarray) -> np.ndarray:
    """Turn a predicted return back into a price via prev_close * (1 + r)."""
    return np.asarray(last_price, dtype=float) * (1.0 + np.asarray(y_ret, dtype=float))


class SequencePreprocessor:
    """Fit per-column ``MinMaxScaler`` on windows and provide (de)scaling."""

    def __init__(self) -> None:
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

    def fit(self, X_flat: np.ndarray, y: np.ndarray) -> "SequencePreprocessor":
        """``X_flat`` is ``(n_samples*lookback, n_features)``; ``y`` is 1-D."""
        self.feature_scaler.fit(X_flat)
        self.target_scaler.fit(y.reshape(-1, 1))
        return self

    def scale_X(self, X: np.ndarray) -> np.ndarray:
        n, lb, nf = X.shape
        flat = self.feature_scaler.transform(X.reshape(-1, nf))
        return flat.reshape(n, lb, nf).astype(np.float32)

    def scale_y(self, y: np.ndarray) -> np.ndarray:
        return self.target_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    def inverse_target(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(
            np.asarray(y_scaled, dtype=float).reshape(-1, 1)).ravel()


def chronological_split(*arrays: np.ndarray,
                        train_ratio: float = config.TRAIN_RATIO):
    """Split any number of parallel arrays by time (no shuffling -> no leakage).

    Returns the concatenated ``(train..., test...)`` slices.
    """
    split = int(len(arrays[0]) * train_ratio)
    out = []
    for a in arrays:
        out.extend((a[:split], a[split:]))
    # reorder to (train_x..., test_x...) grouped as train then test
    n = len(arrays)
    tr = [out[2 * i] for i in range(n)]
    te = [out[2 * i + 1] for i in range(n)]
    return tuple(tr + te)
