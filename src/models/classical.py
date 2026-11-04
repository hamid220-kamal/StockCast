"""Classical (non-deep) regression baselines.

The tree / linear models cannot consume raw 3-D sequences, so each ``lookback``
window is reduced to a compact tabular row: the most-recent feature vector plus
a few summary statistics over the window (mean / std / first / last of the
close-related channels).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

RANDOM_SEED = 42


def tabularize(X: np.ndarray) -> np.ndarray:
    """``(n, lookback, n_features)`` -> ``(n, n_features + 3)`` tabular matrix."""
    last = X[:, -1, :]                 # newest day's features
    close_channel = X[:, :, 3]         # Close is the 4th feature column
    mean_c = close_channel.mean(axis=1, keepdims=True)
    std_c = close_channel.std(axis=1, keepdims=True)
    slope_c = (close_channel[:, -1] - close_channel[:, 0])[:, None]
    return np.hstack([last, mean_c, std_c, slope_c])


def build_classical_models():
    """Return freshly-instantiated, unfitted baseline regressors."""
    return {
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_SEED),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, random_state=RANDOM_SEED),
    }
