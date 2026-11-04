"""Shared helpers: seeding, regression metrics and direction accuracy."""
from __future__ import annotations

import random

import numpy as np

from . import config


def set_seed(seed: int | None = None) -> None:
    """Make every downstream experiment reproducible."""
    seed = config.RANDOM_SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # torch optional at import-time
        pass


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return the standard regression error metrics plus R^2."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    denom = np.abs(y_true)
    mask = denom > 1e-9
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom[mask])) * 100.0)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE_%": mape, "R2": r2}


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                       baseline: np.ndarray) -> dict:
    """Compare predicted vs. actual next-day movement against a naive baseline.

    ``baseline`` is the last known price for each sample, i.e. the "no change"
    forecast used by the naive strategy.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    baseline = np.asarray(baseline, dtype=float).ravel()

    true_dir = y_true > baseline
    pred_dir = y_pred > baseline
    naive_dir = baseline >= baseline  # naive always predicts "flat/up"

    model_acc = float(np.mean(true_dir == pred_dir)) * 100.0
    naive_acc = float(np.mean(true_dir == naive_dir)) * 100.0
    return {
        "direction_accuracy_%": model_acc,
        "naive_baseline_accuracy_%": naive_acc,
    }
