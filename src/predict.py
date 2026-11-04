"""Forward forecasting: predict the next trading day's close from a live model.

Reuses the exact scaling parameters that were fitted on the training set during
:func:`src.pipeline.run_pipeline`, so a retrained LSTM can be swapped in without
changing any other code.
"""
from __future__ import annotations

import numpy as np
import torch

from . import config, data, preprocessing
from .models.lstm import _LSTMNet


def _load_state():
    ckpt = torch.load(config.RESULTS_DIR / "lstm_state.pt", map_location="cpu",
                      weights_only=True)
    model = _LSTMNet(ckpt["n_features"], ckpt["hidden"], ckpt["layers"],
                     ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    cache = np.load(config.RESULTS_DIR / "_prep_cache.npz")
    return model, cache


def forecast_next_day(steps: int = 1) -> list[dict]:
    """Autoregressively forecast ``steps`` future closes.

    Each predicted close is fed back into the rolling feature window so the
    multi-step forecast stays self-consistent (a standard recursive strategy).
    """
    model, cache = _load_state()
    df = data.build_dataset()
    feats = preprocessing.add_features(df)

    p_min, p_max = np.ravel(cache["prep_min"]), np.ravel(cache["prep_max"])
    t_min = float(np.ravel(cache["tgt_min"])[0])
    t_max = float(np.ravel(cache["tgt_max"])[0])

    window = feats.to_numpy(dtype=float)[-config.LOOKBACK:].copy()
    out: list[dict] = []

    for step in range(steps):
        scaled = (window - p_min) / (p_max - p_min + 1e-12)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, L, F)
        with torch.no_grad():
            y_scaled = float(model(x).item())
        pred_ret = y_scaled * (t_max - t_min) + t_min   # de-scale -> return

        last_close = float(window[-1, 3])
        price = last_close * (1.0 + pred_ret)           # return -> price
        out.append({
            "step": step + 1,
            "forecast_close": round(price, 3),
            "prev_close": round(last_close, 3),
            "change_%": round(pred_ret * 100, 3),
            "direction": "UP" if pred_ret > 0 else "DOWN",
        })

        # shift the window forward, reusing the forecast as the new close
        window = np.vstack([window[1:], window[-1]])
        window[-1, 3] = price          # Close
        window[-1, 0] = price          # Open ~ Close
        window[-1, 1] = max(window[-1, 1], price)
        window[-1, 2] = min(window[-1, 2], price)
        window[-1, 5] = price / last_close - 1.0   # Return_1d

    return out
