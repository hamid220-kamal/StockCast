"""Central configuration for the Stock Price Prediction project.

All paths, hyper-parameters and dataset settings live here so the rest of
the code base stays free of magic numbers.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORT_DIR / "figures"
RESULTS_DIR = REPORT_DIR / "results"

for _d in (RAW_DATA_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
TICKER = "AAPL"                       # used for titles / file naming
DATASET_CSV = RAW_DATA_DIR / f"{TICKER}.csv"

# Synthetic-generator settings (used when no real CSV is supplied).
SYNTH_START_PRICE = 150.0             # opening price of the first day
SYNTH_ANNUAL_DRIFT = 0.18             # ~18% average yearly upward drift
SYNTH_ANNUAL_VOL = 0.28               # ~28% annualised volatility
SYNTH_TRADING_DAYS = 1500             # ~6 years of daily bars
SYNTH_SEED = 42                       # reproducible dataset

# --------------------------------------------------------------------------
# Feature engineering / windowing
# --------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "Return_1d", "SMA_5", "SMA_20", "RSI_14", "Volatility_10",
]
TARGET_COLUMN = "Close"
LOOKBACK = 60                        # days fed to the LSTM for one prediction
TRAIN_RATIO = 0.80                    # chronological 80/20 split
SEQ_STRIDE = 1                        # sliding-window stride

# --------------------------------------------------------------------------
# LSTM hyper-parameters
# --------------------------------------------------------------------------
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 60
LSTM_BATCH = 64
LSTM_LR = 1e-3
LSTM_PATIENCE = 10                    # early-stopping patience (epochs)
DEVICE = "cpu"                        # set to "cuda" when available

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42
