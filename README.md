# 📈 StockCast — Stock Price Prediction using LSTM & Classical Machine Learning

*ML Minor Project — forecasting the next-day close price and direction of a stock.*

**Author:** Hamid Kamal · **Internship:** 4 Nov – 5 Dec 2026

A complete, reproducible machine-learning project that forecasts the next-day
**closing price** and **direction** of a stock. It compares a **PyTorch LSTM**
against classical baselines (**Ridge**, **Random Forest**, **Gradient
Boosting**) on a chronological train/test split, and ships with EDA figures,
evaluation metrics, saved model weights and a forward-forecast CLI.

The current dataset is **real AAPL data from Yahoo Finance** (Nov 2020 – Nov
2026, cached at `data/raw/AAPL.csv`).

> **Minor-project brief:** *"The art of forecasting stock prices … Good and
> effective prediction systems help traders, investors and analysts by providing
> supportive information like the future direction of the stock market."*

---

## ✨ What you get

| Component | Location |
|---|---|
| Package / config | `src/config.py` |
| Data (synthetic GBM + real-CSV loader + optional yfinance) | `src/data.py` |
| Feature engineering & windowing | `src/preprocessing.py` |
| Exploratory Data Analysis | `src/eda.py` |
| Models (LSTM + sklearn baselines) | `src/models/` |
| End-to-end train/eval pipeline | `src/pipeline.py` |
| Forward forecasting | `src/predict.py` |
| CLI | `main.py` |
| Guided notebook | `notebooks/StockCast.ipynb` |
| Written report | `Project_Report.md` |
| Generated figures | `reports/figures/` |
| Metrics & predictions | `reports/results/` |

---

## 🚀 Quick start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. create/cache the dataset (works fully offline out-of-the-box)
python main.py data

# 3. run EDA + train all models + evaluate + save figures/metrics
python main.py train

# 4. forecast the next 5 trading days
python main.py forecast --steps 5

# or do everything in one shot
python main.py all
```

---

## 🗂️ Dataset / using your own data

`data/raw/AAPL.csv` currently holds **real AAPL data from Yahoo Finance**. If
that file is ever missing, `build_dataset()` falls back to a **realistic
synthetic** series (Geometric Brownian Motion, seeded → reproducible) so the
project still runs fully offline. To point it at different data:

**Option A – drop in a CSV.** Replace `data/raw/AAPL.csv` with any Kaggle/Yahoo
file containing `Date, Open, High, Low, Close, Volume` (column names are
auto-detected). Then run `python main.py train`.

**Option B – pull from Yahoo Finance** (needs internet):

```bash
pip install yfinance
python -c "from src.data import fetch_yfinance; \
fetch_yfinance('AAPL','6y').to_csv('data/raw/AAPL.csv')"
python main.py train
```

Change `TICKER` in `src/config.py` to relabel titles/files.

---

## 🧠 Methodology (short version)

1. **Features** – raw OHLCV plus engineered indicators: 1-day return, SMA(5),
   SMA(20), RSI(14), rolling volatility(10).
2. **Target** – the **next-day return** (not raw price): returns are stationary,
   so tree models don't have to extrapolate a trending price; the forecast close
   is reconstructed as `prev_close × (1 + r̂)`.
3. **Sequences** – a 60-day sliding window predicts one day ahead.
4. **Split** – 80/20 **chronological** (no shuffling → no future leakage);
   scaling is fit on the training set only.
5. **Models** – Ridge / RandomForest / GradientBoosting on tabularised windows,
   and a 2-layer LSTM on the raw sequences, plus a naive random-walk benchmark.
6. **Evaluation** – RMSE / MAE / MAPE / R² for price level, and directional
   accuracy vs. an always-long baseline.

See `Project_Report.md` for the full write-up and results.

---

## 📦 Project layout

```
StockCast/
├── main.py                     # CLI entry point
├── requirements.txt
├── Project_Report.md           # submission write-up
├── data/raw/AAPL.csv           # dataset (cached)
├── notebooks/                  # guided Jupyter notebook
├── reports/
│   ├── figures/                # 7 generated charts
│   └── results/                # metrics.json, predictions, model weights
└── src/
    ├── config.py  data.py  preprocessing.py  eda.py
    ├── pipeline.py  predict.py  utils.py
    └── models/  classical.py  lstm.py
```

---

## ⚠️ Disclaimer

This is an educational project. Stock markets are noisy and non-stationary;
goodness-of-fit metrics can be misleading. Nothing here is investment advice.
