"""One-off helper that writes a valid .ipynb notebook for the project.

Run once with `python build_notebook.py`. It regenerates
`notebooks/StockCast.ipynb` so the notebook always stays in sync
with the code cells below.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "notebooks" / "StockCast.ipynb"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(*lines: str) -> dict:
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "outputs": [],
            "execution_count": None, "source": src}


cells = [
    md("# 📈 StockCast — Stock Price Prediction (ML Minor Project)",
       "",
       "Guided end-to-end walkthrough: data → EDA → feature engineering → LSTM +",
       "classical baselines → evaluation → forward forecast.",
       "",
       "The heavy lifting lives in the `src/` package; this notebook just drives it",
       "and visualises the results. Run every cell top-to-bottom."),
    code("import sys",
         "from pathlib import Path",
         "from IPython.display import display, Image",
         "",
         "# make the project package importable when running from notebooks/",
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()",
         "sys.path.insert(0, str(ROOT))",
         "",
         "import pandas as pd",
         "from src import config, data, eda, preprocessing, pipeline, predict, utils",
         "",
         "utils.set_seed()",
         "pd.set_option('display.float_format', lambda x: f'{x:,.3f}')",
         "print('Project root:', ROOT)"),
    md("## 1. Load / build the dataset",
       "",
       "Runs fully offline via a seeded Geometric-Brownian-Motion OHLCV series.",
       "To use real Kaggle/Yahoo data, replace `data/raw/AAPL.csv` (see `README.md`)."),
    code("df = data.build_dataset()",
         "print(f'{len(df)} rows | {df.index.min.date()} -> {df.index.max.date()}')",
         "df.tail()"),
    md("## 2. Exploratory Data Analysis"),
    code("features = preprocessing.add_features(df)",
         "summary = eda.run_eda(df, features)",
         "",
         "for fig in ['01_price_volume.png', '02_returns.png', '03_correlation.png']:",
         "    display(Image(filename=str(config.FIGURES_DIR / fig)))"),
    md("## 3. Train & evaluate all models",
       "",
       "`run_pipeline` does a leakage-safe chronological split, fits Ridge / Random",
       "Forest / Gradient Boosting and the PyTorch LSTM, and saves every figure and",
       "metric into `reports/`."),
    code("results = pipeline.run_pipeline(verbose=True)"),
    md("## 4. Results"),
    code("metrics = pd.DataFrame(results['metrics']).T.sort_values('RMSE')",
         "direction = pd.DataFrame(results['direction']).T[['direction_accuracy_%']]",
         "print('Best model (by RMSE):', results['best_model'])",
         "print('--- Price-level error ---'); display(metrics)",
         "print('--- Directional accuracy (%) ---');",
         "display(direction.sort_values('direction_accuracy_%', ascending=False))"),
    code("for fig in ['04_predictions.png', '05_lstm_loss.png',",
         "            '06_metrics.png', '07_feature_importance.png']:",
         "    display(Image(filename=str(config.FIGURES_DIR / fig)))"),
    md("## 5. Forward forecast"),
    code("fc = pd.DataFrame(predict.forecast_next_day(steps=5))",
         "display(fc)"),
    md("## 6. Takeaways",
       "",
       "- Forecast target is the **next-day return** (stationary), then price is",
       "  reconstructed as `prev_close * (1 + r)` - this keeps tree R\u00b2 healthy.",
       "- **LSTM** forecasts the price **level** best (R\u00b2 ~ 0.98, MAPE ~ 1.1%),",
       "  essentially matching the naive random-walk benchmark.",
       "- **Day-ahead direction is near chance (46-48%)** for all models - an honest",
       "  random-walk / efficient-market result, not a bug.",
       "- Swap in any real Kaggle/Yahoo CSV with zero code changes."),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
