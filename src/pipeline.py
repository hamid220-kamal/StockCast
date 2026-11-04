"""End-to-end training / evaluation pipeline.

`run_pipeline` is the single entry point used by both ``main.py`` and the
Jupyter notebook: it loads data, engineers features, runs EDA, trains the LSTM
and the classical baselines on a chronological split, evaluates them on the
unseen test tail, then writes every figure and metric to ``reports/``.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config, data, eda, preprocessing, utils
from .models import LSTMRegressor, classical


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def _plot_predictions(test_dates, y_true, y_pred_series: dict, last_price):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_dates, y_true, label="Actual", color="black", lw=1.6)
    colors = plt.cm.tab10.colors
    for i, (name, y_pred) in enumerate(y_pred_series.items()):
        ax.plot(test_dates, y_pred, label=name, lw=1.2, alpha=0.85,
                color=colors[i % len(colors)])
    ax.plot(test_dates, last_price, "--", label="Naive (last price)",
            color="grey", lw=1.0, alpha=0.7)
    ax.set_title("Test-set Close Price: Actual vs. Predictions")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "04_predictions.png", dpi=130)
    plt.close(fig)


def _plot_loss(history: dict):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["train_loss"], label="Train loss", color="steelblue")
    ax.plot(history["val_loss"], label="Validation loss", color="crimson")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled)")
    ax.set_title("LSTM Training / Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "05_lstm_loss.png", dpi=130)
    plt.close(fig)


def _plot_metrics_table(all_metrics: dict):
    dfm = pd.DataFrame(all_metrics).T
    fig, ax = plt.subplots(figsize=(10, 4))
    dfm[["RMSE", "MAE", "MAPE_%"]].plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_title("Regression Error by Model (lower is better)")
    ax.set_ylabel("Value")
    ax.set_xticklabels(dfm.index, rotation=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "06_metrics.png", dpi=130)
    plt.close(fig)


def _plot_feature_importance(model, X_test_tab, y_test):
    try:
        r = permutation_importance(model, X_test_tab, y_test, n_repeats=8,
                                   random_state=config.RANDOM_SEED, scoring="r2")
    except Exception:
        return
    order = np.argsort(r.importances_mean)
    names = [config.FEATURE_COLUMNS[min(i, len(config.FEATURE_COLUMNS) - 1)]
             for i in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, r.importances_mean[order], xerr=r.importances_std[order],
            color="seagreen")
    ax.set_title("Permutation Importance of Features (RandomForest)")
    ax.set_xlabel("R^2 drop")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "07_feature_importance.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run_pipeline(ticker: str = config.TICKER, verbose: bool = True) -> dict:
    utils.set_seed()
    log = print if verbose else (lambda *a, **k: None)

    # 1. Data + features ---------------------------------------------------
    df = data.build_dataset()
    df = df.rename(columns={c: c for c in df.columns})
    features = preprocessing.add_features(df)
    target = features[config.TARGET_COLUMN]

    log(f"[data] {len(df)} rows, {features.shape[1]} features "
        f"({df.index.min().date()} -> {df.index.max().date()})")

    # 2. EDA ---------------------------------------------------------------
    eda_summary = eda.run_eda(df, features, ticker)

    # 3. Supervised windows + chronological split --------------------------
    X, y_price, y_ret, last_price = preprocessing.build_supervised(features, target)
    (Xtr, yret_tr, yprice_tr, lp_tr,
     Xte, yret_te, yprice_te, lp_te) = preprocessing.chronological_split(
        X, y_ret, y_price, last_price)
    yte = yprice_te            # actual test closes, used for reporting
    log(f"[split] train={len(Xtr)}  test={len(Xte)}  lookback={config.LOOKBACK}")

    # 4. Scale (fit on train only). Models forecast the RETURN, which is
    #    stationary - so tree ensembles no longer have to extrapolate raw price.
    prep = preprocessing.SequencePreprocessor()
    prep.fit(Xtr.reshape(-1, Xtr.shape[-1]), yret_tr)
    Xtr_s = prep.scale_X(Xtr)
    Xte_s = prep.scale_X(Xte)
    ytr_s = prep.scale_y(yret_tr)
    ret_te_s = prep.scale_y(yret_te)

    # matching test dates for plotting
    test_dates = features.index[config.LOOKBACK:][int(len(X) * config.TRAIN_RATIO):]

    results: dict = {"ticker": ticker, "eda": eda_summary,
                     "split": {"train": int(len(Xtr)), "test": int(len(Xte)),
                               "lookback": config.LOOKBACK},
                     "metrics": {}, "direction": {}}
    preds_real: dict[str, np.ndarray] = {}

    # 5. Classical baselines ----------------------------------------------
    Xtr_tab = classical.tabularize(Xtr_s)
    Xte_tab = classical.tabularize(Xte_s)
    rf_model = None
    for name, model in classical.build_classical_models().items():
        model.fit(Xtr_tab, ytr_s)
        pred_ret = prep.inverse_target(model.predict(Xte_tab))   # return space
        yhat = preprocessing.reconstruct_price(lp_te, pred_ret)  # back to price
        preds_real[name] = yhat
        results["metrics"][name] = utils.regression_metrics(yte, yhat)
        results["direction"][name] = utils.direction_accuracy(yte, yhat, lp_te)
        if name == "RandomForest":
            rf_model = model
        log(f"[train] {name:16s} RMSE={results['metrics'][name]['RMSE']:.3f}")

    # 6. LSTM --------------------------------------------------------------
    lstm = LSTMRegressor(device=config.DEVICE)
    lstm.fit(Xtr_s, ytr_s)
    pred_ret_lstm = prep.inverse_target(lstm.predict(Xte_s))
    yhat_lstm = preprocessing.reconstruct_price(lp_te, pred_ret_lstm)
    preds_real["LSTM"] = yhat_lstm
    results["metrics"]["LSTM"] = utils.regression_metrics(yte, yhat_lstm)
    results["direction"]["LSTM"] = utils.direction_accuracy(yte, yhat_lstm, lp_te)
    results["lstm_epochs_run"] = lstm.history["epochs_run"]
    log(f"[train] {'LSTM':16s} RMSE={results['metrics']['LSTM']['RMSE']:.3f} "
        f"(epochs={lstm.history['epochs_run']})")

    # 6b. Naive persistence baseline (forecast = last close) --------------
    preds_real["Naive"] = lp_te
    results["metrics"]["Naive"] = utils.regression_metrics(yte, lp_te)

    # 7. Figures -----------------------------------------------------------
    _plot_predictions(test_dates, yte, preds_real, lp_te)
    _plot_loss(lstm.history)
    _plot_metrics_table(results["metrics"])
    _plot_feature_importance(rf_model, Xte_tab, ret_te_s)

    # 8. Persist outputs ---------------------------------------------------
    out_df = pd.DataFrame({"Actual": yte}, index=test_dates)
    for name, yhat in preds_real.items():
        out_df[name] = yhat
    out_df.to_csv(config.RESULTS_DIR / "test_predictions.csv")

    # 9. Pick best trained model by RMSE (Naive is a benchmark, not a model)
    trained = [k for k in results["metrics"] if k != "Naive"]
    best = min(trained, key=lambda k: results["metrics"][k]["RMSE"])
    results["best_model"] = best
    best_dir = max(trained, key=lambda k: results["direction"][k]["direction_accuracy_%"])
    results["best_direction_model"] = best_dir
    log(f"[best] {best} (RMSE={results['metrics'][best]['RMSE']:.3f}); "
        f"best direction: {best_dir}")

    # 8b. Persist metrics now that the summary fields are complete
    with open(config.RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)

    # cache artefacts for forward forecasting
    np.savez(config.RESULTS_DIR / "_prep_cache.npz",
             last_window=X[-1], prep_min=prep.feature_scaler.data_min_,
             prep_max=prep.feature_scaler.data_max_,
             tgt_min=prep.target_scaler.data_min_,
             tgt_max=prep.target_scaler.data_max_)
    lstm.model.eval()
    torch_save = config.RESULTS_DIR / "lstm_state.pt"
    import torch
    torch.save({"state_dict": lstm.model.state_dict(),
                "n_features": Xtr_s.shape[-1],
                "hidden": config.LSTM_HIDDEN, "layers": config.LSTM_LAYERS,
                "dropout": config.LSTM_DROPOUT}, torch_save)

    return results
