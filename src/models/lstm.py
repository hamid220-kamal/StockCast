"""Stacked LSTM forecaster implemented in PyTorch.

The network maps a ``(batch, lookback, n_features)`` window to a single scaled
closing-price prediction for the next trading day.  Training uses Adam with
early stopping on a held-out slice of the training window.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .. import config


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


@dataclass
class LSTMRegressor:
    """Sklearn-ish wrapper around :class:`_LSTMNet`."""

    hidden: int = config.LSTM_HIDDEN
    layers: int = config.LSTM_LAYERS
    dropout: float = config.LSTM_DROPOUT
    epochs: int = config.LSTM_EPOCHS
    batch_size: int = config.LSTM_BATCH
    lr: float = config.LSTM_LR
    patience: int = config.LSTM_PATIENCE
    device: str = field(default_factory=lambda: config.DEVICE)
    history: dict = field(default_factory=dict)

    def _torch_device(self):
        dev = self.device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        return torch.device(dev)

    def fit(self, X: np.ndarray, y: np.ndarray,
            val_ratio: float = 0.15) -> "LSTMRegressor":
        torch.manual_seed(config.RANDOM_SEED)
        dev = self._torch_device()
        self.model = _LSTMNet(X.shape[2], self.hidden, self.layers,
                              self.dropout).to(dev)

        # Hold out the tail of the (already time-ordered) training data.
        n = len(X)
        n_val = max(int(n * val_ratio), 1)
        Xtr, ytr = X[:-n_val], y[:-n_val]
        Xva, yva = X[-n_val:], y[-n_val:]

        tXtr = torch.tensor(Xtr, dtype=torch.float32, device=dev)
        tytr = torch.tensor(ytr, dtype=torch.float32, device=dev)
        tXva = torch.tensor(Xva, dtype=torch.float32, device=dev)
        tyva = torch.tensor(yva, dtype=torch.float32, device=dev)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
        loss_fn = nn.MSELoss()

        train_ds = torch.utils.data.TensorDataset(tXtr, tytr)
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True)

        best_val, best_state, bad = float("inf"), None, 0
        hist_train, hist_val = [], []

        for epoch in range(self.epochs):
            self.model.train()
            tot, cnt = 0.0, 0
            for xb, yb in loader:
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                opt.step()
                tot += loss.item() * len(xb)
                cnt += len(xb)
            train_loss = tot / max(cnt, 1)

            self.model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model(tXva), tyva).item())
            hist_train.append(train_loss)
            hist_val.append(val_loss)
            sched.step(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone()
                              for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.history = {"train_loss": hist_train, "val_loss": hist_val,
                        "epochs_run": len(hist_train)}
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        dev = self._torch_device()
        self.model.eval()
        preds = []
        for i in range(0, len(X), self.batch_size):
            xb = torch.tensor(X[i:i + self.batch_size],
                              dtype=torch.float32, device=dev)
            preds.append(self.model(xb).cpu().numpy())
        return np.concatenate(preds)
