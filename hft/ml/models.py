"""Daily and minute models for the ML day trader.

Daily: gradient-boosted trees (LightGBM) and a small multilayer network, each mapping
one symbol-day's features to its expected open-to-close return. Minute: three
sequence networks (1-D convolution, GRU, small transformer) mapping the last minutes
to the expected return over the next few minutes. Every torch model maps its input
batch to predictions of shape (N,).
"""

import lightgbm as lgb
import numpy as np
import torch
from torch import nn


class DailyMLP(nn.Module):
    def __init__(self, features: int, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MinuteCNN(nn.Module):
    def __init__(self, channels: int = 4, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, width, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=5, padding=2, dilation=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(2 * width, 1)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))  # (N, width, T)
        pooled = torch.cat([h[..., -1], h.mean(-1)], dim=-1)  # latest minute and the window
        return self.head(pooled).squeeze(-1)


class MinuteGRU(nn.Module):
    def __init__(self, channels: int = 4, width: int = 32):
        super().__init__()
        self.gru = nn.GRU(channels, width, batch_first=True)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        _, last = self.gru(x)
        return self.head(last[-1]).squeeze(-1)


class MinuteTransformer(nn.Module):
    def __init__(self, channels: int = 4, width: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(channels, width)
        self.position = nn.Parameter(torch.zeros(1, 512, width))
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.embed(x) + self.position[:, : x.shape[1]]
        return self.head(self.encoder(h)[:, -1]).squeeze(-1)


MINUTE_MODELS = {"cnn": MinuteCNN, "gru": MinuteGRU, "transformer": MinuteTransformer}


def fit_torch(model, x, y, *, epochs=10, batch=1024, lr=1e-3, weight_decay=1e-4, device="cpu"):
    """Mean-squared-error training with Adam; returns the model in eval mode."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    generator = torch.Generator().manual_seed(int(torch.initial_seed()) % (2**31))
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(xt), batch):
            index = order[start : start + batch]
            prediction = model(xt[index].to(device))
            loss = nn.functional.mse_loss(prediction, yt[index].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def train_lightgbm(x, y, x_stop, y_stop, *, seed=0, threads=1, params=None, rounds=2000):
    """Trees with early stopping on (x_stop, y_stop), a slice held out from training."""
    settings = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "num_threads": threads,
        "deterministic": True,
        "force_row_wise": True,
        "seed": seed,
        "verbosity": -1,
        **(params or {}),
    }
    train = lgb.Dataset(np.asarray(x), np.asarray(y))
    stop = lgb.Dataset(np.asarray(x_stop), np.asarray(y_stop), reference=train)
    return lgb.train(
        settings,
        train,
        rounds,
        valid_sets=[stop],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
