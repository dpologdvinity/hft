"""Demonstration: can a neural network "learn to profit" from patternless prices?

Part 1: coin-flip prices (no pattern by construction). Train until training profit is
positive, then test on fresh coin flips. Also combine three networks.
Part 2: the same network on real daily bars (46 stocks), trained 2016-2021, tested 2022-2024.
Profits are sums of per-period returns in percent (not compounded). Results and context:
docs/project-history.md, experiment 12. Takes about 22 minutes on one CPU core:

    .venv/bin/python scripts/overfitting_demo.py  # needs data/daily-bars from overnight_auction.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
torch.manual_seed(0)
rng = np.random.default_rng(0)
LAGS = 20
COST = 1e-4  # 1 bp per unit of position change (generous: real round trips cost more)


def features(returns, bodies, wicks):
    """Lagged returns plus candlestick-style body/wick features, per step."""
    n = len(returns)
    x = np.zeros((n - LAGS, 3 * LAGS), dtype=np.float32)
    for k in range(LAGS):
        x[:, k] = returns[LAGS - 1 - k : n - 1 - k]
        x[:, LAGS + k] = bodies[LAGS - 1 - k : n - 1 - k]
        x[:, 2 * LAGS + k] = wicks[LAGS - 1 - k : n - 1 - k]
    y = returns[LAGS:].astype(np.float32)  # the return the position earns
    scale = x.std(axis=0) + 1e-12
    return x / scale, y


def coin_flip_market(n):
    returns = rng.normal(0, 0.01, n)  # independent: no pattern exists
    bodies = rng.normal(0, 1, n)
    wicks = np.abs(rng.normal(0, 1, n))
    return features(returns, bodies, wicks)


class Trader(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * LAGS, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)  # position: 0 (cash) to 1 (long)


def profit(position, y):
    """Total return in percent: position x next return minus trading costs."""
    turnover = torch.abs(position[1:] - position[:-1]).sum() + position[0]
    return (100 * ((position * y).sum() - COST * turnover)).item()


def train(x, y, epochs=400, label=""):
    model = Trader()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for epoch in range(epochs):
        position = model(xt)
        turnover = torch.abs(position[1:] - position[:-1]).sum()
        loss = -((position * yt).sum() - COST * turnover)  # maximize profit after costs
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                print(f"  {label} epoch {epoch:>3}: training profit {profit(model(xt), yt):+8.1f}%")
    return model


def evaluate(models, x, y):
    with torch.no_grad():
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        positions = torch.stack([m(xt) for m in models]).mean(0)  # combine by averaging
        return profit(positions, yt), (yt.sum() * 100).item()


results = {}
print("PART 1: coin-flip prices (there is no pattern to find)")
x_train, y_train = coin_flip_market(20_000)
x_test, y_test = coin_flip_market(20_000)
models = [train(x_train, y_train, label=f"net {i + 1}") for i in range(3)]
for name, ms in (("one network", models[:1]), ("three combined", models)):
    train_p, _ = evaluate(ms, x_train, y_train)
    test_p, hold = evaluate(ms, x_test, y_test)
    print(
        f"  {name:<15} training {train_p:+8.1f}%   fresh data {test_p:+8.1f}%   (buy and hold {hold:+.1f}%)"
    )
    results[f"coin_{name}"] = {"train": train_p, "test": test_p, "hold": hold}

print("\nPART 2: real daily bars, 46 stocks; train 2016-2021, test 2022-2024")
cache = Path("data/daily-bars")
xs_tr, ys_tr, xs_te, ys_te = [], [], [], []
for path in sorted(cache.glob("*.json")):
    if path.stem in ("SPY", "QQQ", "IWM", "DIA"):
        continue
    bars = json.loads(path.read_text())
    o = np.array([b["open"] for b in bars])
    c = np.array([b["close"] for b in bars])
    dates = [b["date"] for b in bars]
    returns = np.diff(np.log(c), prepend=np.log(c[0]))
    bodies = np.log(c / o)  # the day's candle body
    wicks = np.log(o / np.concatenate([[c[0]], c[:-1]]))  # the overnight gap
    x, y = features(returns, bodies, wicks)
    split = next(i for i, d in enumerate(dates[LAGS:]) if d >= "2022-01-01")
    xs_tr.append(x[:split])
    ys_tr.append(y[:split])
    xs_te.append(x[split:])
    ys_te.append(y[split:])
real = {}
models = []
for i in range(3):
    print(f" net {i + 1}")
    model = Trader()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(300):
        loss = 0
        for x, y in zip(xs_tr, ys_tr, strict=True):
            position = model(torch.from_numpy(x))
            yt = torch.from_numpy(y)
            loss = loss - (
                (position * yt).sum() - COST * torch.abs(position[1:] - position[:-1]).sum()
            )
        opt.zero_grad()
        loss.backward()
        opt.step()
    models.append(model)
for name, ms in (("one network", models[:1]), ("three combined", models)):
    tr = [evaluate(ms, x, y) for x, y in zip(xs_tr, ys_tr, strict=True)]
    te = [evaluate(ms, x, y) for x, y in zip(xs_te, ys_te, strict=True)]
    train_p, train_hold = np.mean([a for a, _ in tr]), np.mean([b for _, b in tr])
    test_p, test_hold = np.mean([a for a, _ in te]), np.mean([b for _, b in te])
    print(
        f"  {name:<15} 2016-21 {train_p:+7.1f}% (hold {train_hold:+7.1f}%)   "
        f"2022-24 {test_p:+7.1f}% (hold {test_hold:+7.1f}%)   per stock, log returns"
    )
    real[name] = [train_p, train_hold, test_p, test_hold]
sys.stdout.flush()
