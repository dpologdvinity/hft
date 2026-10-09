"""Train one ML day-trader variant and score it against holding the same symbols.

A variant is a config: which daily model picks up to K symbols each day, and which
minute model (or none) times one long round trip in each of them, flat five minutes
before the close. Scoring simulates those trades with costs on the chosen split. The
final test split can only be scored with --final and a frozen list of variant ids,
so it is run once, after the choice is made.

    .venv/bin/python -m hft.ml.train --config cfg.json --data data/ml --out artifacts/ml
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

from .datasets import data_fingerprint, split_mask
from .download import load_calendar
from .evaluate import report
from .features import CHANNELS, DAILY_FEATURES, _aligned, daily_frame, minute_store, windows
from .fills import SLIPPAGE_BPS, half_spread_bps, simulate_day
from .models import MINUTE_MODELS, DailyMLP, fit_torch, train_lightgbm
from .universe import UNIVERSE

DEFAULT = {
    "seed": 0,
    "symbols": None,  # None: every trainable symbol in the universe
    "daily": {
        "model": "lgbm",
        "target": "return",  # return: today's open-to-close move; range: log(high / low)
        "k": 5,
        "gate": False,  # True: only pick symbols whose predicted return beats the cost
        "threshold_bp": 0.0,
        "width": 64,
        "epochs": 20,
    },
    "minute": {
        "model": "none",  # none, cnn, gru or transformer
        "horizon": 15,
        "width": 32,
        "epochs": 5,
        "lookback": 60,
        "per_day": 10,  # training windows sampled per symbol-day
        "train_fraction": 1.0,  # share of training symbol-days used for minute windows
        "margin_bp": 0.0,
    },
    "device": "cpu",
    "threads": 1,
}
RETURN_SCALE = 1e3  # minute returns are ~1e-3; scaled near unit size for training
BP = 1e4


def _merge(base, override):
    out = dict(base)
    for key, value in (override or {}).items():
        out[key] = _merge(base[key], value) if isinstance(value, dict) else value
    return out


def variant_id(config) -> str:
    """Hash of what a variant is; where it runs (device, threads) is not part of it."""
    what = {k: v for k, v in config.items() if k not in ("device", "threads")}
    text = json.dumps(what, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _costs(symbols):
    spreads = {s.ticker: s.half_spread_bps for s in UNIVERSE}
    return np.array([spreads.get(s, 4.0) for s in symbols])


# Daily ---------------------------------------------------------------------------


def _daily_target(inputs, target):
    """Per symbol-day label for the daily model: the open-to-close return, or the day's
    range, a measure of how much a stock moves (stocks "in play" for day trading)."""
    if target == "return":
        # Trades fill from the second minute's open and are flat by the last minute, so
        # that is the move to learn; the official open-to-close move is not tradable.
        out = np.full(inputs.frame.label.shape, np.nan, dtype=np.float32)
        day_index = {d: i for i, d in enumerate(inputs.frame.dates)}
        for s, symbol in enumerate(inputs.symbols):
            data = inputs.store[symbol]
            last = data.open[np.arange(len(data.dates)), data.last_minute]
            moves = np.log(last / data.open[:, 1])
            for r, day in enumerate(data.dates):
                if day in day_index:
                    out[day_index[day], s] = moves[r]
        return np.where(inputs.frame.valid, out, np.nan)
    if target != "range":
        raise ValueError("daily target must be return or range")
    out = np.full(inputs.frame.label.shape, np.nan, dtype=np.float32)
    for s, symbol in enumerate(inputs.symbols):
        _, high, low, _, _ = _aligned(inputs.root, symbol, inputs.frame.dates)
        out[:, s] = np.log(high / low)
    return np.where(inputs.frame.valid, out, np.nan)


def _fit_daily(config, frame, labels, train_days, device):
    x = frame.x[train_days]
    y = labels[train_days]
    keep = frame.valid[train_days] & np.isfinite(y)  # a missing bar leaves no label
    rows = np.flatnonzero(keep.any(axis=1))
    stop_from = rows[int(len(rows) * 0.85)] if len(rows) else 0  # last 15% stops training
    fit, stop = keep.copy(), keep.copy()
    fit[stop_from:] = False
    stop[:stop_from] = False
    settings = config["daily"]
    if settings["model"] == "lgbm":
        booster = train_lightgbm(
            x[fit], y[fit], x[stop], y[stop], seed=config["seed"], threads=config["threads"]
        )
        return DailyModel(config, booster=booster)
    model = DailyModel(
        config, mean=np.nanmean(x[fit], axis=0), std=np.nanstd(x[fit], axis=0) + 1e-9
    )
    torch.manual_seed(config["seed"])
    model.network = fit_torch(
        DailyMLP(len(DAILY_FEATURES), settings["width"]),
        model.prepare(x[fit]),
        torch.as_tensor(y[fit] * BP),
        epochs=settings["epochs"],
        device=device,
    )
    return model


class DailyModel:
    """A trained daily model that predicts, saves and loads the same way for both kinds."""

    def __init__(self, config, *, booster=None, network=None, mean=None, std=None):
        self.config, self.booster, self.network = config, booster, network
        self.mean, self.std = mean, std
        self.device = config["device"]

    def prepare(self, features):
        scaled = np.nan_to_num((features - self.mean) / self.std)
        return torch.as_tensor(scaled.astype(np.float32))

    def predict(self, features):
        if self.booster is not None:
            return self.booster.predict(features)
        with torch.no_grad():
            return self.network(self.prepare(features).to(self.device)).cpu().numpy() / BP

    def save(self, folder):
        if self.booster is not None:
            self.booster.save_model(str(folder / "daily.txt"))
            return ["daily.txt"]
        torch.save(self.network.state_dict(), folder / "daily.pt")
        np.savez(folder / "daily-scale.npz", mean=self.mean, std=self.std)
        return ["daily.pt", "daily-scale.npz"]

    @classmethod
    def load(cls, folder, config):
        if config["daily"]["model"] == "lgbm":
            return cls(config, booster=lgb.Booster(model_file=str(folder / "daily.txt")))
        scale = np.load(folder / "daily-scale.npz")
        network = DailyMLP(len(DAILY_FEATURES), config["daily"]["width"])
        network.load_state_dict(torch.load(folder / "daily.pt", map_location="cpu"))
        model = cls(config, network=network.eval(), mean=scale["mean"], std=scale["std"])
        model.network.to(model.device)
        return model


def _select(prediction, valid, costs_bp, k, trainable, *, gate=False, threshold_bp=0.0):
    """Per day, the k symbols with the highest prediction. With `gate`, only those whose
    predicted return beats the round-trip cost plus `threshold_bp` are kept."""
    score = np.where(valid & trainable, prediction, -np.inf)
    edge = prediction * BP - 2 * (costs_bp + SLIPPAGE_BPS) - threshold_bp
    picks = np.argsort(-score, axis=1)[:, :k]
    return [
        [s for s in day if np.isfinite(score[d, s]) and (not gate or edge[d, s] > 0)]
        for d, day in enumerate(picks)
    ]


# Minute --------------------------------------------------------------------------


def _session_rows(store, symbols):
    return {s: {d: r for r, d in enumerate(store[s].dates)} for s in symbols}


def _scale(x):
    x = x.copy()
    x[..., 0] *= RETURN_SCALE
    return x


def _fit_minute(config, store, symbols, frame, train_days, rng, device):
    settings = config["minute"]
    rows = _session_rows(store, symbols)
    keys = []
    for d in np.flatnonzero(train_days):
        day = frame.dates[d]
        for s, symbol in enumerate(symbols):
            if not frame.valid[d, s] or rng.random() > settings["train_fraction"]:
                continue
            row = rows[symbol].get(day)
            if row is None:
                continue
            last = int(store[symbol].last_minute[row]) - settings["horizon"] - 2
            if last <= 0:
                continue
            for minute in rng.integers(0, last, settings["per_day"]):
                keys.append((s, row, minute))
    x, y = windows(
        store, symbols, np.array(keys), lookback=settings["lookback"], horizon=settings["horizon"]
    )
    keep = np.isfinite(y)
    torch.manual_seed(config["seed"])
    model = MINUTE_MODELS[settings["model"]](x.shape[-1], settings["width"])
    return fit_torch(
        model,
        _scale(x[keep]),
        (y[keep] * BP).astype(np.float32),
        epochs=settings["epochs"],
        device=device,
    )


def _signal(model, store, symbol_index, symbols, row, settings, device):
    """Predicted gain in basis points for every minute of one session."""
    keys = np.array([(symbol_index, row, m) for m in range(390)])
    x, _ = windows(store, symbols, keys, lookback=settings["lookback"], horizon=settings["horizon"])
    with torch.no_grad():
        return model(torch.as_tensor(_scale(x)).to(device)).cpu().numpy()


# Scoring -------------------------------------------------------------------------


def _holding(root, symbols, frame, days, allowed):
    """Daily returns of holding every symbol the strategy could trade in the split, in
    equal slots: each bought at its first valid open in the split (its slot is cash
    until then, for late listings) and held to the split's end."""
    rows = np.flatnonzero(days)
    members = [s for s in np.flatnonzero(allowed) if frame.valid[rows, s].any()]
    wealth = []
    for s in members:
        o, _, _, c, _ = _aligned(root, symbols[s], frame.dates)
        first = rows[np.argmax(frame.valid[rows, s])]
        value = np.where(rows >= first, c[rows] / o[first], 1.0)
        # Carry the last value through missing days.
        index = np.where(np.isfinite(value), np.arange(len(value)), 0)
        np.maximum.accumulate(index, out=index)
        wealth.append(value[index])
    total = np.mean(wealth, axis=0)
    return total / np.concatenate([[1.0], total[:-1]]) - 1


class FinalGrant:
    """Issued by `hft.ml.search.run_final` after it records the one-time test run."""


def code_commit() -> str:
    """The code version: the git commit (with a dirty flag), or the COMMIT file that
    packaging puts into the Kaggle code archive."""
    root = Path(__file__).resolve().parents[2]
    if (root / "COMMIT").exists():
        return (root / "COMMIT").read_text().strip()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return head + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Inputs:
    root: Path
    symbols: list
    frame: object
    store: dict
    session_rows: dict
    costs_bp: np.ndarray
    trainable: np.ndarray
    data_hash: str


def load_inputs(data_root, symbols=None) -> Inputs:
    """Daily features and minute arrays, loaded once and shared by many variants."""
    root = Path(data_root)
    trade_only = {s.ticker for s in UNIVERSE if s.trade_only}
    symbols = list(symbols or [s.ticker for s in UNIVERSE if not s.trade_only])
    store = minute_store(root, symbols, load_calendar(root))
    return Inputs(
        root,
        symbols,
        daily_frame(root, symbols),
        store,
        _session_rows(store, symbols),
        _costs(symbols),
        np.array([s not in trade_only for s in symbols]),
        data_fingerprint(root),
    )


def _key(*parts):
    return json.dumps(parts, sort_keys=True, default=str)


def run_variant(
    config,
    data_root,
    out,
    *,
    split="validation",
    trials=1,
    inputs=None,
    cache=None,
    saved=None,
    grant=None,
    confidence=95.0,
):
    """Train (or reuse from `cache`, or load from `saved`) and score one variant.

    The test split needs a `FinalGrant` from `hft.ml.search.run_final`, which records
    that the one-time test was used and scores only saved, frozen models.
    """
    if split == "test" and not (isinstance(grant, FinalGrant) and saved is not None):
        raise PermissionError("the test split is scored only through hft.ml.search.run_final")
    config = _merge(DEFAULT, config)
    torch.set_num_threads(config["threads"])
    rng = np.random.default_rng(config["seed"])
    device = config["device"]
    cache = {} if cache is None else cache
    if inputs is None or (config["symbols"] and list(config["symbols"]) != inputs.symbols):
        inputs = load_inputs(data_root, config["symbols"])
    root, symbols, frame, store = inputs.root, inputs.symbols, inputs.frame, inputs.store
    session_rows, costs_bp, trainable = inputs.session_rows, inputs.costs_bp, inputs.trainable
    train_days = split_mask(frame.dates, "train")
    score_days = split_mask(frame.dates, split)
    rows = np.flatnonzero(score_days)

    settings_daily = config["daily"]
    daily_key = _key(
        "daily",
        settings_daily["model"],
        settings_daily["target"],
        settings_daily["width"],
        settings_daily["epochs"],
        config["seed"],
        split,
    )
    if saved is not None:
        cache.pop(daily_key, None)
        daily_model = DailyModel.load(Path(saved), config)
    elif daily_key not in cache:
        labels = _daily_target(inputs, settings_daily["target"])
        daily_model = _fit_daily(config, frame, labels, train_days, device)
    if daily_key not in cache:
        features = frame.x[rows].reshape(-1, frame.x.shape[-1])
        predicted = daily_model.predict(features).reshape(len(rows), -1)
        cache[daily_key] = (predicted, daily_model)
    predicted, daily_model = cache[daily_key]
    picks = _select(
        predicted,
        frame.valid[rows],
        costs_bp,
        settings_daily["k"],
        trainable,
        gate=settings_daily["gate"] and settings_daily["target"] == "return",
        threshold_bp=settings_daily["threshold_bp"],
    )

    settings = config["minute"]
    minute_model = None
    minute_key = _key(
        "minute", {k: v for k, v in settings.items() if k != "margin_bp"}, config["seed"]
    )
    if settings["model"] != "none":
        if saved is not None:
            minute_model = MINUTE_MODELS[settings["model"]](len(CHANNELS), settings["width"])
            minute_model.load_state_dict(torch.load(Path(saved) / "minute.pt", map_location="cpu"))
            minute_model = minute_model.to(device).eval()
            minute_key = _key("saved", str(saved))
            cache[minute_key] = minute_model
        elif minute_key not in cache:
            cache[minute_key] = _fit_minute(config, store, symbols, frame, train_days, rng, device)
        minute_model = cache[minute_key]

    daily_returns, trade_returns, daily_3x, trade_3x = [], [], [], []
    k = config["daily"]["k"]
    for d, chosen in zip(rows, picks, strict=True):
        day_total, day_3x = [0.0], [0.0]
        for s in chosen:
            symbol = symbols[s]
            row = session_rows[symbol].get(frame.dates[d])
            if row is None:
                continue
            data = store[symbol]
            prices = data.open[row][data.valid[row]]
            spread = half_spread_bps(costs_bp[s], float(prices[0])) if len(prices) else costs_bp[s]
            if minute_model is None:
                signal = np.full(390, np.inf)
                enter = 0.0
            else:
                signal_key = (minute_key, s, row)
                if signal_key not in cache:
                    cache[signal_key] = _signal(
                        minute_model, store, s, symbols, row, settings, device
                    )
                signal = cache[signal_key]
                enter = 2 * (spread + SLIPPAGE_BPS) + settings["margin_bp"]
            for multiple, trades, totals in (
                (1.0, trade_returns, day_total),
                (3.0, trade_3x, day_3x),
            ):
                trade = simulate_day(
                    data.open[row],
                    data.valid[row],
                    signal,
                    enter=enter,
                    half_spread_bps=spread,
                    last_minute=int(data.last_minute[row]),
                    cost_multiple=multiple,
                )
                if trade is not None:
                    trades.append(trade.net_return)
                    totals[0] += trade.net_return / k
        daily_returns.append(day_total[0])
        daily_3x.append(day_3x[0])

    holding = _holding(root, symbols, frame, score_days, trainable)
    metrics = report(daily_returns, holding, trade_returns, trials=trials, confidence=confidence)
    triple = report(daily_3x, holding, trade_3x, trials=trials, confidence=confidence)
    metrics["at_3x_cost"] = {
        key: triple[key]
        for key in (
            "strategy",
            "holding",
            "per_trade",
            "daily_difference",
            "daily_difference_ci95",
            "beats_holding",
        )
    }
    metrics.update(
        {
            "split": split,
            "variant": variant_id(config),
            "first_day": str(frame.dates[rows[0]]),
            "last_day": str(frame.dates[rows[-1]]),
            "days_with_trades": int(sum(1 for r in daily_returns if r != 0)),
            "data_hash": inputs.data_hash,
            "code": code_commit(),
        }
    )
    folder = Path(out) / variant_id(config)
    folder.mkdir(parents=True, exist_ok=True)
    if saved is None:  # a frozen model's files are never overwritten
        names = daily_model.save(folder)
        if minute_model is not None:
            torch.save(minute_model.state_dict(), folder / "minute.pt")
            names.append("minute.pt")
        metrics["weights"] = {name: _sha256(folder / name) for name in names}
        (folder / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (folder / f"metrics-{split}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/ml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/ml"))
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    metrics = run_variant(config, args.data, args.out)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
