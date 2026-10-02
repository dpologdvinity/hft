"""Offline candidate search, strictly temporal evaluation and cost-matched benchmarks."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .account import Costs
from .data import MarketData, session_day
from .env import TradingEnv
from .features import LOOKBACK, TARGETS
from .policy import export_model


@dataclass(frozen=True)
class Split:
    train_stop: int
    test_start: int
    test_stop: int

    @property
    def context_start(self):
        return self.test_start - LOOKBACK - 1


def day_boundaries(data):
    starts = [0]
    for i in range(1, len(data)):
        if session_day(data.rows[i - 1].timestamp) != session_day(data.rows[i].timestamp):
            starts.append(i)
    return starts + [len(data)]


def walk_forward(data, train_days, test_days):
    bounds = day_boundaries(data)
    if train_days < 1 or test_days < 1:
        raise ValueError("walk-forward windows must be positive")
    splits = [
        Split(bounds[d], bounds[d], bounds[d + test_days])
        for d in range(train_days, len(bounds) - test_days, test_days)
    ]
    if not splits or any(s.context_start < 0 for s in splits):
        raise ValueError("insufficient history for temporal validation")
    return splits


def metrics(daily_equity, trade_pnl, equity_curve=None):
    equity = np.asarray(daily_equity, dtype=np.float64)
    if len(equity) < 2 or not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("metrics need two finite positive equity observations")
    pnl = np.asarray(trade_pnl, dtype=np.float64)
    if not np.isfinite(pnl).all():
        raise ValueError("nonfinite trade PnL")
    returns = np.diff(equity) / equity[:-1]
    deviation = np.std(returns, ddof=1) if len(returns) > 1 else 0
    curve = np.asarray(equity_curve if equity_curve is not None else equity, dtype=np.float64)
    if not np.isfinite(curve).all() or (curve <= 0).any():
        raise ValueError("invalid equity curve")
    peaks = np.maximum.accumulate(curve)
    profit, loss = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    return {
        "net_profit": float(equity[-1] - equity[0]),
        "return": float(equity[-1] / equity[0] - 1),
        "sharpe": float(np.mean(returns) / deviation * np.sqrt(252)) if deviation > 1e-15 else 0.0,
        "max_drawdown": float(np.max((peaks - curve) / peaks)),
        "expectancy": float(np.mean(pnl)) if len(pnl) else 0.0,
        "profit_factor": profit / loss if loss else None,
        "gross_profit": profit,
        "gross_loss": loss,
        "trades": len(pnl),
        "daily_returns": returns.tolist(),
    }


def paired_bootstrap(agent, baseline, seed=42):
    a, b = np.asarray(agent), np.asarray(baseline)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired returns must align and be finite")
    if len(a) < 2:
        return {"lower": None, "upper": None, "positive": False, "samples": len(a)}
    # Resample short contiguous blocks to retain some daily serial dependence.
    delta = a - b
    rng = np.random.default_rng(seed)
    length = max(1, round(np.sqrt(len(a))))
    starts = rng.integers(0, len(a), size=(2000, (len(a) + length - 1) // length))
    indices = (starts[..., None] + np.arange(length)) % len(a)
    means = delta[indices.reshape(2000, -1)[:, : len(a)]].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"lower": float(low), "upper": float(high), "positive": bool(low > 0), "samples": len(a)}


def _rollout(data, policy, costs, initial_cash):
    env = TradingEnv(data, costs=costs, initial_cash=initial_cash, episode_steps=len(data))
    obs, _ = env.reset()
    daily, curve, targets, dates = {}, [initial_cash], [], []
    done = False
    while not done:
        action = int(policy(obs))
        targets.append(TARGETS[action] if action in range(3) else None)
        obs, _, terminated, truncated, info = env.step(action)
        day = session_day(info["timestamp"])
        daily[day] = info["equity"]
        curve.append(info["equity"])
        dates.append(day)
        done = terminated or truncated
    result = metrics([initial_cash] + list(daily.values()), env.account.closed_pnl, curve)
    result.update(
        daily_equity=list(daily.values()),
        dates=list(daily),
        fills=len(env.account.fills),
        target_changes=sum(t != p for t, p in zip(targets, [0] + targets[:-1])),
    )
    return result, targets


def evaluate(data, policy, *, costs=None, initial_cash=10_000.0, seed=42):
    costs = costs or Costs()
    agent, targets = _rollout(data, policy, costs, initial_cash)
    long, _ = _rollout(data, lambda obs: 1, costs, initial_cash)
    changes = [t != p for t, p in zip(targets, [0] + targets[:-1])]
    rng, random_actions, target = np.random.default_rng(seed), [], 0
    for change in changes:
        if change:
            target = int(rng.choice([t for t in TARGETS if t != target]))
        random_actions.append(TARGETS.index(target))
    iterator = iter(random_actions)
    random, _ = _rollout(data, lambda obs: next(iterator), costs, initial_cash)
    candle, _ = _rollout(
        data, lambda obs: 1 if obs[4] > 0.25 else 2 if obs[4] < -0.25 else 0, costs, initial_cash
    )
    return {
        "policy": agent,
        "buy_hold": long,
        "random": random,
        "candlestick": candle,
        "edge": {
            name: paired_bootstrap(agent["daily_returns"], result["daily_returns"], seed)
            for name, result in [("buy_hold", long), ("random", random), ("candlestick", candle)]
        },
    }


def _fit(data, timesteps, seed, candidate, initial_cash, costs, n_envs):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(1)
    env = DummyVecEnv(
        [
            lambda: TradingEnv(
                data,
                costs=costs,
                initial_cash=initial_cash,
                random_start=True,
                augment=True,
                episode_steps=256,
                drawdown_penalty=candidate["drawdown_penalty"],
            )
            for _ in range(n_envs)
        ]
    )
    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=candidate["learning_rate"],
            n_steps=256,
            batch_size=64,
            seed=seed,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=timesteps)
        return model
    finally:
        env.close()


def train(
    data: MarketData,
    output,
    *,
    timesteps=20_000,
    candidates=3,
    seed=42,
    initial_cash=10_000.0,
    costs=None,
    n_envs=4,
):
    """Select on expanding validation folds; refit, then evaluate the untouched final days."""
    if timesteps < 1 or candidates < 1 or n_envs < 1:
        raise ValueError("training budgets must be positive")
    costs, output = costs or Costs(), Path(output)
    bounds = day_boundaries(data)
    days = len(bounds) - 1
    if days < 5:
        raise ValueError("need at least five sessions for training, validation and test")
    train_days = max(2, int(days * 0.6))
    val_days = max(1, int(days * 0.2))
    test_start = bounds[train_days + val_days]
    development = data.subset(0, test_start)
    splits = walk_forward(development, train_days, 1)
    output.mkdir(parents=True, exist_ok=True)
    search = []
    for i in range(candidates):
        candidate = {
            "learning_rate": (3e-4, 1e-4, 5e-4)[i % 3],
            "drawdown_penalty": (0.0, 0.1, 0.5)[i % 3],
        }
        scores, validation = [], []
        for j, split in enumerate(splits):
            model = _fit(
                development.subset(0, split.train_stop),
                timesteps,
                seed + i * 100 + j,
                candidate,
                initial_cash,
                costs,
                n_envs,
            )
            result = evaluate(
                development.subset(split.context_start, split.test_stop),
                lambda obs: model.predict(obs, deterministic=True)[0],
                costs=costs,
                initial_cash=initial_cash,
                seed=seed,
            )
            scores.append(
                result["policy"]["net_profit"] - result["policy"]["max_drawdown"] * initial_cash
            )
            validation.append(result)
        search.append(
            {"parameters": candidate, "score": float(np.mean(scores)), "folds": validation}
        )
        print(f"candidate {i + 1}/{candidates}: validation score={np.mean(scores):.4f}", flush=True)
    winner = max(range(candidates), key=lambda i: search[i]["score"])
    model = _fit(
        development, timesteps, seed, search[winner]["parameters"], initial_cash, costs, n_envs
    )
    model.save(str(output / "ppo.zip"))
    test = evaluate(
        data.subset(test_start - LOOKBACK - 1, len(data)),
        lambda obs: model.predict(obs, deterministic=True)[0],
        costs=costs,
        initial_cash=initial_cash,
        seed=seed,
    )
    report = {
        "synthetic": data.synthetic,
        "symbol": data.symbol,
        "seed": seed,
        "train_days": train_days,
        "validation_days": val_days,
        "test_days": days - train_days - val_days,
        "selected_candidate": winner,
        "search": search,
        "test": test,
        "ready_for_paper_review": bool(
            not data.synthetic
            and test["policy"]["sharpe"] > 1
            and all(v["positive"] for v in test["edge"].values())
        ),
    }
    (output / "research.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    export_model(
        model,
        output / "policy.onnx",
        symbol=data.symbol,
        costs=costs,
        synthetic=data.synthetic,
        bar_seconds=data.bar_seconds,
        initial_cash=initial_cash,
    )
    return report
