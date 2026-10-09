"""Score a strategy's daily returns against holding the same symbols.

The pass test is the spec's: the strategy's mean daily net return minus holding's,
with a 95% bootstrap interval over days entirely above zero. The Sharpe ratio is
also deflated for the number of variants tried (Bailey and Lopez de Prado), because
the best of many random strategies looks good by luck.
"""

import numpy as np
from scipy.stats import norm

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def _drawdown(returns):
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float((wealth / peak - 1).min()) if len(wealth) else 0.0


def _summary(returns):
    returns = np.asarray(returns, dtype=float)
    if not len(returns):
        return {"days": 0}
    std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    total = float(np.prod(1 + returns) - 1)
    return {
        "days": len(returns),
        "total": total,
        "annualized": float((1 + total) ** (TRADING_DAYS / len(returns)) - 1),
        "sharpe": float(returns.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0,
        "max_drawdown": _drawdown(returns),
    }


def deflated_sharpe(sharpe, trials, days):
    """Probability that the true (annual) Sharpe exceeds zero after picking the best of
    `trials` variants, assuming roughly normal daily returns."""
    daily = sharpe / np.sqrt(TRADING_DAYS)
    if trials > 1:
        spread = np.sqrt(1 / max(days - 1, 1))  # standard error of a daily Sharpe under zero
        expected_max = spread * (
            (1 - EULER_GAMMA) * norm.ppf(1 - 1 / trials)
            + EULER_GAMMA * norm.ppf(1 - 1 / (trials * np.e))
        )
    else:
        expected_max = 0.0
    scale = np.sqrt((1 + 0.5 * daily**2) / max(days - 1, 1))
    return float(norm.cdf((daily - expected_max) / scale))


def report(returns, holding, trades, *, trials=1, resamples=10_000, seed=0) -> dict:
    """`returns` and `holding`: daily net returns on the same days; `trades`: per-trade
    net returns."""
    returns, holding = np.asarray(returns, float), np.asarray(holding, float)
    trades = np.asarray(trades, float)
    difference = returns - holding
    rng = np.random.default_rng(seed)
    low = high = 0.0
    if len(difference) > 1:
        means = rng.choice(difference, size=(resamples, len(difference))).mean(axis=1)
        low, high = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    strategy = _summary(returns)
    return {
        "strategy": strategy,
        "holding": _summary(holding),
        "trades": len(trades),
        "per_trade": float(trades.mean()) if len(trades) else 0.0,
        "hit_rate": float((trades > 0).mean()) if len(trades) else 0.0,
        "daily_difference": float(difference.mean()) if len(difference) else 0.0,
        "daily_difference_ci95": [low, high],
        "beats_holding": bool(low > 0),
        "deflated_sharpe": deflated_sharpe(strategy.get("sharpe", 0.0), trials, len(returns)),
        "trials": trials,
    }
