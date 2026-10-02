"""JSON-safe session statistics and family-corrected paired block bootstrap."""

import numpy as np


def metrics(daily_equity, trade_pnl, equity_curve=None, cash_flows=None):
    equity = np.asarray(daily_equity, dtype=np.float64)
    pnl = np.asarray(trade_pnl, dtype=np.float64)
    if equity.ndim != 1 or len(equity) < 2 or not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("metrics need finite positive session equity")
    if pnl.ndim != 1 or not np.isfinite(pnl).all():
        raise ValueError("invalid completed trade PnL")
    flows = np.zeros(len(equity) - 1) if cash_flows is None else np.asarray(cash_flows, dtype=float)
    if flows.shape != (len(equity) - 1,) or not np.isfinite(flows).all():
        raise ValueError("cash flows must align to sessions")
    returns = (np.diff(equity) - flows) / equity[:-1]
    curve = np.asarray(equity if equity_curve is None else equity_curve, dtype=float)
    if curve.ndim != 1 or not len(curve) or not np.isfinite(curve).all() or (curve <= 0).any():
        raise ValueError("invalid intraday equity")
    peak = np.maximum.accumulate(curve)
    profit, loss = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    sd = np.std(returns, ddof=1) if len(returns) > 1 else 0
    return {
        "net_profit": float(equity[-1] - equity[0] - flows.sum()),
        "sharpe": float(returns.mean() / sd * np.sqrt(252)) if sd > 1e-15 else 0.0,
        "max_drawdown": float(np.max((peak - curve) / peak)),
        "expectancy": float(pnl.mean()) if len(pnl) else 0.0,
        "profit_factor": profit / loss if loss else ("infinite" if profit else None),
        "profit_factor_infinite": bool(profit > 0 and loss == 0),
        "gross_profit": profit,
        "gross_loss": loss,
        "trades": len(pnl),
        "sessions": len(returns),
        "daily_returns": returns.tolist(),
        "sample_counts": {
            "sessions": len(returns),
            "completed_trades": len(pnl),
            "intraday_marks": len(curve),
        },
    } | {"return": float(np.prod(1 + returns) - 1)}


def paired_bootstrap(agent, baseline, seed=42):
    a, b = np.asarray(agent, dtype=float), np.asarray(baseline, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired returns must align and be finite")
    quantiles = [0.05 / 6, 1 - 0.05 / 6]
    result = {
        "samples": len(a),
        "resamples": 2000,
        "quantiles": quantiles,
        "family_alpha": 0.05,
        "comparisons": 3,
    }
    if len(a) < 2:
        return result | {"lower": None, "upper": None, "positive": False}
    length = max(1, round(np.sqrt(len(a))))
    starts = np.random.default_rng(seed).integers(
        0, len(a), size=(2000, (len(a) + length - 1) // length)
    )
    indices = (starts[..., None] + np.arange(length)) % len(a)
    means = (a - b)[indices.reshape(2000, -1)[:, : len(a)]].mean(axis=1)
    low, high = np.quantile(means, quantiles)
    return result | {
        "lower": float(low),
        "upper": float(high),
        "positive": bool(low > 0),
        "block_length": length,
    }


class EquitySummary:
    """Constant-memory intraday extrema, composable across ordered sessions."""

    def __init__(self, initial):
        self.minimum = self.maximum = self.peak = float(initial)
        self.drawdown = 0.0
        self.marks = 0
        self.add(initial)

    def add(self, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("invalid intraday equity")
        self.minimum = min(self.minimum, value)
        self.maximum = self.peak = max(self.peak, value)
        self.drawdown = max(self.drawdown, (self.peak - value) / self.peak)
        self.marks += 1

    def combine(self, later):
        self.drawdown = max(self.drawdown, later.drawdown, (self.peak - later.minimum) / self.peak)
        self.minimum = min(self.minimum, later.minimum)
        self.maximum = self.peak = max(self.peak, later.maximum)
        self.marks += later.marks
