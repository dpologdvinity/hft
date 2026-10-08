"""Replay a past trading day through the paper bot's decision path.

Each stock runs through the same market engine, strategy, budget sizing and risk
gateway as `hft trade --paper`, with the quote-level fill simulator standing in
for the broker (fills need a genuine later quote after modeled latency). It shows
what the bot would have done on that day; it is not broker evidence and one day
says nothing about profitability.
"""

import json
from decimal import Decimal
from pathlib import Path

from ..data import load_session
from ..market_engine import make_market_engine
from ..paper import PaperEngine, replay_sessions
from ..research import is_real_executable
from ..sizing import SizingConfig
from ..training import reserved_final_sessions
from .budget import budget_gateway, budget_intent_factory

ROOT = Path(__file__).resolve().parents[2]


def find_cached_session(symbol, day, data_root=ROOT / "data"):
    """A complete, checksummed session already on disk (any earlier download)."""
    for manifest in sorted(Path(data_root).rglob(f"{symbol}/{day}/manifest.json")):
        try:
            meta = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("status") == "complete" and meta.get("symbol") == symbol:
            return manifest
    return None


def ensure_session(symbol, day, data_root=ROOT / "data", download=None):
    """Session manifest for (symbol, day), downloading read-only IEX data if needed."""
    if (symbol, day) in reserved_final_sessions():
        raise ValueError(f"{symbol} {day} is reserved final-test data and cannot be replayed")
    cached = find_cached_session(symbol, day, data_root)
    if cached:
        return cached
    if download is None:
        from ..history import download_sessions as download
    index = download(symbol, day, day, Path(data_root) / "replay" / symbol.lower())
    manifest = Path(index).parent / symbol / day / "manifest.json"
    if not manifest.exists():
        raise ValueError(f"no {symbol} session on {day} (market holiday or no data)")
    return manifest


def replay_stock(manifest, budget, strategy, log_dir, *, engine="auto", risk=None):
    session = load_session(manifest, metadata="execution")
    if session.synthetic or not is_real_executable(session):
        raise ValueError("replay needs real, verified IEX data")
    symbol, budget = session.symbol, Decimal(str(budget))
    market = make_market_engine(
        symbol, strategy.name if strategy.is_rule else "model", implementation=engine
    )
    decisions = []
    engine_ = PaperEngine(
        None,
        log_path=Path(log_dir) / f"{symbol}-{session.session_id}.jsonl",
        initial_cash=budget,
        risk=risk or budget_gateway(budget),
        sizing=SizingConfig(),
        symbol=symbol,
        synthetic=False,
        source="trade-replay",
        metadata={"strategy": strategy.name, "engine": market.implementation, "replay": True},
        market_engine=market,
        intent_factory=budget_intent_factory(budget),
        log_market=False,
    )
    if strategy.is_rule:

        def policy(observation):
            decisions.append(engine_.last_update.action)
            return engine_.last_update.action

    else:
        from ..policy import OnnxPolicy

        model = OnnxPolicy(strategy.bundle, symbol=symbol, feed="iex")

        def policy(observation):
            action = model(observation)
            decisions.append(action)
            return action

    engine_.policy = policy
    result = replay_sessions(engine_, [session])
    prices = session.trade_price
    return {
        "symbol": symbol,
        "date": session.session_id,
        "engine": market.implementation,
        "budget": str(budget),
        "final_cash": str(engine_.account.cash),
        "position": str(engine_.account.position),
        "net": str(engine_.account.cash - budget),
        "return_pct": float((engine_.account.cash - budget) / budget * 100),
        "round_trips": result["trades"],
        "decisions": len(decisions),
        "long_decisions": sum(decisions),
        "stock_move_pct": float((prices[-1] / prices[0] - 1) * 100) if len(prices) else None,
        "journal": result["log"],
    }


def format_replay(rows):
    lines = [
        "symbol  date        decisions  long%  round trips   net $   return  stock moved",
    ]
    for r in rows:
        long_share = r["long_decisions"] / r["decisions"] * 100 if r["decisions"] else 0
        move = "n/a" if r["stock_move_pct"] is None else f"{r['stock_move_pct']:+.2f}%"
        lines.append(
            f"{r['symbol']:<7} {r['date']}  {r['decisions']:>9}  {long_share:4.0f}%  "
            f"{r['round_trips']:>11}  {Decimal(r['net']):>+7.2f}  {r['return_pct']:+6.2f}%  "
            f"{move:>11}"
        )
    total = sum(Decimal(r["net"]) for r in rows)
    budget = sum(Decimal(r["budget"]) for r in rows)
    lines.append(f"total net {total:+.2f} on {budget:.2f} budgeted ({total / budget * 100:+.2f}%)")
    lines.append(
        "Simulated fills on recorded IEX quotes (not broker fills); one day is not evidence of an edge."
    )
    return "\n".join(lines)
