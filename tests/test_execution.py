import json
from dataclasses import replace

import pytest

from hft.account import Account, Costs
from hft.data import NS, Bar, synthetic_data
from hft.feed import BarAggregator, parse_timestamp
from hft.paper import PaperBroker, PaperEngine
from hft.risk import RiskConfig, RiskGateway


def quote(time=1_735_830_000 * NS, price=100):
    return Bar(time, price, price, price, price, 100, price - 0.01, price + 0.01, 10, 10)


def test_paper_latency_uses_future_quote_and_deduplicates():
    account = Account(1000, Costs(0, 0))
    risk = RiskGateway(RiskConfig())
    broker = PaperBroker(account, risk, latency_ms=75)
    q = quote()
    assert broker.submit(1, q, q.timestamp) is None
    assert broker.submit(1, q, q.timestamp) is None
    assert len(risk.order_times) == 1
    assert (
        broker.on_quote(replace(q, timestamp=q.timestamp + 50_000_000), q.timestamp + 50_000_000)
        is None
    )
    later = quote(q.timestamp + 100_000_000, 102)
    fill = broker.on_quote(later, later.timestamp)
    assert fill.price == 102.01
    assert account.position == 1
    assert broker.pending is None
    assert broker.submit(1, later, later.timestamp) is None
    assert len(account.fills) == 1


def test_pending_fill_is_revalidated_and_daily_kill_can_flatten_over_rate_limit():
    account = Account(1000, Costs(0, 0))
    risk = RiskGateway(RiskConfig(max_orders_per_minute=1, max_exposure=110))
    broker = PaperBroker(account, risk)
    q = quote()
    broker.submit(1, q, q.timestamp)
    expensive = quote(q.timestamp + NS, 120)
    assert broker.on_quote(expensive, expensive.timestamp) is None
    assert account.position == 0
    assert broker.last_rejection == "exposure_limit"
    account.target(1, q)
    loss = quote(q.timestamp + 2 * NS, 70)
    assert risk.check(1, loss, account, loss.timestamp) == "daily_loss"
    assert risk.check(0, loss, account, loss.timestamp) is None
    broker.flatten(loss)
    assert account.position == 0
    assert risk.check(1, loss, account, loss.timestamp) == "daily_loss"


def test_stale_quote_news_leverage_and_volatility_guards():
    q, a = quote(), Account(1000)
    risk = RiskGateway(
        RiskConfig(max_quote_age_seconds=2), blackouts=[(q.timestamp, q.timestamp + NS)]
    )
    assert risk.check(1, q, a, q.timestamp + 3 * NS) == "stale_quote"
    assert risk.check(1, q, a, q.timestamp) == "news_blackout"
    assert (
        RiskGateway(RiskConfig(max_leverage=0.05)).check(1, q, a, q.timestamp) == "leverage_limit"
    )
    history = [quote(q.timestamp + i * NS) for i in range(65)]
    history[-1] = replace(history[-1], high=110, low=90)
    assert (
        RiskGateway(RiskConfig()).check(1, history[-1], a, history[-1].timestamp, history=history)
        == "volatility_breaker"
    )
    with pytest.raises(ValueError):
        RiskConfig(max_daily_loss=float("nan"))


def test_stream_aggregation_is_causal_with_trade_volume():
    agg = BarAggregator("AAPL", bar_seconds=1)
    t = "2025-01-02T14:30:00.100000000Z"
    q = {"T": "q", "S": "AAPL", "t": t, "bp": 99.99, "ap": 100.01, "bs": 5, "as": 10}
    assert agg.add(q) is None
    assert (
        agg.add({"T": "t", "S": "AAPL", "t": "2025-01-02T14:30:00.500000000Z", "p": 100, "s": 7})
        is None
    )
    later = dict(q, t="2025-01-02T14:30:01.100000000Z", bp=109.99, ap=110.01)
    bar = agg.add(later)
    assert bar.close == 100
    assert bar.volume == 7
    assert bar.bid == 99.99
    assert bar.timestamp == parse_timestamp("2025-01-02T14:30:01Z")
    assert parse_timestamp(t) % NS == 100_000_000
    with pytest.raises(ValueError):
        agg.add(q)  # out-of-order data cannot be folded into a future candle


def test_engine_replay_logs_fill_equity_and_final_flatten(tmp_path):
    data = synthetic_data(days=2, bars_per_day=100)
    path = tmp_path / "paper.jsonl"
    engine = PaperEngine(
        lambda obs: 1,
        log_path=path,
        initial_cash=1000,
        costs=Costs(0, 0),
        symbol="SYNTH",
        synthetic=True,
    )
    for b in data.rows:
        engine.on_bar(b)
    engine.finish(data.rows[-1], complete=False)
    assert engine.account.position == 0
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[0]["event"] == "start"
    assert any(e["event"] == "fill" and "slippage_bps" in e for e in events)
    assert any(e["event"] == "decision" for e in events)
    assert events[-1]["event"] == "finish"
    assert not events[-1]["complete"]
