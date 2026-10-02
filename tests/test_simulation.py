from decimal import Decimal as D

import numpy as np
import pytest

from hft.account import Account
from hft.calendar import SessionWindow
from hft.data import Quote, synthetic_sessions
from hft.env import TradingEnv
from hft.execution import LocalExecution
from hft.risk import RiskGateway, Snapshot
from hft.sizing import OrderIntent

NS = 10**9


def quote(id, ns, ask="100", size=".2"):
    return Quote(id, ns, None, D("99.99"), D(ask), D(size), D(size))


def kernel():
    a = Account(500)
    g = RiskGateway()
    e = LocalExecution(a, g)
    session = SessionWindow("day", NS, 1000 * NS, None)
    q = quote("old", 5 * NS)
    s = Snapshot(q, session, D(0), D(500), D(500), D(500), ())
    i = OrderIntent("order", "SPY", "buy", D(".4"), D("100.01"), 5 * NS)
    assert e.submit(i, s, 5 * NS).allowed
    return a, g, e, session


def test_latency_displayed_size_dedup_and_complete_roundtrip():
    a, _g, e, s = kernel()
    assert not e.on_quote(quote("early", 5 * NS + 50_000_000), 5 * NS + 50_000_000, session=s)
    t = 5 * NS + 100_000_000
    q = quote("fill", t)
    executions = e.on_quote(q, t, session=s)
    assert executions[0].price == D("100.01")
    assert a.position == D(".2") and e.pending
    assert not e.on_quote(q, t, session=s)
    e.on_quote(quote("second", t + 1), t + 1, session=s)
    assert a.position == D(".4") and e.pending is None


def test_noncrossing_expiry_stale_halt_never_fills():
    a, g, e, s = kernel()
    t = 5 * NS + 100_000_000
    assert not e.on_quote(quote("expensive", t, ask="101"), t, session=s)
    assert not a.fills
    e.on_quote(quote("expired", 10 * NS), 10 * NS, session=s)
    assert e.pending is None and e.unfilled == 1
    a.execute(".4", 100, 0, 1)
    g.halted = "daily_loss"
    old = quote("stale", 10 * NS)
    snap = Snapshot(old, s, a.position, a.mark(old.mid), a.initial_cash, a.cash, ())
    i = OrderIntent("exit", "SPY", "sell", a.position, D(99), 10 * NS)
    assert e.submit(i, snap, 10 * NS).allowed
    assert not e.on_quote(quote("stale2", 10 * NS + 100_000_000), 13 * NS, session=s)
    assert a.position == D(".4")


def test_environment_schema_seed_and_checker():
    from gymnasium.utils.env_checker import check_env

    s = synthetic_sessions(days=1, bars_per_day=100)[0]
    env = TradingEnv(s, random_start=True, episode_steps=10)
    x, _ = env.reset(seed=22)
    y, _ = env.reset(seed=22)
    np.testing.assert_array_equal(x, y)
    assert x.shape == (17,) and x.dtype == np.float32 and env.action_space.n == 2
    # Checker resets active episodes; keep its sampled probe flat. Inventory
    # reset rejection is covered separately with a real executed entry.
    env.action_space.seed(123)
    check_env(env, skip_render_check=True)


def test_terminal_without_liquidity_is_incomplete():
    s = synthetic_sessions(days=1, bars_per_day=80)[0]
    env = TradingEnv(s, episode_steps=2)
    env.reset()
    env.step(1)
    assert env.account.position > 0
    env.simulation.execution.latency_ns = 10**15
    _, _, _, truncated, info = env.step(0)
    assert truncated and info["incomplete"] and env.account.position > 0


def test_fill_revalidation_observes_new_blackout():
    a, g, e, s = kernel()
    g.blackouts = ((5 * NS + 90_000_000, 6 * NS),)
    t = 5 * NS + 100_000_000
    assert not e.on_quote(quote("blocked", t), t, session=s)
    assert e.pending is None and e.last_decision.reason == "news_blackout"
    assert a.position == 0


def test_simulator_does_not_discard_not_yet_arrived_quote():
    from dataclasses import replace

    from hft.execution import Simulation

    data = synthetic_sessions(days=1, bars_per_day=80)[0]
    decision = data.open_ns + 305 * NS
    arrivals = data.quote_ns.copy()
    k = int(np.searchsorted(data.quote_ns, decision + 100_000_000))
    arrivals[k] = data.quote_ns[k] + 500_000_000
    manifest = dict(data.manifest)
    manifest["quote_metadata"] = {"arrival_ns": arrivals}
    data = replace(data, manifest=manifest)
    sim = Simulation(data)
    sim.advance_to(decision + 200_000_000)
    assert sim.quote.arrival_ns < decision + 200_000_000
    sim.advance_to(decision + 800_000_000)
    assert sim.now_ns == decision + 800_000_000
    assert sim.quote.arrival_ns < sim.now_ns


def test_equity_curve_retains_intra_bar_price_excursion():
    from dataclasses import replace

    from hft.execution import Simulation

    data = synthetic_sessions(days=1, bars_per_day=80)[0]
    decision = data.open_ns + 305 * NS
    bid = np.full(len(data.bid), 99.999999)
    ask = np.full(len(data.ask), 100.012345)
    k = int(np.searchsorted(data.quote_ns, decision + NS))
    bid[k : k + 5] = 101.999999
    ask[k : k + 5] = 102.012345
    data = replace(data, bid=bid, ask=ask, trade_price=np.full(len(data.trade_price), 100.0))
    sim = Simulation(data)
    sim.submit_action(1)
    sim.advance_to(decision + 5 * NS)
    assert len(sim.equity_curve) > 2
    assert max(sim.equity_curve) > float(sim.snapshot().equity) + 0.5
    assert sim.equity_curve[0] == 500
    assert sim.equity_curve[-1] == float(sim.snapshot().equity)


def test_simulation_quote_identity_matches_streaming_normalization():
    from hft.execution import Simulation
    from hft.feed import quote_from_event

    session = synthetic_sessions(days=1, bars_per_day=80)[0]
    sim = Simulation(session)
    first = next(event for event in session.iter_events() if event["T"] == "q")
    assert sim._quote(0).quote_id == quote_from_event(first).quote_id


def gap_session():
    from dataclasses import replace

    data = synthetic_sessions(days=1, bars_per_day=160)[0]
    cut_start, cut_end = data.open_ns + 306 * NS, data.open_ns + 313 * NS
    q = (data.quote_ns < cut_start) | (data.quote_ns >= cut_end)
    t = (data.trade_ns < cut_start) | (data.trade_ns >= cut_end)
    return replace(
        data,
        **{
            name: getattr(data, name)[q]
            for name in ("quote_ns", "bid", "ask", "bid_size", "ask_size")
        },
        **{name: getattr(data, name)[t] for name in ("trade_ns", "trade_price", "trade_size")},
    )


def test_historical_gap_cancels_order_and_requires_new_contiguous_warmup(tmp_path):
    from hft.execution import Simulation
    from hft.paper import PaperEngine
    from hft.risk import RiskConfig

    data = gap_session()
    sim = Simulation(data, risk_config=RiskConfig(order_expiry_seconds=20), latency_ms=10000)
    assert sim.submit_action(1).allowed
    sim.advance_to(data.open_ns + 315 * NS)
    assert sim.execution.pending is None
    assert not sim.decision_eligible
    assert sim.submit_action(1).reason == "insufficient_warmup"
    sim.advance_to(data.open_ns + 610 * NS)
    assert not sim.decision_eligible
    sim.advance_to(data.open_ns + 615 * NS)
    assert not sim.decision_eligible
    sim.advance_to(data.open_ns + 620 * NS)
    assert sim.decision_eligible
    assert len(sim.history) == 61
    assert sim.history[0].end_ns > data.open_ns + 313 * NS

    paper = PaperEngine(lambda obs: 0, log_path=tmp_path / "gap.jsonl", symbol=data.symbol)
    paper.start_session(SessionWindow(data.session_id, data.open_ns, data.close_ns))
    for event in data.iter_events():
        now = event.get("arrival_ns", event["event_ns"])
        if now > data.open_ns + 620 * NS:
            break
        paper.on_event(event, now)
    assert tuple(paper.history) == sim.history
    paper.finish()


def test_simulation_reset_uses_same_bounded_risk_history_as_stream():
    from hft.execution import Simulation

    data = synthetic_sessions(days=1, bars_per_day=160)[0]
    sim = Simulation(data)
    sim.reset(100)
    assert len(sim.history) == 61
    assert sim.history[0].end_ns == data.open_ns + 205 * NS


def test_incomplete_episode_cannot_reset_away_held_inventory():
    data = synthetic_sessions(days=1, bars_per_day=80)[0]
    env = TradingEnv(data, episode_steps=2)
    env.reset()
    env.step(1)
    env.simulation.execution.latency_ns = 10**15
    _, _, _, _, info = env.step(0)
    assert info["incomplete"]
    held = env.account.position
    with pytest.raises(RuntimeError, match="inventory"):
        env.reset()
    assert env.account.position == held


def test_terminal_exit_refreshes_limit_using_future_actual_quotes():
    from dataclasses import replace

    data = synthetic_sessions(days=1, bars_per_day=100)[0]
    changed = data.quote_ns >= data.open_ns + 310 * NS
    data = replace(
        data,
        bid=np.where(changed, 99.5, 99.99),
        ask=np.where(changed, 99.52, 100.01),
        trade_price=np.full(len(data.trade_price), 100.0),
    )
    env = TradingEnv(data, episode_steps=2)
    env.reset()
    env.step(1)
    assert env.account.position > 0
    _, _, _, truncated, info = env.step(0)
    assert truncated and not info["incomplete"]
    assert env.account.position == 0
    assert env.account.completed_trades[-1].exit_ns >= data.open_ns + 315 * NS
    assert env.account.completed_trades[-1].pnl < 0


def test_boundary_timer_publishes_bar_without_using_future_arrivals():
    from dataclasses import replace

    from hft.execution import Simulation

    data = synthetic_sessions(days=1, bars_per_day=100)[0]
    boundary = data.open_ns + 305 * NS
    # Delay every event at the boundary by 200ms, within aggregator tolerance.
    qm = data.quote_ns == boundary
    tm = data.trade_ns == boundary
    manifest = dict(data.manifest)
    manifest["quote_metadata"] = {"arrival_ns": data.quote_ns + qm * 200_000_000}
    manifest["trade_metadata"] = {"arrival_ns": data.trade_ns + tm * 200_000_000}
    # The next quote is also delayed, so first publication is exactly +200ms.
    manifest["quote_metadata"]["arrival_ns"][data.quote_ns == boundary + 100_000_000] += 200_000_000
    sim = Simulation(replace(data, manifest=manifest))
    assert sim.decision_eligible
    assert sim.history[-1].end_ns == boundary
    assert sim.quote.event_ns < boundary
    before = sim.history
    sim.advance_to(boundary + 200_000_000)
    assert sim.decision_eligible
    assert sim.history[-1].end_ns == boundary
    assert sim.history == before


def test_reset_reuses_prepared_events_without_changing_observation_or_quotes():
    data = synthetic_sessions(days=1, bars_per_day=100)[0]
    env = TradingEnv(data, episode_steps=2)
    initial, _ = env.reset(seed=4)
    prepared = env.simulation
    env.step(0)
    env.step(0)
    repeated, _ = env.reset(seed=4)
    assert env.simulation is prepared
    np.testing.assert_array_equal(initial, repeated)
    assert env.simulation.quote.event_ns == data.open_ns + 304900000000


def test_atr_pause_starts_when_trade_publishes_bar_before_next_quote():
    from dataclasses import replace

    from hft.execution import Simulation

    data = synthetic_sessions(days=1, bars_per_day=100)[0]
    boundary = data.open_ns + 315 * NS
    keep = data.quote_ns != boundary
    prices = np.full(len(data.trade_price), 100.0)
    prices[data.trade_ns == boundary - NS] = 110
    data = replace(
        data,
        trade_price=prices,
        **{
            name: getattr(data, name)[keep]
            for name in ("quote_ns", "bid", "ask", "bid_size", "ask_size")
        },
    )
    sim = Simulation(data)
    sim.advance_to(boundary + 200_000_000)
    assert sim.risk.breaker_until == boundary + 60 * NS


def test_environment_decision_clock_waits_for_bar_publication():
    from dataclasses import replace

    data = synthetic_sessions(days=1, bars_per_day=100)[0]
    boundary = data.open_ns + 305 * NS
    qarrivals = data.quote_ns.copy()
    tarrivals = data.trade_ns.copy()
    qarrivals[data.quote_ns == boundary] += 200_000_000
    qarrivals[data.quote_ns == boundary + 100_000_000] += 200_000_000
    tarrivals[data.trade_ns == boundary] += 200_000_000
    data = replace(
        data,
        manifest=data.manifest
        | {
            "quote_metadata": {"arrival_ns": qarrivals},
            "trade_metadata": {"arrival_ns": tarrivals},
        },
    )
    env = TradingEnv(data, episode_steps=2)
    _, info = env.reset()
    assert info["timestamp"] == boundary
    assert info["decision_eligible"]
    assert env.history[-1].end_ns == boundary


@pytest.mark.parametrize("edge", ["initial", "trailing", "trailing_late_arrival"])
def test_session_edge_silence_counts_as_runtime_feed_gap(edge):
    from dataclasses import replace

    from hft.execution import Simulation

    data = synthetic_sessions(days=1, bars_per_day=100)[0]
    if edge == "initial":
        q = data.quote_ns >= data.open_ns + 10 * NS
        t = data.trade_ns >= data.open_ns + 10 * NS
    else:
        q = data.quote_ns < data.close_ns - 10 * NS
        if edge == "trailing_late_arrival":
            q[-1] = True
        t = data.trade_ns < data.close_ns - 10 * NS
    data = replace(
        data,
        **{
            name: getattr(data, name)[q]
            for name in ("quote_ns", "bid", "ask", "bid_size", "ask_size")
        },
        **{name: getattr(data, name)[t] for name in ("trade_ns", "trade_price", "trade_size")},
    )
    if edge == "trailing_late_arrival":
        arrivals = data.quote_ns.copy()
        arrivals[-1] = data.close_ns + NS
        data = replace(data, manifest=data.manifest | {"quote_metadata": {"arrival_ns": arrivals}})
    sim = Simulation(data)
    sim.advance_to(data.close_ns)
    assert sim.gap_count == 1


@pytest.mark.parametrize("case", ["future", "equal_stamp"])
def test_reset_quote_matches_last_valid_streaming_quote(case):
    from dataclasses import replace

    from hft.execution import Simulation

    data = synthetic_sessions(days=1, bars_per_day=100)[0]
    boundary = data.open_ns + 305 * NS
    expected = int(np.searchsorted(data.quote_ns, boundary)) - 1
    if case == "future":
        arrivals = data.quote_ns.copy()
        arrivals[expected + 1] -= 500_000_000
        data = replace(data, manifest=data.manifest | {"quote_metadata": {"arrival_ns": arrivals}})
    else:
        stamps = data.quote_ns.copy()
        stamps[expected - 1] = stamps[expected]
        data = replace(data, quote_ns=stamps)
    sim = Simulation(data)
    assert sim.quote.bid == D(str(data.bid[expected]))
    assert sim.quote.ask == D(str(data.ask[expected]))


def test_reward_penalizes_real_intra_bar_drawdown_with_flat_endpoints():
    from dataclasses import replace

    from hft.account import Costs

    data = synthetic_sessions(days=1, bars_per_day=100)[0]
    prices = np.full(len(data.bid), 100.0)
    k = int(np.searchsorted(data.quote_ns, data.open_ns + 306 * NS))
    prices[k : k + 5] = 102
    data = replace(data, bid=prices, ask=prices, trade_price=np.full(len(data.trade_price), 100.0))
    env = TradingEnv(data, costs=Costs(0, 0), episode_steps=2)
    env.reset()
    env.account.execute(".4", 100, 0, data.open_ns + 305 * NS)
    _, reward, _, _, info = env.step(1)
    assert info["equity"] == 500
    assert reward == pytest.approx(-1.6)


def test_bar_quote_uses_last_arrival_for_equal_event_timestamps():
    from hft.feed import BarAggregator

    start = 100 * NS
    agg = BarAggregator("SPY", session=SessionWindow("day", start, start + 10 * NS))
    agg.add({"T": "t", "S": "SPY", "event_ns": start + NS, "p": 100, "s": 1})
    for identity, price in [("first", 100), ("latest", 101)]:
        agg.add(
            {
                "T": "q",
                "S": "SPY",
                "event_ns": start + 4900000000,
                "i": identity,
                "bp": price,
                "ap": price + 0.02,
                "bs": 1,
                "as": 1,
            }
        )
    bar = agg.advance_to(start + 5 * NS)[0]
    assert bar.bid == 101 and bar.ask == 101.02
