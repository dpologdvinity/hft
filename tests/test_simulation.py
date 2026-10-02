from decimal import Decimal as D

import numpy as np

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
