"""Frozen chronological experiments, cost-matched controls and honest evidence gates."""

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import metrics, paired_bootstrap

CANDIDATES = ((3e-4, 0.0), (1e-4, 0.01), (5e-4, 0.01))


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class Fold:
    train: tuple[str, ...]
    validation: tuple[str, ...]


def make_folds(session_ids):
    ids = tuple(session_ids)
    initial = math.floor(0.6 * len(ids))
    block = (len(ids) - initial) // 3
    if initial < 30 or block < 5:
        raise ValueError("insufficient-history: need 30 initial train and 5 per validation fold")
    return tuple(
        Fold(
            ids[: initial + i * block],
            ids[initial + i * block : initial + (i + 1) * block]
            if i < 2
            else ids[initial + 2 * block :],
        )
        for i in range(3)
    )


def temporal_split(session_ids):
    ids = list(session_ids)
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("session dates must be unique and chronological")
    ntest = max(30, math.ceil(0.25 * len(ids)))
    dev, test = ids[:-ntest], ids[-ntest:]
    if len(test) < 30:
        raise ValueError("insufficient-history: need 30 test sessions")
    make_folds(dev)
    return dev, test


def is_real_executable(session):
    manifest = session.manifest
    provenance = manifest.get("provenance")
    return (
        manifest.get("synthetic") is False
        and manifest.get("feed") == "iex"
        and provenance in ("alpaca-historical-iex", "alpaca-realtime-iex")
        and (provenance != "alpaca-realtime-iex" or manifest.get("complete_session") is True)
    )


def freeze_experiment(dataset_manifest, config, output):
    from .data import load_dataset
    from .training import validate_config

    source = Path(dataset_manifest).resolve()
    sessions = load_dataset(source)
    config = validate_config(config)
    if len({s.symbol for s in sessions}) > 1:
        raise ValueError("research requires a single stock symbol")
    ids = [s.session_id for s in sessions]
    result = {
        "dataset_manifest": str(source),
        "dataset_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
        "session_ids": ids,
        "config": config,
        "final_test_consumed": False,
        "symbol": sessions[0].symbol if sessions else None,
        "synthetic": any(s.manifest.get("synthetic", False) for s in sessions),
        "feed": sessions[0].manifest.get("feed", "unknown") if sessions else "unknown",
        "real_executable_data": bool(sessions) and all(is_real_executable(s) for s in sessions),
        "session_hashes": {
            s.session_id: canonical_hash(
                {
                    k: v
                    for k, v in s.manifest.items()
                    if k not in ("quote_metadata", "trade_metadata")
                }
            )
            for s in sessions
        },
    }
    try:
        dev, test = temporal_split(ids)
        result.update(
            status="frozen",
            development=dev,
            final_test=test,
            folds=[
                {"train": list(f.train), "validation": list(f.validation)} for f in make_folds(dev)
            ],
        )
    except ValueError as exc:
        result.update(
            status="insufficient-data", reasons=[str(exc)], development=[], final_test=[], folds=[]
        )
    result["experiment_hash"] = canonical_hash(result)
    output = Path(output)
    if output.exists():
        raise FileExistsError("experiment already frozen")
    atomic_json(output, result)
    return result


def ema(values, span):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.array([])
    result = np.empty(len(values))
    result[0] = values[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def _rollout(sessions, policy, costs=None, initial_cash=500, latency_ms=75, deadline=float("inf")):
    from .env import TradingEnv

    daily = [float(initial_cash)]
    curve = [float(initial_cash)]
    pnl = []
    observations = []
    fills = rejects = unfilled = 0
    turnover = 0.0
    latencies = []
    session_fills = []
    incomplete = False
    previous_risk = None
    for session in sessions:
        env = TradingEnv(
            session, initial_cash=daily[-1], costs=costs, episode_steps=10**9, latency_ms=latency_ms
        )
        obs, _ = env.reset()
        if previous_risk is not None:
            env.simulation.risk.load_state(previous_risk)
            env.simulation.risk._atr_bar = None
            env.simulation.risk.observe(env.simulation.snapshot(), env.simulation.now_ns)
            env.peak = float(env.simulation.risk.peak)
            env.last_drawdown = max(0.0, env.peak - daily[-1])
            obs = env._observation()
        done = False
        quote_marks_seen = 0
        while not done:
            if time.monotonic() >= deadline:
                env.close()
                raise TimeoutError("research wall-clock ceiling reached")
            observations.append(obs.copy())
            action = int(policy(obs, env))
            obs, _, terminated, truncated, info = env.step(action)
            quote_marks = getattr(env.simulation, "equity_curve", [])
            curve.extend(quote_marks[quote_marks_seen:])
            quote_marks_seen = len(quote_marks)
            curve.append(float(info["equity"]))
            done = terminated or truncated
        previous_risk = env.simulation.risk.state_dict()
        daily.append(float(info["equity"]))
        pnl.extend(float(t.pnl) for t in env.account.completed_trades)
        fills += len(env.account.fills)
        session_fills.append(len(env.account.fills))
        turnover += sum(abs(float(f.quantity)) * float(f.price) for f in env.account.fills)
        rejects += info.get("rejects", 0)
        unfilled += info.get("unfilled", 0)
        latencies.extend(info.get("latencies", []))
        incomplete |= info.get("incomplete", False)
        env.close()
    result = metrics(daily, pnl, curve)
    result.update(
        daily_equity=daily[1:],
        dates=[s.session_id for s in sessions],
        fills=fills,
        session_fills=session_fills,
        turnover=turnover,
        rejects=rejects,
        unfilled_orders=unfilled,
        incomplete=incomplete,
        latency_ms={
            str(p): float(np.percentile(latencies, p)) if latencies else None for p in (50, 95, 99)
        },
    )
    return result, np.asarray(observations, dtype=np.float32)


def evaluate(
    sessions,
    policy,
    *,
    costs=None,
    initial_cash=500,
    seed=42,
    primary_control=None,
    stress=True,
    deadline=float("inf"),
):
    from .account import Costs

    costs = costs or Costs()
    agent, observations = _rollout(
        sessions, lambda obs, env: policy(obs), costs, initial_cash, deadline=deadline
    )
    cash, _ = _rollout(sessions, lambda obs, env: 0, costs, initial_cash, deadline=deadline)
    long, _ = _rollout(sessions, lambda obs, env: 1, costs, initial_cash, deadline=deadline)

    def crossover(obs, env):
        closes = [b.close for b in env.history]
        return int(ema(closes, 5)[-1] > ema(closes, 20)[-1])

    crossover_result, _ = _rollout(sessions, crossover, costs, initial_cash, deadline=deadline)
    random_results = []
    for random_seed in range(seed, seed + 20):
        rng = np.random.default_rng(random_seed)
        session_index = {s.session_id: i for i, s in enumerate(sessions)}

        state = {}

        def random_policy(obs, env, rng=rng, session_index=session_index, state=state):
            # Randomize eligible decision times, then count actual executions. Entry
            # attempts that fail safety are never reported as matching fills.
            sid = env.session.session_id
            if sid not in state:
                target = agent["session_fills"][session_index[sid]]
                cycles = target // 2
                eligible = max(0, env.remaining_steps - 15)
                slots = np.linspace(0, eligible, cycles + 1, dtype=int)
                starts = [
                    int(rng.integers(slots[i], max(slots[i] + 1, slots[i + 1])))
                    for i in range(cycles)
                ]
                state[sid] = {
                    "decision": 0,
                    "starts": starts,
                    "exit_at": None,
                    "hold_bars": int(rng.integers(1, 6)),
                }
            current = state[sid]
            decision = current["decision"]
            current["decision"] += 1
            if float(env.account.position) > 0:
                if current["exit_at"] is None:
                    if current["starts"]:
                        current["starts"].pop(0)
                    current["exit_at"] = decision + current["hold_bars"]
                return int(decision < current["exit_at"])
            current["exit_at"] = None
            if current["starts"] and decision >= current["starts"][0]:
                return 1
            return 0

        result, _ = _rollout(sessions, random_policy, costs, initial_cash, deadline=deadline)
        random_results.append(result)
    random_mean = np.mean([r["daily_returns"] for r in random_results], axis=0)
    random_control = {
        "daily_returns": random_mean.tolist(),
        "seeds": list(range(seed, seed + 20)),
        "runs": random_results,
        "matched_fill_frequency": all(
            r["session_fills"] == agent["session_fills"] for r in random_results
        ),
        "target_session_fills": agent["session_fills"],
        "net_profit": float(np.mean([r["net_profit"] for r in random_results])),
    }
    controls = {"intraday_long": long, "ema_5_20": crossover_result, "random": random_control}
    result = {
        "policy": agent,
        "cash": cash,
        "stock_return_context": {
            "return": float(sessions[-1].trade_price[-1] / sessions[0].trade_price[0] - 1),
            "risk_matched": False,
        },
        **controls,
        "edge": {
            n: paired_bootstrap(agent["daily_returns"], r["daily_returns"], seed)
            for n, r in controls.items()
        },
    }
    if primary_control:
        result["primary_control"] = primary_control
    if stress:
        result["stress"], _ = _rollout(
            sessions,
            lambda obs, env: policy(obs),
            Costs(costs.commission * 2, 3),
            initial_cash,
            250,
            deadline=deadline,
        )
    return result, observations


def eligibility(report, *, synthetic, primary_control, real_executable_data=None):
    real_executable_data = not synthetic if real_executable_data is None else real_executable_data
    policy = report["policy"]
    pf = policy["profit_factor"]
    reasons = []

    def finite(value):
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )

    finite_statistics = all(
        finite(policy.get(key))
        for key in ("net_profit", "expectancy", "max_drawdown", "gross_profit", "gross_loss")
    )
    finite_statistics = finite_statistics and finite(report["stress"].get("net_profit"))
    finite_statistics = finite_statistics and all(
        type(policy.get(key)) is int and policy[key] >= 0 for key in ("sessions", "trades")
    )
    edge = report["edge"][primary_control]
    lower = edge.get("lower")
    checks = {
        "invalid-metric-samples": finite_statistics,
        "synthetic-data": not synthetic,
        "unverified-data-provenance": real_executable_data,
        "insufficient-test-sessions": policy["sessions"] >= 30,
        "insufficient-completed-trades": policy["trades"] >= 100,
        "nonpositive-expectancy": policy["expectancy"] > 0,
        "nonpositive-profit": policy["net_profit"] > 0,
        "profit-factor-below-1.3": (
            pf == "infinite"
            and finite(policy.get("gross_profit"))
            and policy["gross_profit"] > 0
            and policy.get("gross_loss") == 0
        )
        or (finite(pf) and pf >= 1.3),
        "drawdown-limit": policy["max_drawdown"] < 0.05,
        "nonpositive-stress-profit": report["stress"]["net_profit"] > 0,
        "failed-control-edge": edge.get("positive") is True and finite(lower) and lower > 0,
        "incomplete-liquidation": not policy.get("incomplete", False),
        "unmatched-random-control": report.get("random", {}).get("matched_fill_frequency", True),
    }
    reasons = [reason for reason, passed in checks.items() if not passed]
    return {
        "passed": not reasons,
        "reasons": reasons,
        "checks": checks,
        "real_executable_data": bool(real_executable_data),
    }


def run_search(experiment, output=None, resume=False):
    from .training import run_search as run

    return run(experiment, output, resume)


def train(sessions, output, **kwargs):
    from .training import train as run

    return run(sessions, output, **kwargs)
