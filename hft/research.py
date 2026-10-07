"""Frozen chronological experiments, cost-matched controls and honest evidence gates."""

import hashlib
import json
import math
import os
import resource
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


def _session_manifest_hash(manifest):
    return canonical_hash(
        {k: v for k, v in manifest.items() if k not in ("quote_metadata", "trade_metadata")}
    )


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
    return _is_real_executable_manifest(session.manifest)


def _is_real_executable_manifest(manifest):
    provenance = manifest.get("provenance")
    return (
        manifest.get("synthetic") is False
        and manifest.get("feed") == "iex"
        and provenance in ("alpaca-historical-iex", "alpaca-realtime-iex")
        and (provenance != "alpaca-realtime-iex" or manifest.get("complete_session") is True)
    )


def freeze_experiment(dataset_manifest, config, output):
    from .data import AGGREGATION_VERSION, load_dataset_manifests
    from .features import FEATURE_VERSION
    from .training import validate_config

    source = Path(dataset_manifest).resolve()
    sessions = [manifest for _, manifest in load_dataset_manifests(source)]
    config = validate_config(config)
    if len({s["symbol"] for s in sessions}) > 1:
        raise ValueError("research requires a single stock symbol")
    ids = [s["session_id"] for s in sessions]
    result = {
        "dataset_manifest": str(source),
        "dataset_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
        "session_ids": ids,
        "config": config,
        "final_test_consumed": False,
        "feature_version": FEATURE_VERSION,
        "aggregation_version": AGGREGATION_VERSION,
        "symbol": sessions[0]["symbol"] if sessions else None,
        "synthetic": any(s.get("synthetic", False) for s in sessions),
        "feed": sessions[0].get("feed", "unknown") if sessions else "unknown",
        "real_executable_data": bool(sessions)
        and all(_is_real_executable_manifest(s) for s in sessions),
        "session_hashes": {s["session_id"]: _session_manifest_hash(s) for s in sessions},
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


def load_experiment(path):
    """Validate the frozen identity without opening any market partitions."""
    from .data import AGGREGATION_VERSION
    from .features import FEATURE_VERSION

    path = Path(path)
    manifest = json.loads(path.read_text())
    if (
        manifest.get("feature_version") != FEATURE_VERSION
        or manifest.get("aggregation_version") != AGGREGATION_VERSION
    ):
        raise ValueError("incompatible frozen execution contract; freeze a new experiment")
    identity = {k: v for k, v in manifest.items() if k != "experiment_hash"}
    identity["final_test_consumed"] = False
    if canonical_hash(identity) != manifest["experiment_hash"]:
        raise ValueError("frozen experiment identity changed")
    source = Path(manifest["dataset_manifest"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["dataset_hash"]:
        raise ValueError("dataset identity changed")
    return manifest | {"path": str(path)}


def _validate_development_partition(manifest):
    development = manifest.get("development", [])
    final_test = manifest.get("final_test", [])
    if (
        len(development) != len(set(development))
        or len(final_test) != len(set(final_test))
        or set(development) & set(final_test)
    ):
        raise ValueError("invalid development partition")


def load_phase_sessions(experiment, phase, *, resident_bytes=0):
    """Load one frozen phase, checking prices only when that phase is authorized."""
    from .data import checksum, load_dataset

    if phase not in ("development", "final_test"):
        raise ValueError("unknown research phase")
    _validate_development_partition(experiment)
    if phase == "final_test" and not experiment.get("final_test_consumed"):
        raise ValueError("final test must be reserved before loading prices")
    source = Path(experiment["dataset_manifest"])
    if checksum(source) != experiment["dataset_hash"]:
        raise ValueError("dataset identity changed")
    # Research only simulates; archive-only metadata stays on disk.
    sessions = load_dataset(
        source,
        session_ids=experiment[phase],
        resident_bytes=resident_bytes,
        metadata="execution",
    )
    if [s.session_id for s in sessions] != experiment[phase]:
        raise ValueError("research phase session order changed")
    for session in sessions:
        if (
            _session_manifest_hash(session.manifest)
            != experiment["session_hashes"][session.session_id]
        ):
            raise ValueError("session identity changed")
    return sessions


def preflight_experiment(path, *, deadline=float("inf"), inspect_all=False):
    """Check development decision coverage before fitting; never open final-test data.

    By default stop at the first unusable session. Explicit full inspection
    continues ordinary usability failures, preserving integrity and resource guards.
    This establishes data usability, never profitability or trading eligibility.
    """
    from .data import checksum, load_session, metadata_bytes, metadata_column_bytes, session_bytes
    from .execution import Simulation
    from .features import FEATURE_VERSION

    manifest = load_experiment(path)
    source = Path(manifest["dataset_manifest"])
    development = manifest.get("development", [])
    _validate_development_partition(manifest)
    entries = json.loads(source.read_text()).get("sessions", [])
    by_id = {entry["session_id"]: entry for entry in entries}
    if len(entries) != len(by_id) or any(date not in by_id for date in development):
        raise ValueError("invalid dataset session index")
    result = {
        "experiment_hash": manifest["experiment_hash"],
        "feature_version": FEATURE_VERSION,
        "aggregation_version": manifest["aggregation_version"],
        "status": "data-ready",
        "paper_eligible": False,
        "eligible_decisions": 0,
        "ineligible_decisions": 0,
        "feed_gaps": 0,
        "coverage_sessions": 0,
        "development_sessions": len(development),
        "coverage_complete": False,
        "inspection_mode": "all-development" if inspect_all else "stop-first-failure",
        "live_comparable": True,
        "sessions": [],
        "reasons": [] if development else ["insufficient-history"],
    }
    for date in development:
        if time.monotonic() >= deadline:
            raise TimeoutError("research wall-clock ceiling reached during data preflight")
        entry = by_id[date]
        target = (source.parent / entry["manifest"]).resolve()
        if not target.is_relative_to(source.parent.resolve()):
            raise ValueError("session manifest escapes dataset")
        if checksum(target) != entry["sha256"]:
            raise ValueError("session manifest checksum mismatch")
        session = load_session(target, metadata="execution")
        if (
            session.session_id != date
            or _session_manifest_hash(session.manifest) != manifest["session_hashes"][date]
        ):
            raise ValueError("session identity changed")
        row = {
            "date": date,
            "eligible_decisions": 0,
            "ineligible_decisions": 0,
            "feed_gaps": 0,
            "reasons": [],
        }
        if not is_real_executable(session) and not manifest.get("synthetic"):
            row["reasons"].append("unverified-data-provenance")
        try:
            simulation = Simulation(session)
        except ValueError as exc:
            if str(exc) != "insufficient session warmup":
                raise
            row["reasons"].append("insufficient-decision-warmup")
        else:
            for next_bar in simulation.bars[61:]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("research wall-clock ceiling reached during data preflight")
                key = (
                    "eligible_decisions" if simulation.decision_eligible else "ineligible_decisions"
                )
                row[key] += 1
                simulation.advance_to(next_bar.end_ns)
            row["feed_gaps"] = simulation.gap_count
            if row["feed_gaps"]:
                row["reasons"].append("runtime-feed-gap")
            if not row["eligible_decisions"]:
                row["reasons"].append("insufficient-decision-warmup")
            del simulation
        metadata = metadata_bytes(session.manifest)
        retained = session_bytes(session)
        row["memory"] = {
            "numeric_bytes": retained - metadata,
            "metadata_bytes": metadata,
            "retained_bytes": retained,
            "metadata_columns_bytes": metadata_column_bytes(session.manifest),
            # Byte counts cover the loaded execution view, not every archived column.
            "metadata_projection": session.metadata_projection,
            # Linux ru_maxrss is KiB; this is a process lifetime peak, not a session allocation.
            "process_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        }
        del session
        total = row["eligible_decisions"] + row["ineligible_decisions"]
        row["eligible_fraction"] = row["eligible_decisions"] / total if total else None
        result["reasons"].extend(row["reasons"])
        result["sessions"].append(row)
        result["coverage_sessions"] += 1
        for key in ("eligible_decisions", "ineligible_decisions", "feed_gaps"):
            result[key] += row[key]
        if row["reasons"] and not inspect_all:
            break
    total = result["eligible_decisions"] + result["ineligible_decisions"]
    result["eligible_fraction"] = result["eligible_decisions"] / total if total else None
    result["coverage_complete"] = result["coverage_sessions"] == len(development)
    result["reasons"] = list(dict.fromkeys(result["reasons"]))
    if result["reasons"]:
        result.update(status="insufficient-data", live_comparable=False)
    if time.monotonic() >= deadline:
        raise TimeoutError("research wall-clock ceiling reached during data preflight")
    atomic_json(Path(path).parent / "diagnostics.json", result)
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
    dates = []
    unresolved_position = 0.0
    eligible_decisions = ineligible_decisions = feed_gaps = 0
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
            if getattr(env.simulation, "decision_eligible", True):
                observations.append(obs.copy())
                action = int(policy(obs, env))
            else:
                action = 0
            obs, _, terminated, truncated, info = env.step(action)
            quote_marks = getattr(env.simulation, "equity_curve", [])
            curve.extend(quote_marks[quote_marks_seen:])
            quote_marks_seen = len(quote_marks)
            curve.append(float(info["equity"]))
            done = terminated or truncated
        previous_risk = env.simulation.risk.state_dict()
        daily.append(float(info["equity"]))
        dates.append(session.session_id)
        pnl.extend(float(t.pnl) for t in env.account.completed_trades)
        fills += len(env.account.fills)
        session_fills.append(len(env.account.fills))
        turnover += sum(abs(float(f.quantity)) * float(f.price) for f in env.account.fills)
        rejects += info.get("rejects", 0)
        unfilled += info.get("unfilled", 0)
        latencies.extend(info.get("latencies", []))
        incomplete |= info.get("incomplete", False)
        eligible_decisions += info.get("eligible_decisions", 0)
        ineligible_decisions += info.get("ineligible_decisions", 0)
        feed_gaps += info.get("feed_gaps", 0)
        unresolved_position = float(info.get("position", 0))
        env.close()
        if incomplete:
            break
    result = metrics(daily, pnl, curve)
    result.update(
        daily_equity=daily[1:],
        dates=dates,
        fills=fills,
        session_fills=session_fills,
        turnover=turnover,
        rejects=rejects,
        unfilled_orders=unfilled,
        incomplete=incomplete,
        live_comparable=feed_gaps == 0 and not incomplete,
        unresolved_position=unresolved_position,
        failure=(
            "incomplete-liquidation"
            if incomplete
            else "insufficient-decision-warmup"
            if eligible_decisions == 0
            else None
        ),
        eligible_decisions=eligible_decisions,
        ineligible_decisions=ineligible_decisions,
        feed_gaps=feed_gaps,
        latency_ms={
            str(p): float(np.percentile(latencies, p)) if latencies else None for p in (50, 95, 99)
        },
    )
    from .features import FEATURE_NAMES

    return result, np.asarray(observations, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))


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
        session_fills = dict(zip(agent["dates"], agent["session_fills"], strict=True))

        state = {}

        def random_policy(obs, env, rng=rng, session_fills=session_fills, state=state):
            # Randomize eligible decision times, then count actual executions. Entry
            # attempts that fail safety are never reported as matching fills.
            sid = env.session.session_id
            if sid not in state:
                target = session_fills.get(sid, 0)
                cycles = target // 2
                eligible = max(0, env.remaining_steps - 15)
                slots = np.linspace(0, eligible, cycles + 1, dtype=int)
                starts = [
                    int(rng.integers(slots[i], max(slots[i] + 1, slots[i + 1])))
                    for i in range(cycles)
                ]
                state[sid] = {
                    "start_index": env.index,
                    "starts": starts,
                    "exit_at": None,
                    "hold_bars": int(rng.integers(1, 6)),
                }
            current = state[sid]
            decision = env.index - current["start_index"]
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
    random_aligned = all(r["dates"] == agent["dates"] for r in random_results)
    random_mean = (
        np.mean([r["daily_returns"] for r in random_results], axis=0)
        if random_aligned
        else np.array([])
    )
    random_control = {
        "daily_returns": random_mean.tolist(),
        "seeds": list(range(seed, seed + 20)),
        "runs": random_results,
        "dates": agent["dates"] if random_aligned else [],
        "incomplete": any(r["incomplete"] for r in random_results),
        "matched_fill_frequency": random_aligned
        and all(r["session_fills"] == agent["session_fills"] for r in random_results),
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
            n: (
                paired_bootstrap(agent["daily_returns"], r["daily_returns"], seed)
                if not agent["incomplete"]
                and not r.get("incomplete", False)
                and agent["dates"] == r["dates"]
                else {
                    "positive": False,
                    "lower": None,
                    "upper": None,
                    "reason": "incomplete-or-unaligned-sessions",
                }
            )
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
        "incomplete-stress-liquidation": not report["stress"].get("incomplete", False),
        "runtime-feed-gap": policy.get("feed_gaps", 0) == 0 and policy.get("live_comparable", True),
        "runtime-stress-feed-gap": report["stress"].get("feed_gaps", 0) == 0
        and report["stress"].get("live_comparable", True),
        "incomplete-primary-control": not report.get(primary_control, {}).get("incomplete", False),
        "insufficient-decision-warmup": policy.get("eligible_decisions", 1) > 0,
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
