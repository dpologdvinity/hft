"""Bounded, resumable offline PPO search; runtime never imports this module."""

import fcntl
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np

from .research import (
    CANDIDATES,
    atomic_json,
    canonical_hash,
    eligibility,
    evaluate,
    is_real_executable,
)

TEST_REGISTRY = Path(__file__).resolve().parents[1] / ".state" / "research-final-tests.json"

DEFAULTS = {
    "timesteps": 100000,
    "seeds": [42, 43],
    "n_envs": 4,
    "wall_seconds": 7200,
    "initial_cash": 500.0,
}


def reserve_final_test(experiment):
    """Consume real test identities under one project-wide serialized registry."""
    if experiment.get("synthetic"):
        return
    TEST_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        canonical_hash(
            {"symbol": experiment["symbol"], "feed": experiment["feed"], "session_id": sid}
        )
        for sid in experiment["final_test"]
    ]
    with TEST_REGISTRY.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry = json.loads(TEST_REGISTRY.read_text()) if TEST_REGISTRY.exists() else {}
        if any(key in registry for key in keys):
            raise ValueError("final test sessions already consumed; reserve later unseen dates")
        for key, sid in zip(keys, experiment["final_test"], strict=True):
            registry[key] = {
                "symbol": experiment["symbol"],
                "feed": experiment["feed"],
                "session_id": sid,
                "session_hash": experiment.get("session_hashes", {}).get(sid),
                "dataset_hash": experiment.get("dataset_hash"),
                "experiment_hash": experiment["experiment_hash"],
            }
        atomic_json(TEST_REGISTRY, registry)


def validate_config(config):
    if set(config) - set(DEFAULTS):
        raise ValueError("unknown research configuration keys")
    result = DEFAULTS | config
    for key in ("timesteps", "n_envs", "wall_seconds", "initial_cash"):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("invalid research budget")
    if (
        int(result["timesteps"]) != result["timesteps"]
        or int(result["n_envs"]) != result["n_envs"]
        or result["n_envs"] > 4
        or result["wall_seconds"] > 7200
    ):
        raise ValueError("research budget exceeds ceiling or requires integers")
    if (
        not isinstance(result["seeds"], list)
        or not result["seeds"]
        or any(type(s) is not int for s in result["seeds"])
    ):
        raise ValueError("invalid seeds")
    if len(set(result["seeds"])) != len(result["seeds"]):
        raise ValueError("seeds must be independent unique values")
    return result


def trial_key(experiment, fold, seed, settings):
    from .data import AGGREGATION_VERSION
    from .features import FEATURE_NAMES, FEATURE_VERSION

    return canonical_hash(
        {
            "experiment": experiment,
            "fold": fold,
            "seed": seed,
            "settings": settings,
            "feature_version": FEATURE_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "aggregation_version": AGGREGATION_VERSION,
        }
    )


def select_champion(trials):
    eligible = []
    for candidate in sorted({t["candidate"] for t in trials}):
        rows = [t for t in trials if t["candidate"] == candidate]
        if any(
            t["validation"]["max_drawdown"] >= 0.05
            or t["validation"].get("incomplete", False)
            or t["validation"].get("eligible_decisions", 1) == 0
            or not t["validation"].get("live_comparable", True)
            or t["validation"].get("feed_gaps", 0) > 0
            for t in rows
        ):
            continue
        eligible.append(
            (
                float(np.median([t["validation"]["return"] for t in rows])),
                -float(np.median([t["validation"]["turnover"] for t in rows])),
                candidate,
            )
        )
    if not eligible:
        raise ValueError(
            "no complete validation candidate has eligible decisions and meets drawdown limit"
        )
    candidate = max(eligible)[2]
    seeds = {t["seed"] for t in trials if t["candidate"] == candidate}
    seed = max(
        seeds,
        key=lambda s: (
            np.median(
                [
                    t["validation"]["return"]
                    for t in trials
                    if t["candidate"] == candidate and t["seed"] == s
                ]
            ),
            -np.median(
                [
                    t["validation"]["turnover"]
                    for t in trials
                    if t["candidate"] == candidate and t["seed"] == s
                ]
            ),
            -s,
        ),
    )
    return candidate, seed


def fit_candidate(training_sessions, settings, seed, budget):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from .env import TradingEnv

    torch.set_num_threads(2)
    deadline = budget.get("deadline", float("inf"))

    class Deadline(BaseCallback):
        def _on_step(self):
            return (
                time.monotonic() < deadline
                and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 8 * 1024 * 1024
            )

    env = DummyVecEnv(
        [
            lambda: TradingEnv(
                training_sessions,
                initial_cash=budget.get("initial_cash", 500),
                random_start=True,
                episode_steps=256,
            )
            for _ in range(budget.get("n_envs", 4))
        ]
    )
    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=settings["learning_rate"],
            ent_coef=settings["entropy_coefficient"],
            n_steps=256,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
            seed=seed,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=budget["timesteps"], callback=Deadline())
        model.fit_complete = (
            model.num_timesteps >= budget["timesteps"] and time.monotonic() < deadline
        )
        model.peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return model
    finally:
        env.close()


def _save_checkpoint(model, path):
    temporary = path.with_name(path.stem + ".tmp.zip")
    model.save(str(temporary))
    with temporary.open("rb") as file:
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _search_impl(sessions, experiment, output, resume=False, smoke=False):
    from .policy import export_bundle

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = validate_config(experiment["config"])
    deadline = experiment.get("_deadline", time.monotonic() + cfg["wall_seconds"])
    by_id = {s.session_id: s for s in sessions}
    trials = []
    folds = experiment["folds"]
    settings = [{"learning_rate": lr, "entropy_coefficient": ec} for lr, ec in CANDIDATES]
    if smoke:
        settings = settings[: experiment.get("candidates", 1)]
    for candidate, params in enumerate(settings):
        for seed in cfg["seeds"]:
            for fold_index, fold in enumerate(folds):
                key = trial_key(experiment["experiment_hash"], fold, seed, params)
                record = output / (key + ".json")
                checkpoint = output / (key + ".zip")
                if resume and record.exists():
                    saved = json.loads(record.read_text())
                    if (
                        saved["key"] != key
                        or saved["checkpoint_hash"]
                        != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    ):
                        raise ValueError("trial checkpoint identity mismatch")
                    trials.append(saved)
                    continue
                if time.monotonic() >= deadline:
                    return _partial(output, trials)
                model = fit_candidate(
                    [by_id[i] for i in fold["train"]], params, seed, cfg | {"deadline": deadline}
                )
                if not model.fit_complete:
                    return _partial(output, trials)
                validation, _ = evaluate(
                    [by_id[i] for i in fold["validation"]],
                    lambda o, model=model: int(model.predict(o, deterministic=True)[0]),
                    initial_cash=cfg["initial_cash"],
                    seed=seed,
                    stress=False,
                    deadline=deadline,
                )
                _save_checkpoint(model, checkpoint)
                saved = {
                    "key": key,
                    "candidate": candidate,
                    "seed": seed,
                    "fold": fold_index,
                    "settings": params,
                    "validation": validation["policy"],
                    "controls": {
                        n: validation[n]["net_profit"]
                        for n in ("intraday_long", "ema_5_20", "random")
                    },
                    "requested_steps": cfg["timesteps"],
                    "actual_steps": model.num_timesteps,
                    "peak_rss_bytes": model.peak_rss_bytes,
                    "checkpoint_hash": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                }
                atomic_json(record, saved)
                trials.append(saved)
    try:
        candidate, seed = select_champion(trials)
    except ValueError as exc:
        report = {
            "status": "failed-edge",
            "eligibility": {"passed": False, "reasons": [str(exc)]},
            "trials": trials,
        }
        atomic_json(output / "research.json", report)
        return report
    if experiment.get("final_test_consumed"):
        if (output / "research.json").exists():
            return json.loads((output / "research.json").read_text())
        raise ValueError("final test already consumed; reserve later unseen dates")
    if time.monotonic() >= deadline:
        return _partial(output, trials)
    model = fit_candidate(
        [by_id[i] for i in experiment["development"]],
        settings[candidate],
        seed,
        cfg | {"deadline": deadline},
    )
    if not model.fit_complete:
        return _partial(output, trials)
    checkpoint = output / "selected.zip"
    _save_checkpoint(model, checkpoint)
    primary = max(
        ("intraday_long", "ema_5_20", "random"),
        key=lambda n: np.median(
            [t["controls"][n] for t in trials if t["candidate"] == candidate and t["seed"] == seed]
        ),
    )
    if not smoke:
        reserve_final_test(experiment)
    experiment["final_test_consumed"] = True
    if experiment.get("path"):
        atomic_json(
            experiment["path"],
            {k: v for k, v in experiment.items() if k not in ("path", "_deadline")},
        )
    test, observations = evaluate(
        [by_id[i] for i in experiment["final_test"]],
        lambda o, model=model: int(model.predict(o, deterministic=True)[0]),
        initial_cash=cfg["initial_cash"],
        seed=seed,
        primary_control=primary,
        deadline=deadline,
    )
    gate = eligibility(
        test,
        synthetic=experiment["synthetic"],
        primary_control=primary,
        real_executable_data=experiment.get("real_executable_data", False),
    )
    report = {
        "status": "research-only"
        if smoke
        else ("paper-eligible" if gate["passed"] else "failed-edge"),
        "synthetic": experiment["synthetic"],
        "paper_eligible": gate["passed"] and not smoke,
        "symbol": experiment["symbol"],
        "experiment_hash": experiment["experiment_hash"],
        "dataset_hash": experiment.get("dataset_hash"),
        "selected_candidate": candidate,
        "selected_seed": seed,
        "primary_control": primary,
        "test_days": len(experiment["final_test"]),
        "trials": trials,
        "test": test,
        "eligibility": gate,
        "ready_for_paper_review": gate["passed"] and not smoke,
        "smoke": smoke,
    }
    atomic_json(output / "research.json", report)
    export_bundle(
        checkpoint, experiment | {"research": report}, output / "bundle", observations=observations
    )
    return report


def _search(sessions, experiment, output, resume=False, smoke=False, setup_seconds=0.0):
    output = Path(output)
    budget_path = output / "budget.json"
    config = validate_config(experiment["config"])
    used = 0.0
    if resume and budget_path.exists():
        saved = json.loads(budget_path.read_text())
        if saved["experiment_hash"] != experiment["experiment_hash"]:
            raise ValueError("search budget experiment identity changed")
        used = saved["elapsed_seconds"]
    used += setup_seconds
    start = time.monotonic()
    experiment["_deadline"] = start + max(0.0, config["wall_seconds"] - used)
    try:
        return _search_impl(sessions, experiment, output, resume, smoke)
    except TimeoutError:
        trials = []
        for record in output.glob("*.json"):
            saved = json.loads(record.read_text())
            if "key" in saved and "checkpoint_hash" in saved:
                trials.append(saved)
        return _partial(output, trials)
    finally:
        experiment.pop("_deadline", None)
        atomic_json(
            budget_path,
            {
                "experiment_hash": experiment["experiment_hash"],
                "elapsed_seconds": used + time.monotonic() - start,
                "wall_seconds": config["wall_seconds"],
            },
        )


def _partial(output, trials):
    result = {
        "status": "incomplete-search",
        "trials": trials,
        "eligibility": {"passed": False, "reasons": ["incomplete-search"]},
    }
    atomic_json(output / "research.json", result)
    return result


def run_search(experiment, output=None, resume=False):
    from .data import load_dataset
    from .research import load_experiment, preflight_experiment

    started = time.monotonic()
    path = Path(experiment)
    manifest = load_experiment(path)
    if manifest["status"] == "insufficient-data":
        return manifest
    output = Path(output or path.parent / "search")
    report_path = output / "research.json"
    if manifest.get("final_test_consumed") and report_path.exists():
        completed = json.loads(report_path.read_text())
        if completed.get("status") in (
            "paper-eligible",
            "failed-edge",
            "research-only",
        ) and isinstance(completed.get("test"), dict):
            if completed.get("experiment_hash") != manifest["experiment_hash"]:
                raise ValueError("completed report experiment identity changed")
            return completed
    config = validate_config(manifest["config"])
    used = 0.0
    budget_path = output / "budget.json"
    if resume and budget_path.exists():
        budget = json.loads(budget_path.read_text())
        if budget["experiment_hash"] != manifest["experiment_hash"]:
            raise ValueError("search budget experiment identity changed")
        used = budget["elapsed_seconds"]
    deadline = started + max(0.0, config["wall_seconds"] - used)

    def record_setup():
        atomic_json(
            budget_path,
            {
                "experiment_hash": manifest["experiment_hash"],
                "elapsed_seconds": used + time.monotonic() - started,
                "wall_seconds": config["wall_seconds"],
            },
        )

    def trials():
        records = [json.loads(p.read_text()) for p in output.glob("*.json")]
        return [r for r in records if "key" in r and "checkpoint_hash" in r]

    source = Path(manifest["dataset_manifest"])
    try:
        if manifest.get("synthetic") is False and not manifest.get("final_test_consumed"):
            coverage = preflight_experiment(path, deadline=deadline)
            if coverage["status"] != "data-ready":
                result = {
                    "status": "insufficient-data",
                    "experiment_hash": manifest["experiment_hash"],
                    "paper_eligible": False,
                    "trials": trials(),
                    "eligibility": {"passed": False, "reasons": coverage["reasons"]},
                }
                atomic_json(output / "research.json", result)
                record_setup()
                return result
        if time.monotonic() >= deadline:
            raise TimeoutError("research wall-clock ceiling reached during setup")
        sessions = load_dataset(source)
        if time.monotonic() >= deadline:
            raise TimeoutError("research wall-clock ceiling reached during dataset loading")
    except TimeoutError:
        record_setup()
        return _partial(output, trials())
    return _search(sessions, manifest, output, resume, setup_seconds=time.monotonic() - started)


def train(
    sessions, output, timesteps=2048, candidates=1, seed=42, initial_cash=500, smoke=False, **kwargs
):
    from .data import AGGREGATION_VERSION
    from .features import FEATURE_VERSION
    from .research import make_folds, temporal_split

    if kwargs.keys() - {"n_envs"}:
        raise ValueError("unknown training arguments")
    if candidates not in (1, 2, 3):
        raise ValueError("candidate count must be 1 through 3")
    sessions = list(sessions)
    ids = [s.session_id for s in sessions]
    synthetic = bool(sessions) and all(s.manifest.get("synthetic", False) for s in sessions)
    if smoke:
        if not synthetic:
            raise ValueError("smoke sufficiency bypass only supports labeled synthetic data")
        if len(ids) < 3:
            raise ValueError("smoke requires at least 3 sessions")
        dev, test = ids[:-1], ids[-1:]
        folds = [{"train": dev[:-1], "validation": dev[-1:]}]
    else:
        dev, test = temporal_split(ids)
        folds = [
            {"train": list(f.train), "validation": list(f.validation)} for f in make_folds(dev)
        ]
    config = validate_config(
        {
            "timesteps": timesteps,
            "seeds": [seed] if smoke else [seed, seed + 1],
            "initial_cash": initial_cash,
            "n_envs": kwargs.get("n_envs", 1 if smoke else 4),
        }
    )
    experiment = {
        "development": dev,
        "final_test": test,
        "folds": folds,
        "synthetic": synthetic,
        "symbol": sessions[0].symbol,
        "feed": sessions[0].manifest.get("feed", "synthetic"),
        "real_executable_data": all(is_real_executable(s) for s in sessions),
        "config": config,
        "candidates": candidates,
        "session_ids": ids,
        "feature_version": FEATURE_VERSION,
        "aggregation_version": AGGREGATION_VERSION,
    }
    experiment["experiment_hash"] = canonical_hash(experiment)
    return _search(sessions, experiment, output, smoke=smoke)
