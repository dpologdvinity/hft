"""Measure one session through load -> simulation -> diagnostics -> rollout.

Each repetition runs in a fresh child process so peak RSS is that run's own
high-water mark rather than an earlier run's. The child also prints a
fingerprint of bars, gaps, decision eligibility and the always-long rollout
result, so two loader configurations can be checked for identical behavior.

Only pass development sessions. With --experiment, the session must be listed
in that experiment's development split; reserved final-test sessions are refused
before any partition is opened.

    .venv/bin/python benchmarks/session_pipeline.py \
        --dataset data/mcd/manifest.json --session 2026-06-04 \
        --experiment artifacts/mcd-v3/experiment.json --metadata execution --repeat 3
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rss_bytes():
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def _digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()[:16]


def child(args):
    sys.path.insert(0, str(ROOT))
    from hft.data import load_dataset, metadata_bytes, session_bytes
    from hft.execution import Simulation
    from hft.research import _rollout

    kwargs = {} if args.metadata == "all" else {"metadata": args.metadata}
    start = time.perf_counter()
    (session,) = load_dataset(args.dataset, session_ids=[args.session], **kwargs)
    load_seconds = time.perf_counter() - start
    rss_after_load = _rss_bytes()

    start = time.perf_counter()
    simulation = Simulation(session)
    simulation_seconds = time.perf_counter() - start

    # Same traversal as the development data-quality diagnostic.
    start = time.perf_counter()
    eligible = ineligible = 0
    for bar in simulation.bars[61:]:
        if simulation.decision_eligible:
            eligible += 1
        else:
            ineligible += 1
        simulation.advance_to(bar.end_ns)
    diagnostic_seconds = time.perf_counter() - start
    gaps = simulation.gap_count
    bars = simulation.bars
    del simulation

    rollout = {}
    rollout_seconds = None
    if args.rollout:
        start = time.perf_counter()
        rollout, _ = _rollout([session], lambda obs, env: 1)
        rollout_seconds = time.perf_counter() - start

    print(
        json.dumps(
            {
                "quotes": len(session.quote_ns),
                "trades": len(session.trade_ns),
                "retained_bytes": session_bytes(session),
                "metadata_bytes": metadata_bytes(session.manifest),
                "rss_after_load_bytes": rss_after_load,
                # Linux ru_maxrss is KiB.
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "load_seconds": load_seconds,
                "simulation_seconds": simulation_seconds,
                "diagnostic_seconds": diagnostic_seconds,
                "rollout_seconds": rollout_seconds,
                "fingerprint": {
                    "bars": len(bars),
                    "bars_sha": _digest(bars),
                    "gaps": gaps,
                    "eligible": eligible,
                    "ineligible": ineligible,
                    "rollout_sha": _digest(sorted(rollout.items())) if rollout else None,
                },
            }
        )
    )


def check_development(args):
    if args.experiment is None:
        return
    experiment = json.loads(Path(args.experiment).read_text())
    if args.session in experiment.get("final_test", []):
        raise SystemExit(f"{args.session} is a reserved final-test session; refusing to load it")
    if args.session not in experiment.get("development", []):
        raise SystemExit(f"{args.session} is not in the experiment's development split")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--metadata", default="all")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--rollout", action="store_true", help="also time an always-long rollout")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    check_development(args)
    command = [sys.executable, __file__, "--child", "--dataset", args.dataset]
    command += ["--session", args.session, "--metadata", args.metadata]
    command += ["--rollout"] if args.rollout else []
    runs = []
    for _ in range(args.repeat):
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        runs.append(json.loads(output.strip().splitlines()[-1]))
    if len({json.dumps(r["fingerprint"], sort_keys=True) for r in runs}) != 1:
        raise SystemExit("nondeterministic fingerprint across repetitions")
    timed = ("load_seconds", "simulation_seconds", "diagnostic_seconds", "rollout_seconds")
    sized = ("retained_bytes", "metadata_bytes", "rss_after_load_bytes", "peak_rss_bytes")
    summary = {
        "dataset": args.dataset,
        "session": args.session,
        "metadata": args.metadata,
        "repeat": args.repeat,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "quotes": runs[0]["quotes"],
        "trades": runs[0]["trades"],
        "fingerprint": runs[0]["fingerprint"],
        "median": {
            key: statistics.median(r[key] for r in runs)
            for key in timed + sized
            if runs[0][key] is not None
        },
        "runs": runs,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
