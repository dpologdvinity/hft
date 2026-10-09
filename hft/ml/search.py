"""Train many ML day-trader variants and freeze the best few before the final test.

Every variant is scored on the validation years only, by simulated trading against
holding the same symbols. Daily models, minute models and minute signals are shared
between variants that use the same settings, so the grid costs a few model fits plus
many cheap scorings. Every variant tried is logged (the deflated Sharpe ratio uses
the count), and the top five by validation edge over holding are written to
`frozen.json`, the only variants the one-time test may score.

    .venv/bin/python -m hft.ml.search --data data/ml --out artifacts/ml-search --device cuda
"""

import argparse
import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from .train import DEFAULT, FinalGrant, _merge, _sha256, code_commit, load_inputs, run_variant

TOP = 5
REGISTRY = Path(__file__).resolve().parents[2] / ".state" / "ml-final-test.json"


def grid(device="cpu", threads=1, seed=0):
    """12 daily settings x 7 minute settings (84 variants).

    Daily: predict the return (top k, with or without a cost gate) or the day's range
    (top k movers), each with trees or a network. Minute: none (hold the picks from
    the open) or one of three networks at two horizons.
    """
    daily = [
        {"model": model, "target": "return", "k": k, "gate": gate}
        for model, k, gate in itertools.product(("lgbm", "mlp"), (3, 5), (False, True))
    ] + [
        {"model": model, "target": "range", "k": k}
        for model, k in itertools.product(("lgbm", "mlp"), (3, 5))
    ]
    minute = [{"model": "none"}] + [
        {"model": model, "horizon": horizon}
        for model, horizon in itertools.product(("cnn", "gru", "transformer"), (15, 30))
    ]
    return [
        {"seed": seed, "device": device, "threads": threads, "daily": d, "minute": m}
        for d, m in itertools.product(daily, minute)
    ]


def run_search(configs, data_root, out, *, symbols=None, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(data_root, symbols)
    cache, results = {}, []
    log_path = out / "runs.jsonl"
    earlier = len(log_path.read_text().splitlines()) if log_path.exists() else 0
    trials = earlier + len(configs)  # every variant ever tried in this search folder
    with open(out / "runs.jsonl", "a") as runs:
        for number, config in enumerate(configs, 1):
            config = {**config, "symbols": symbols}
            metrics = run_variant(config, data_root, out, trials=trials, inputs=inputs, cache=cache)
            row = {"variant": metrics["variant"], "config": _merge(DEFAULT, config), **metrics}
            runs.write(json.dumps(row) + "\n")
            runs.flush()
            results.append(row)
            log(
                f"{number:>3}/{len(configs)} {row['variant']} daily {config['daily']} minute "
                f"{config['minute']}: edge {metrics['daily_difference'] * 1e4:+.2f} bp/day, "
                f"strategy {metrics['strategy'].get('total', 0):+.1%}, "
                f"holding {metrics['holding'].get('total', 0):+.1%}"
            )
    ranked = sorted(results, key=lambda r: r["daily_difference"], reverse=True)
    frozen = [r["variant"] for r in ranked[:TOP]]
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    summary = {
        "trials": trials,
        "frozen": frozen,
        "best": {k: ranked[0][k] for k in ("variant", "daily_difference", "beats_holding")}
        if ranked
        else None,
        "any_beats_holding": any(r["beats_holding"] for r in results),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_final(search_dir, data_root, out, *, registry=REGISTRY, log=print):
    """Score the frozen variants on the test split, once, from their saved weights.

    The run is recorded in `registry` (with the frozen list, the code version and the
    data fingerprint) before any test day is read, and a second run is refused. Each
    variant's interval is widened for the number of frozen variants (Bonferroni).
    """
    search_dir, registry = Path(search_dir), Path(registry)
    if registry.exists():
        raise RuntimeError(f"the final test was already run: {registry.read_text()}")
    frozen = json.loads((search_dir / "frozen.json").read_text())
    rows = {}
    for line in (search_dir / "runs.jsonl").read_text().splitlines():
        row = json.loads(line)
        rows[row["variant"]] = row
    for variant in frozen:
        for name, digest in rows[variant]["weights"].items():
            if _sha256(search_dir / variant / name) != digest:
                raise ValueError(f"saved weights of {variant} changed since the search: {name}")
    configs = [rows[v]["config"] for v in frozen]
    inputs = load_inputs(data_root, configs[0]["symbols"])
    entry = {
        "frozen": frozen,
        "code": code_commit(),
        "data_hash": inputs.data_hash,
        "started": datetime.now(UTC).isoformat(),
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(entry, indent=2) + "\n")  # the test counts as used now
    grant, confidence = FinalGrant(), 100 - 5 / len(frozen)
    results = []
    for variant, config in zip(frozen, configs, strict=True):
        metrics = run_variant(
            config,
            data_root,
            out,
            split="test",
            trials=rows[variant]["trials"],
            inputs=inputs,
            saved=search_dir / variant,
            grant=grant,
            confidence=confidence,
        )
        results.append(metrics)
        log(
            f"test {variant}: strategy {metrics['strategy'].get('total', 0):+.1%}, holding "
            f"{metrics['holding'].get('total', 0):+.1%}, beats holding: {metrics['beats_holding']}"
        )
    final = {**entry, "confidence": confidence, "results": results}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "final.json").write_text(json.dumps(final, indent=2) + "\n")
    registry.write_text(
        json.dumps({**entry, "finished": datetime.now(UTC).isoformat()}, indent=2) + "\n"
    )
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/ml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/ml-search"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--symbols", nargs="*", help="default: every trainable symbol")
    parser.add_argument("--limit", type=int, help="run only the first N variants (smoke test)")
    parser.add_argument(
        "--final", action="store_true", help="score the frozen variants on the test split, once"
    )
    args = parser.parse_args(argv)
    if args.final:
        print(json.dumps(run_final(args.out, args.data, args.out / "final"), indent=2))
        return 0
    configs = grid(args.device, args.threads)[: args.limit]
    summary = run_search(configs, args.data, args.out, symbols=args.symbols)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
