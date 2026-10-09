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
from pathlib import Path

from .train import DEFAULT, _merge, load_inputs, run_variant

TOP = 5


def grid(device="cpu", threads=1, seed=0):
    """The spec's search: 8 daily settings x 7 minute settings (56 variants)."""
    daily = [
        {"model": model, "k": k, "threshold_bp": threshold}
        for model, k, threshold in itertools.product(("lgbm", "mlp"), (3, 5), (0.0, 10.0))
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
    with open(out / "runs.jsonl", "a") as runs:
        for number, config in enumerate(configs, 1):
            config = {**config, "symbols": symbols}
            metrics = run_variant(
                config, data_root, out, trials=len(configs), inputs=inputs, cache=cache
            )
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
        "trials": len(results),
        "frozen": frozen,
        "best": {k: ranked[0][k] for k in ("variant", "daily_difference", "beats_holding")}
        if ranked
        else None,
        "any_beats_holding": any(r["beats_holding"] for r in results),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/ml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/ml-search"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--symbols", nargs="*", help="default: every trainable symbol")
    parser.add_argument("--limit", type=int, help="run only the first N variants (smoke test)")
    args = parser.parse_args(argv)
    configs = grid(args.device, args.threads)[: args.limit]
    summary = run_search(configs, args.data, args.out, symbols=args.symbols)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
