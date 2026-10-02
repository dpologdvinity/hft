"""Operator CLI. Import training and mutable broker code only in explicit commands."""

import argparse
import json
import os
import sys
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(
        description="Historical AI learning and risk-gated stock trading"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "smoke", help="offline synthetic engineering proof; never live eligible"
    )
    smoke.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    smoke.add_argument("--timesteps", type=int, default=2048)
    for name in ("data-probe", "download", "prepare", "record"):
        command = commands.add_parser(name)
        command.add_argument("--symbol", default="AAPL")
        command.add_argument("--feed", choices=["iex"], default="iex")
        if name == "data-probe":
            command.add_argument("--session", required=True)
        elif name == "download":
            command.add_argument("--start", required=True)
            command.add_argument("--end", required=True)
            command.add_argument("--output", type=Path, default=Path("data/cache"))
        elif name == "prepare":
            command.add_argument("--input", type=Path)
            command.add_argument("--bar-seconds", type=int, choices=[5], default=5)
        else:
            command.add_argument("--output", type=Path, default=Path("data/recordings"))
            command.add_argument("--duration", type=float)
    experiment = commands.add_parser("experiment")
    experiment.add_argument("--config", type=Path, default=Path("config/research.toml"))
    experiment.add_argument("--dataset", type=Path)
    experiment.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--experiment", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--experiment", type=Path, required=True)
    for name in ("replay", "dry-run", "broker-paper", "live"):
        command = commands.add_parser(name)
        command.add_argument("--model", type=Path)
        command.add_argument("--config", type=Path)
        command.add_argument("--capital", type=float)
        command.add_argument("--output", type=Path)
        if name == "replay":
            command.add_argument("--dataset", type=Path, required=True)
        else:
            command.add_argument("--duration", type=float)
        if name == "live":
            command.add_argument("--enable-live", action="store_true")
            command.add_argument("--max-live-notional", type=float)
            command.add_argument("--logs", type=Path, default=Path("logs/broker-paper"))
    status = commands.add_parser("status")
    status.add_argument("--state", type=Path, default=Path(".state"))
    status.add_argument("--logs", type=Path, default=Path("logs"))
    graduate = commands.add_parser("graduate")
    graduate.add_argument("--model", type=Path, required=True)
    graduate.add_argument("--logs", type=Path, required=True)
    graduate.add_argument("--calendar", type=Path)
    graduate.add_argument("--output", type=Path)
    return parser


def _read_credentials():
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_PAPER_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_PAPER_SECRET_KEY")
    if not key or not secret:
        raise ValueError("set ALPACA_API_KEY and ALPACA_SECRET_KEY for free read-only market data")


def _handle(args):
    if args.command == "live":
        if not args.enable_live:
            raise ValueError("live requires explicit --enable-live and validated paper evidence")
        if args.max_live_notional is None or not 0 < args.max_live_notional <= 10:
            raise ValueError("live requires --max-live-notional in (0,10] for the canary")
    if args.command == "data-probe":
        _read_credentials()
        from .history import probe_access

        return probe_access(args.symbol, args.session)
    if args.command == "download":
        _read_credentials()
        from .history import download_sessions

        return {"dataset": str(download_sessions(args.symbol, args.start, args.end, args.output))}
    # Handlers are added with their verified implementation in later plan tasks.
    raise ValueError(f"{args.command} is not implemented yet")


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        result = _handle(args)
        if result is not None:
            print(json.dumps(result, indent=2, allow_nan=False, default=str))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
