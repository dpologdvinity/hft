"""Loopback-only, read-only views of local research artifacts. No broker imports."""

import hashlib
import json
import math
import mimetypes
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from urllib.parse import unquote, urlsplit


def _json(path):
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("artifact exceeds dashboard limit")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("invalid artifact")
    return value


def _contained(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact escapes workspace")
    return path


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _checksum(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


class Dashboard:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._lock = RLock()
        self._market_key = None
        self._market = []
        self._paper_cache = {}

    def trading(self):
        """Paper trading runs per stock, from saved state and journal tails only."""
        from .trading.status import stock_rows

        base = self.root / ".state" / "trade"
        runs = sorted(p.name for p in base.glob("*") if p.is_dir()) if base.is_dir() else []
        return {
            "read_only": True,
            "runs": [{"name": name, "stocks": stock_rows(self.root, name)} for name in runs],
        }

    def _read(self, path):
        return _json(_contained(self.root, path))

    def _running(self, process, experiment):
        try:
            pid = process["pid"]
            if type(pid) is not int or pid <= 0:
                return False
            directory = Path("/proc") / str(pid)
            arguments = (directory / "cmdline").read_bytes().split(b"\0")
            return (
                (directory / "stat").read_text().split()[2] != "Z"
                and b"hft" in arguments
                and b"train" in arguments
                and str(experiment).encode() in arguments
            )
        except (OSError, KeyError):
            return False

    def _history(self, experiment, warnings):
        """Read development partitions only; never open held-out trade/quote data."""
        try:
            source = _contained(self.root, experiment["dataset_manifest"])
            if source.parts[-1] != "manifest.json":
                raise ValueError("invalid dataset source")
            if not any(source.is_relative_to(self.root / name) for name in ("data", "artifacts")):
                raise ValueError("dataset source is outside data directories")
            dataset = self._read(source)
            if hashlib.sha256(source.read_bytes()).hexdigest() != experiment["dataset_hash"]:
                raise ValueError("dataset hash mismatch")
            development = set(experiment.get("development", []))
            if development & set(experiment.get("final_test", [])):
                raise ValueError("development overlaps final test")
            selected, identities = [], []
            for entry in dataset.get("sessions", []):
                if entry["session_id"] not in development:
                    continue
                manifest_path = _contained(source.parent, source.parent / entry["manifest"])
                manifest = self._read(manifest_path)
                if _checksum(manifest_path) != entry["sha256"]:
                    raise ValueError("session manifest checksum mismatch")
                partition = manifest["partitions"]["trades"]
                path = _contained(manifest_path.parent, manifest_path.parent / partition["path"])
                stat = path.stat()
                identities.append(
                    (
                        entry["session_id"],
                        entry["sha256"],
                        partition["sha256"],
                        str(path),
                        stat.st_ino,
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ctime_ns,
                    )
                )
                selected.append((entry["session_id"], partition, path))
            if {date for date, _, _ in selected} != development or len(selected) != len(
                development
            ):
                raise ValueError("development sessions missing or duplicated")
            key = (
                experiment["dataset_hash"],
                tuple(experiment.get("development", [])),
                tuple(identities),
            )
            if key == self._market_key:
                return self._market
            import pyarrow.parquet as pq

            history = []
            for date, partition, path in selected:
                if _checksum(path) != partition["sha256"]:
                    raise ValueError("trade partition checksum mismatch")
                file = pq.ParquetFile(path)
                if file.num_row_groups:
                    prices = file.read_row_group(file.num_row_groups - 1, columns=["trade_price"])
                    if len(prices):
                        price = _number(prices["trade_price"][-1].as_py())
                        if price is not None and price > 0:
                            history.append({"date": date, "close": price})
            self._market_key, self._market = key, history
            return history
        except (OSError, ValueError, KeyError, TypeError):
            warnings.append("Development market history could not be verified or read.")
            return []

    def snapshot(self):
        with self._lock:
            return self._snapshot()

    def _paper_results(self, result):
        from .logs import iter_log
        from .metrics import EquitySummary

        paths = sorted(
            (self.root / "logs").glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime_ns, reverse=True
        )[:16]
        for path in paths:
            try:
                path = _contained(self.root / "logs", path)
                stat = path.stat()
                if stat.st_size > 128 * 1024**2:
                    result["warnings"].append(
                        "A paper journal exceeds the dashboard's lightweight read limit."
                    )
                    continue
                key = (str(path), stat.st_mtime_ns, stat.st_size)
                if key in self._paper_cache:
                    summary = self._paper_cache[key]
                else:
                    initial = equity = None
                    curve = None
                    sessions, pnl = [], []
                    source = None
                    for row in iter_log(path):
                        if row["event"] == "start":
                            if row.get("synthetic") is not False or row.get("source") not in (
                                "dry-run",
                                "broker-paper",
                            ):
                                source = None
                                break
                            source = row["source"]
                            account = row.get("account_state", {})
                            if float(account.get("position", 0)) == 0:
                                initial = float(account.get("cash", row["initial_cash"]))
                            curve = EquitySummary(initial) if initial is not None else None
                        elif source and row["event"] == "equity":
                            equity = float(row["equity"])
                            if curve is None:
                                curve = EquitySummary(equity)
                            else:
                                curve.add(equity)
                        elif source and row["event"] == "trade":
                            value = float(row["pnl"])
                            if not math.isfinite(value):
                                raise ValueError("invalid journal PnL")
                            pnl.append(value)
                        elif source and row["event"] == "session":
                            cash, position = float(row["equity"]), float(row["position"])
                            if not math.isfinite(cash) or not math.isfinite(position):
                                raise ValueError("invalid session cash or inventory")
                            # end_session records cash under the historical 'equity'
                            # key; with inventory it is not a marked equity value.
                            sessions.append(
                                {
                                    "date": row.get("date"),
                                    "complete": row.get("complete"),
                                    "cash": cash,
                                    "position": position,
                                    "source": source,
                                }
                            )
                            if position == 0:
                                equity = cash
                                if curve is None:
                                    curve = EquitySummary(equity)
                                else:
                                    curve.add(equity)
                    summary = (
                        None
                        if source is None
                        else {
                            "sessions": sessions[-30:],
                            "metrics": {
                                "sessions": len(sessions),
                                "completed_sessions": sum(s["complete"] is True for s in sessions),
                                "trades": len(pnl),
                                "initial_cash": initial,
                                "equity": equity,
                                "net_profit": equity - initial
                                if equity is not None and initial
                                else None,
                                "max_drawdown": curve.drawdown if curve else None,
                                "expectancy": sum(pnl) / len(pnl) if pnl else None,
                                "source": source,
                            },
                        }
                    )
                    self._paper_cache = {key: summary}
                if summary:
                    result["paper"].update(
                        **summary,
                        status="observed",
                        message="Latest recorded paper run; live graduation remains unproven.",
                    )
                    return
            except (OSError, ValueError, KeyError, TypeError):
                result["warnings"].append(
                    "A paper journal is incomplete or failed its integrity check."
                )

    def _snapshot(self):
        result = {
            "updated_at": datetime.now(UTC).isoformat(),
            "read_only": True,
            "dataset": None,
            "market_history": [],
            "runs": [],
            "warnings": [],
            "training": {
                "status": "idle",
                "message": "No frozen research experiment is available.",
                "completed_trials": 0,
                "total_trials": 0,
                "current_trial": None,
                "steps_per_trial": None,
                "steps_completed": None,
                "data_validated": False,
                "held_out_evaluated": False,
                "paper_eligible": False,
                "elapsed_seconds": None,
                "budget_seconds": None,
            },
            "paper": {
                "eligible": False,
                "status": "awaiting-validation",
                "metrics": None,
                "message": "A completed real-data evaluation is required before paper review.",
            },
        }
        paths = sorted(
            (self.root / "artifacts").glob("*/experiment.json"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )[:32]
        selected = False
        for path in paths:
            try:
                experiment = self._read(path)
                if experiment.get("synthetic") is not False:
                    continue
                cfg = experiment.get("config", {})
                report_path = path.parent / "search/research.json"
                report = self._read(report_path) if report_path.exists() else {}
                records = []
                for trial in (path.parent / "search").glob("*.json"):
                    value = self._read(trial)
                    if "key" in value and "checkpoint_hash" in value:
                        records.append(value)
                process_path = path.parent / "training-process.json"
                process = self._read(process_path) if process_path.exists() else {}
                active = self._running(process, path)
                status = (
                    "running"
                    if active
                    else report.get("status") or ("interrupted" if process else "idle")
                )
                messages = {
                    "running": "Training and comparing policies on development sessions.",
                    "interrupted": "Training was interrupted. Its final evaluation is unfinished.",
                    "incomplete-search": "The search stopped before all required work completed.",
                    "failed-edge": "The policy did not meet the research evidence requirements.",
                    "paper-eligible": "The research report passed. Paper evidence is still required.",
                    "idle": "The dataset is frozen and ready for a research run.",
                    "insufficient-data": "More historical sessions are needed for evaluation.",
                }
                if experiment.get("status") == "insufficient-data":
                    status = "insufficient-data"
                eligible = report.get("paper_eligible") is True
                reasons = report.get("eligibility", {}).get("reasons", [])
                details = {
                    k: cfg[k]
                    for k in ("initial_cash", "timesteps", "seeds", "n_envs", "wall_seconds")
                    if k in cfg
                }
                details["reasons"] = reasons
                details["completed_trials"] = len(records)
                diagnostics_path = path.parent / "diagnostics.json"
                coverage_passed = False
                if diagnostics_path.exists():
                    diagnostics = self._read(diagnostics_path)
                    if diagnostics.get("experiment_hash") != experiment.get("experiment_hash"):
                        raise ValueError("coverage experiment identity mismatch")
                    coverage_passed = (
                        diagnostics.get("status") == "data-ready"
                        and diagnostics.get("coverage_complete") is True
                        and diagnostics.get("live_comparable") is True
                    )
                    details["decision_coverage"] = {
                        k: diagnostics[k]
                        for k in (
                            "eligible_decisions",
                            "ineligible_decisions",
                            "feed_gaps",
                            "live_comparable",
                            "coverage_sessions",
                            "development_sessions",
                            "coverage_complete",
                            "feature_version",
                        )
                        if k in diagnostics
                    }
                if "runtime-feed-gap" in reasons:
                    messages["insufficient-data"] = (
                        "Development data exceeds the five-second feed gap limit. "
                        "Training is blocked; the final test remains reserved."
                    )
                if experiment.get("final_test_consumed") is True and isinstance(
                    report.get("test"), dict
                ):
                    test = report["test"]
                    policy = test.get("policy", {})
                    details["result"] = {
                        k: policy[k]
                        for k in (
                            "net_profit",
                            "return",
                            "max_drawdown",
                            "trades",
                            "expectancy",
                            "profit_factor",
                            "sessions",
                        )
                        if k in policy
                    }
                    primary = report.get("primary_control")
                    edge = test.get("edge", {}).get(primary, {})
                    details["edge"] = {"primary_control": primary} | {
                        k: edge[k] for k in ("lower", "upper", "positive") if k in edge
                    }
                result["runs"].append(
                    {
                        "id": path.parent.name,
                        "name": f"{experiment.get('symbol', 'Stock')} research",
                        "data": "Real IEX history",
                        "steps": _number(cfg.get("timesteps")),
                        "status": status,
                        "details": details,
                    }
                )
                if selected:
                    continue
                selected = True
                dates = experiment.get("session_ids", [])
                development = experiment.get("development", [])
                result["dataset"] = {
                    "symbol": experiment.get("symbol"),
                    "source": experiment.get("feed"),
                    "total_sessions": len(dates),
                    "development_sessions": len(development),
                    "final_test_sessions": len(experiment.get("final_test", [])),
                    "start_date": dates[0] if dates else None,
                    "end_date": dates[-1] if dates else None,
                    "development_start_date": development[0] if development else None,
                    "development_end_date": development[-1] if development else None,
                    "initial_cash": _number(cfg.get("initial_cash")),
                }
                budget_path = path.parent / "search/budget.json"
                budget = self._read(budget_path) if budget_path.exists() else {}
                elapsed = _number(budget.get("elapsed_seconds"))
                if active and _number(process.get("started_ns")) is not None:
                    elapsed = max(0, (time.time_ns() - process["started_ns"]) / 1e9)
                result["training"].update(
                    status=status,
                    message=messages.get(status, "Review the research run status."),
                    completed_trials=len(records),
                    total_trials=3 * len(cfg.get("seeds", [])) * len(experiment.get("folds", [])),
                    steps_per_trial=_number(cfg.get("timesteps")),
                    data_validated=coverage_passed,
                    held_out_evaluated=experiment.get("final_test_consumed") is True
                    and isinstance(report.get("test"), dict),
                    paper_eligible=eligible,
                    elapsed_seconds=elapsed,
                    budget_seconds=_number(cfg.get("wall_seconds")),
                )
                previous_warnings = len(result["warnings"])
                result["market_history"] = self._history(experiment, result["warnings"])
                if len(result["warnings"]) != previous_warnings:
                    result["training"]["data_validated"] = False
                if eligible:
                    result["paper"].update(
                        eligible=True,
                        status="unproven",
                        message="Research passed; broker-paper evidence is unproven.",
                    )
            except (OSError, ValueError, KeyError, TypeError):
                result["warnings"].append("A research artifact is incomplete or could not be read.")
        for path in (self.root / "artifacts").glob("*/preflight/result.json"):
            try:
                value = self._read(path)
                result["runs"].append(
                    {
                        "id": path.parent.parent.name + "-preflight",
                        "name": "Training preflight",
                        "data": "1 session",
                        "steps": _number(value.get("timesteps")),
                        "status": "completed" if value.get("fit_complete") else "incomplete-search",
                        "details": {
                            "paper_eligible": False,
                            "purpose": "Training pipeline check only",
                        },
                    }
                )
            except (OSError, ValueError):
                pass
        self._paper_results(result)
        return result


def create_server(root, port=8765):
    dashboard = Dashboard(root)
    assets = Path(root).resolve() / "frontend/dist"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, code, body, content_type="text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _local(self):
            allowed = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in allowed and (
                not origin
                or origin in {"http://" + host for host in allowed}
                or origin in ("http://127.0.0.1:5173", "http://localhost:5173")
            )

        def do_GET(self):
            if not self._local():
                return self._reply(403, b"Local requests only")
            route = unquote(urlsplit(self.path).path)
            if route == "/api/trading":
                try:
                    body = json.dumps(dashboard.trading(), allow_nan=False).encode()
                    return self._reply(200, body, "application/json")
                except (OSError, ValueError, TypeError, KeyError):
                    return self._reply(
                        503,
                        b'{"error":"Paper trading state is temporarily unavailable."}',
                        "application/json",
                    )
            if route == "/api/dashboard":
                try:
                    body = json.dumps(dashboard.snapshot(), allow_nan=False).encode()
                    return self._reply(200, body, "application/json")
                except (OSError, ValueError, TypeError):
                    return self._reply(
                        503,
                        b'{"error":"Local research data is temporarily unavailable."}',
                        "application/json",
                    )
            try:
                path = _contained(
                    assets, assets / ("index.html" if route == "/" else route.lstrip("/"))
                )
                if not path.is_file():
                    return self._reply(404, b"Not found")
                return self._reply(
                    200,
                    path.read_bytes(),
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                )
            except (OSError, ValueError):
                return self._reply(404, b"Not found")

        def do_POST(self):
            self._reply(405, b"Dashboard is read-only")

        do_PUT = do_DELETE = do_PATCH = do_POST

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(root, port=8765):
    if not (Path(root) / "frontend/dist/index.html").is_file():
        raise ValueError("build the dashboard first: cd frontend && npm ci && npm run build")
    server = create_server(root, port)
    print(f"Dashboard: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
