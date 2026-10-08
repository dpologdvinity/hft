"""`hft trade --status`: per-stock state of paper runs from local files only."""

import json
from decimal import Decimal
from pathlib import Path


def _latest_journal(log_dir: Path, symbol: str):
    files = sorted(log_dir.glob(f"{symbol}-*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def stock_rows(root: Path, name: str):
    """Rows of per-stock status for one run, built from state and journal tails."""
    from .command import tail_rows

    state_dir, log_dir = root / ".state" / "trade" / name, root / "logs" / "trade" / name
    states = sorted(state_dir.glob("*.json"))
    if not states:
        return []
    books = json.loads(states[0].read_text())["state"]["books"]
    rows = []
    for symbol, book in books.items():
        account, risk = book["account"], book["risk"]
        position, cash = Decimal(account["position"]), Decimal(account["cash"])
        budget = Decimal(account["initial_cash"])
        journal = _latest_journal(log_dir, symbol)
        tail = tail_rows(journal) if journal else []
        equity = next(
            (Decimal(str(r["equity"])) for r in reversed(tail) if r.get("event") == "equity"), cash
        )
        status = "trading"
        if risk.get("halted"):
            status = f"halted ({risk['halted']})"
        elif any(r.get("event") == "finish" for r in tail[-3:]):
            status = "stopped"
        elif tail and tail[-1].get("event") == "gap":
            status = f"gap ({tail[-1].get('reason')})"
        rows.append(
            {
                "symbol": symbol,
                "status": status,
                "budget": str(budget),
                "position": str(position.normalize()),
                "cash": str(cash.quantize(Decimal("0.01"))),
                "equity": str(equity.quantize(Decimal("0.01"))),
                "profit": str((equity - budget).quantize(Decimal("0.01"))),
                "pending": book["pending"] is not None,
                "updated_ns": tail[-1].get("wall_ns") if tail else None,
            }
        )
    return rows


def format_status(root: Path, name=None):
    runs = [name] if name else sorted(p.name for p in (root / ".state" / "trade").glob("*"))
    if not runs:
        return "no paper trading runs yet; start one with: hft trade --paper --symbols NVDA=200 ..."
    lines = []
    for run in runs:
        rows = stock_rows(root, run)
        lines.append(f"run {run}")
        if not rows:
            lines.append("  (no saved state)")
            continue
        lines.append("  symbol  status            budget    position      equity     profit  order")
        for r in rows:
            lines.append(
                f"  {r['symbol']:<7} {r['status']:<16} {r['budget']:>8} {r['position']:>11} "
                f"{r['equity']:>11} {r['profit']:>10}  {'open' if r['pending'] else '-'}"
            )
    return "\n".join(lines)
