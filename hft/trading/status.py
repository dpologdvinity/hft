"""`hft trade --status`: per-stock state of paper runs from local files only."""

import json
from decimal import Decimal
from pathlib import Path


def _journals(log_dir: Path, symbol: str):
    """The stock's journals, newest first (each restart of a run starts a new one)."""
    return sorted(log_dir.glob(f"{symbol}-*.jsonl"), key=lambda p: p.stat().st_mtime)[::-1]


def _last_equity(journals, tail):
    from .command import tail_rows

    for rows in [tail] + [tail_rows(p) for p in journals[1:]]:
        for row in reversed(rows):
            if row.get("event") == "equity":
                return Decimal(str(row["equity"]))
    return None


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
        journals = _journals(log_dir, symbol)
        tail = tail_rows(journals[0]) if journals else []
        equity = _last_equity(journals, tail)
        if equity is None:
            equity = cash if position == 0 else None  # a holding is never shown at zero
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
                "equity": None if equity is None else str(equity.quantize(Decimal("0.01"))),
                "profit": None
                if equity is None
                else str((equity - budget).quantize(Decimal("0.01"))),
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
                f"{r['equity'] or 'n/a':>11} {r['profit'] or 'n/a':>10}  "
                f"{'open' if r['pending'] else '-'}"
            )
    return "\n".join(lines)
