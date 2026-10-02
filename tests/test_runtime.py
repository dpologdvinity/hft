import asyncio

import pytest

from hft.runtime import receive_market


def test_quote_intake_keeps_latest_and_overflow_halts_without_rest():
    async def exercise():
        async def stream():
            for i in range(3):
                yield {
                    "T": "q",
                    "S": "AAPL",
                    "event_ns": i + 1,
                    "bp": 100,
                    "ap": 101,
                    "bs": 1,
                    "as": 1,
                    "arrival_ns": i + 1,
                }

        queue = asyncio.Queue(maxsize=1)
        latest = {}
        with pytest.raises(RuntimeError, match="overflow"):
            await receive_market(stream(), queue, latest)
        assert latest["event"]["event_ns"] == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_quiet_closed_market_does_not_abort_before_open(tmp_path, monkeypatch, cancel):
    import time
    from dataclasses import asdict
    from typing import ClassVar

    import hft.runtime
    from hft.account import Costs
    from hft.configuration import RuntimeConfig
    from hft.risk import RiskConfig
    from hft.runtime import run_stream
    from hft.sizing import SizingConfig
    from hft.state import AccountStateStore

    now = time.time_ns()
    monkeypatch.setattr(
        hft.runtime,
        "fetch_calendar",
        lambda client=None: [
            {"session_id": "future", "open_ns": now + 3600 * 10**9, "close_ns": now + 7200 * 10**9}
        ],
    )
    monkeypatch.setattr(
        hft.runtime,
        "AccountStateStore",
        lambda identity, mode: AccountStateStore(identity, mode, root=tmp_path / "state"),
    )

    class Policy:
        metadata: ClassVar[dict] = {
            "sha256": str(tmp_path),
            "initial_cash": 500,
            "symbol": "AAPL",
            "feed": "iex",
            "costs": asdict(Costs()),
            "risk": asdict(RiskConfig()),
            "sizing": asdict(SizingConfig()),
            "latency_ms": 75,
            "bar_seconds": 5,
        }
        manifest: ClassVar[dict] = {"contract_hash": "test"}

        def __call__(self, obs):
            return 0

    async def quiet_stream():
        await asyncio.sleep(10)
        yield {}

    config = RuntimeConfig("AAPL", "iex", 500, Costs(), RiskConfig(), SizingConfig())

    async def exercise():
        task = asyncio.create_task(
            run_stream(
                Policy(),
                config,
                log_path=tmp_path / "run.jsonl",
                duration=0.1,
                stream=quiet_stream(),
            )
        )
        if cancel:
            await asyncio.sleep(0.03)
            task.cancel()
        return await task

    result = asyncio.run(exercise())
    assert result["run_completed"] and not result["complete"]
    assert result["position"] == "0"
