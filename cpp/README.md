# hftcore

C++20 market-data and decision engine for the trading system, exposed to Python
with pybind11. It reproduces the Python reference engine bit for bit (bars, gaps,
warmup, features and rule decisions) and powers both backtests and the paper bot;
see the [design](../docs/specs/2026-10-07-paper-trading-runner-design.md) and
[measured results](../docs/performance-results.md).

| Component | Purpose |
| --- | --- |
| `bar_aggregator` | Causal 5-second bars with boundary timers, late/future/duplicate rules |
| `id_set` | Allocation-free 128-bit identity set with O(1) per-bar clearing |
| `features`, `market_engine` | Decision history, gap/warmup rules, 10 market features, EMA crossover / hold-day |
| `replay` | Whole-session replay for backtests (zero-copy Arrow identity buffers) |
| `frame_router` | simdjson parsing of Alpaca stream frames into per-symbol engines |
| `timestamp`, `decimal`, `numeric` | RFC3339 parsing, exact decimal scaling, CPython-compatible float sums |

## Build, test, benchmark

```bash
python -m pip install cmake ninja
cmake -S cpp -B cpp/build -G Ninja -DHFTCORE_BUILD_TESTS=ON
cmake --build cpp/build -j2
ctest --test-dir cpp/build --output-on-failure
cpp/build/hftcore_bench_engine      # tick-to-decision latency percentiles
cpp/build/hftcore_bench_frames      # JSON frame parse-to-decision latency
cpp/build/hftcore_bench_aggregator  # aggregator throughput

python -m pip install ./cpp           # Python module; used automatically when installed
HFT_REQUIRE_HFTCORE=1 python -m pytest -q tests/test_engine_parity.py tests/test_replay_parity.py
```

`-DHFTCORE_SANITIZE=ON` builds with AddressSanitizer and UndefinedBehaviorSanitizer
(also run in CI). Warnings are errors, floating-point contraction is disabled for
reproducible results, and third-party code (simdjson, Catch2) is fetched at pinned
versions with verified SHA-256 hashes.
