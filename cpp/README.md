# hftcore

C++20 market-data and decision engine for the trading system, exposed to Python
with pybind11. It is being built to reproduce the Python reference engine exactly
(bars, gaps, features and rule decisions) while running much faster; see the
[design](../docs/specs/2026-10-07-paper-trading-runner-design.md).

## Build and test

```bash
python -m pip install cmake ninja
cmake -S cpp -B cpp/build -G Ninja -DHFTCORE_BUILD_TESTS=ON
cmake --build cpp/build -j2
ctest --test-dir cpp/build --output-on-failure

python -m pip install ./cpp           # builds the Python module
HFT_REQUIRE_HFTCORE=1 python -m pytest -q tests/test_hftcore.py
```

Options: `-DHFTCORE_SANITIZE=ON` builds with AddressSanitizer and
UndefinedBehaviorSanitizer. Warnings are errors (`-Wall -Wextra -Wpedantic`).
Third-party code (Catch2) is fetched at a pinned version with a verified SHA-256.
