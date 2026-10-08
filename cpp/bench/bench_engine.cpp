// Tick-to-decision latency of the market engine on a synthetic NVDA-like day:
// a quote every 4 ms and a trade every 48 ms for 6.5 hours, EMA-crossover strategy.
// Reports per-event latency and the latency of events that completed a decision.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "hftcore/market_engine.hpp"

using namespace hftcore;
using Clock = std::chrono::steady_clock;

namespace {
double percentile(std::vector<std::int64_t>& v, double p) {
  if (v.empty()) return 0;
  const auto k = static_cast<std::size_t>(p / 100.0 * static_cast<double>(v.size() - 1));
  std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
  return static_cast<double>(v[k]);
}

void report(const char* name, std::vector<std::int64_t> v) {
  std::printf("%-10s n=%-9zu p50=%6.0f ns  p99=%7.0f ns  p99.9=%8.0f ns  max=%9.0f ns\n", name,
              v.size(), percentile(v, 50), percentile(v, 99), percentile(v, 99.9),
              percentile(v, 100));
}
}  // namespace

int main() {
  constexpr std::int64_t open = 1736173800LL * kNs, close = open + 23400LL * kNs;
  constexpr std::int64_t step = 4'000'000;
  const std::size_t n = static_cast<std::size_t>((close - open) / step);
  std::vector<std::string> ids(n);
  for (std::size_t k = 0; k < n; ++k) ids[k] = "q" + std::to_string(k);

  MarketEngine engine("NVDA", Strategy::EmaCrossover);
  engine.start_session(Session{"2025-01-06", open, close}, std::nullopt);
  std::vector<std::int64_t> events, decisions;
  events.reserve(n + n / 12 + 8);
  std::size_t ready = 0;
  auto timed = [&](auto&& call) {
    const auto t0 = Clock::now();
    auto updates = call();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
    events.push_back(ns);
    for (const auto& u : updates) {
      if (const auto* b = std::get_if<BarUpdate>(&u); b && b->ready) {
        decisions.push_back(ns);
        ++ready;
      }
    }
  };
  for (std::size_t k = 0; k < n; ++k) {
    const std::int64_t t = open + static_cast<std::int64_t>(k) * step;
    const double mid = 140.0 + 0.001 * static_cast<double>(k % 997);
    timed([&] { return engine.on_quote(Quote{t, t, mid - 0.01, mid + 0.01, 300, 500}, ids[k]); });
    if (k % 12 == 0) {
      timed([&] { return engine.on_trade(t + 1, t + 1, mid, 25, false, std::nullopt); });
    }
  }
  std::printf("decisions=%zu\n", ready);
  report("event", events);
  report("decision", decisions);
}
