// Throughput of the bar aggregator on a synthetic NVDA-like stream:
// one quote every 4 ms and a trade every 50 ms for a 6.5-hour session.
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "hftcore/bar_aggregator.hpp"

using namespace hftcore;

int main() {
  constexpr std::int64_t open = 1736173800LL * kNs;
  constexpr std::int64_t close = open + 23400LL * kNs;
  constexpr std::int64_t quote_step = 4'000'000, trade_every = 12;
  std::vector<std::string> ids;
  const std::size_t n = static_cast<std::size_t>((close - open) / quote_step);
  ids.reserve(n);
  for (std::size_t k = 0; k < n; ++k) ids.push_back("q" + std::to_string(k));

  BarAggregator agg("NVDA", 5, Session{"2025-01-06", open, close});
  std::size_t events = 0, bars = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t k = 0; k < n; ++k) {
    const std::int64_t t = open + static_cast<std::int64_t>(k) * quote_step;
    const double mid = 140.0 + 0.001 * static_cast<double>(k % 997);
    bars += agg.add_quote(Quote{t, t, mid - 0.01, mid + 0.01, 300, 500}, ids[k]).size();
    ++events;
    if (k % trade_every == 0) {
      bars += agg.add_trade(t + 1, t + 1, mid, 25, false, std::nullopt).size();
      ++events;
    }
  }
  bars += agg.advance_to(close).size();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("events=%zu bars=%zu seconds=%.3f events_per_second=%.0f ns_per_event=%.1f\n",
              events, bars, seconds, events / seconds, seconds * 1e9 / events);
}
