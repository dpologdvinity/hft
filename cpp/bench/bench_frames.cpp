// Parse-to-decision latency: one Alpaca-format JSON frame per call through
// FrameRouter into two engines, for a synthetic 6.5-hour session.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

#include "hftcore/frame_router.hpp"

using namespace hftcore;

namespace {
std::string rfc3339(std::int64_t ns) {
  std::time_t seconds = ns / kNs;
  char date[32];
  std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%S", std::gmtime(&seconds));
  char out[64];
  std::snprintf(out, sizeof out, "%s.%09lldZ", date, static_cast<long long>(ns % kNs));
  return out;
}
}  // namespace

int main() {
  constexpr std::int64_t open = 1736173800LL * kNs, close = open + 23400LL * kNs;
  constexpr std::int64_t step = 4'000'000;
  MarketEngine nvda("NVDA", Strategy::EmaCrossover), aapl("AAPL", Strategy::EmaCrossover);
  const Session session{"2025-01-06", open, close};
  nvda.start_session(session, std::nullopt);
  aapl.start_session(session, std::nullopt);
  FrameRouter router;
  router.add(nvda);
  router.add(aapl);
  std::vector<std::int64_t> latencies;
  std::size_t decisions = 0;
  char frame[512];
  for (std::int64_t t = open, k = 0; t < close; t += step, ++k) {
    const char* symbol = (k % 2) ? "AAPL" : "NVDA";
    const double mid = 140.0 + 0.001 * static_cast<double>(k % 997);
    const std::string stamp = rfc3339(t);
    int n = std::snprintf(frame, sizeof frame,
                          "[{\"T\":\"q\",\"S\":\"%s\",\"bp\":%.2f,\"bs\":3,\"ap\":%.2f,"
                          "\"as\":5,\"t\":\"%s\",\"z\":\"C\"}",
                          symbol, mid - 0.01, mid + 0.01, stamp.c_str());
    if (k % 12 < 2) {  // a trade for each symbol every 12 frames
      n += std::snprintf(frame + n, sizeof frame - static_cast<std::size_t>(n),
                         ",{\"T\":\"t\",\"S\":\"%s\",\"i\":%lld,\"p\":%.2f,\"s\":25,"
                         "\"t\":\"%s\",\"c\":[\"@\"],\"z\":\"C\"}",
                         symbol, static_cast<long long>(k), mid, stamp.c_str());
    }
    n += std::snprintf(frame + n, sizeof frame - static_cast<std::size_t>(n), "]");
    const auto t0 = std::chrono::steady_clock::now();
    auto result = router.on_frame({frame, static_cast<std::size_t>(n)}, t);
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    latencies.push_back(ns);
    for (const auto& [s, u] : result.updates) {
      if (const auto* b = std::get_if<BarUpdate>(&u); b && b->ready) ++decisions;
    }
  }
  auto pct = [&](double p) {
    auto k = static_cast<std::size_t>(p / 100 * static_cast<double>(latencies.size() - 1));
    std::nth_element(latencies.begin(), latencies.begin() + static_cast<long>(k), latencies.end());
    return static_cast<double>(latencies[k]);
  };
  std::printf("frames=%zu decisions=%zu p50=%.0f ns p99=%.0f ns p99.9=%.0f ns\n",
              latencies.size(), decisions, pct(50), pct(99), pct(99.9));
}
