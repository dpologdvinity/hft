#pragma once

#include <cstdint>
#include <string>

namespace hftcore {

inline constexpr std::int64_t kNs = 1'000'000'000;

struct Session {
  std::string id;
  std::int64_t open_ns = 0;
  std::int64_t close_ns = 0;
  bool contains(std::int64_t t) const noexcept { return open_ns <= t && t < close_ns; }
};

struct Quote {
  std::int64_t event_ns = 0;
  std::int64_t arrival_ns = 0;
  double bid = 0, ask = 0, bid_size = 0, ask_size = 0;
};

// Mirrors hft.data.Bar.
struct Bar {
  std::int64_t start_ns = 0, end_ns = 0, quote_ns = 0;
  std::string session_id;
  bool tradable = false;
  double open = 0, high = 0, low = 0, close = 0, volume = 0;
  double bid = 0, ask = 0, bid_size = 0, ask_size = 0, vwap = 0;
};

}  // namespace hftcore
