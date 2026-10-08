#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hftcore/types.hpp"

namespace hftcore {

// Read-only view of an Arrow string column (offsets + bytes), or absent.
struct IdColumn {
  const std::int32_t* offsets32 = nullptr;
  const std::int64_t* offsets64 = nullptr;
  const char* data = nullptr;
  std::size_t size = 0;
  std::string_view at(std::size_t i) const noexcept {
    if (offsets32) return {data + offsets32[i], static_cast<std::size_t>(offsets32[i + 1] - offsets32[i])};
    return {data + offsets64[i], static_cast<std::size_t>(offsets64[i + 1] - offsets64[i])};
  }
};

struct ReplayInput {
  std::string symbol;
  Session session;
  int bar_seconds = 5;
  std::span<const std::int64_t> quote_ns, quote_arrival;  // arrival empty when unrecorded
  std::span<const double> bid, ask, bid_size, ask_size;
  bool quote_ids = false;
  IdColumn quote_id_column;
  std::span<const std::int64_t> trade_ns, trade_arrival;
  std::span<const double> price, size;
  std::span<const std::uint8_t> trade_excluded;
  bool trade_ids = false;
  IdColumn trade_id_column;
};

struct ReplayResult {
  std::vector<Bar> bars;
  std::vector<std::int64_t> ready_ns;
  std::vector<std::int64_t> gaps_ns;
  std::map<std::string, std::int64_t> quality;
};

// Exact port of the bar/ready/gap construction in hft.execution.Simulation
// over hft.feed.historical_timeline(session).
ReplayResult replay(const ReplayInput& input);

}  // namespace hftcore
