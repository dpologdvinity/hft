#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "hftcore/types.hpp"

namespace hftcore {

// Exact port of hft.feed.BarAggregator: causal bars published at fixed
// boundaries from information available at that time. Identities are the
// provider id text when present; std::nullopt selects the numeric fallback
// (event time and payload values), as the Python reference does.
class BarAggregator {
 public:
  BarAggregator(std::string symbol, int bar_seconds, std::optional<Session> session,
                std::int64_t late_tolerance_ns = 250'000'000, std::size_t max_events = 100'000);

  std::vector<Bar> add_quote(const Quote& quote, std::optional<std::string_view> identity);
  std::vector<Bar> add_trade(std::int64_t event_ns, std::int64_t arrival_ns, double price,
                             double size, bool excluded, std::optional<std::string_view> identity);
  std::vector<Bar> add_correction(std::int64_t event_ns, std::int64_t arrival_ns,
                                  bool cancellation);
  std::vector<Bar> advance_to(std::int64_t now_ns);

  const std::string& symbol() const noexcept { return symbol_; }
  const std::map<std::string, std::int64_t>& quality() const noexcept { return quality_; }
  const std::optional<Quote>& last_quote() const noexcept { return last_quote_; }
  std::optional<std::int64_t> start() const noexcept { return start_; }
  void set_start(std::int64_t start) noexcept { start_ = start; }

 private:
  struct TradeRecord {
    std::int64_t event_ns;
    std::size_t seq;
    double price;
    double size;
  };

  void count(const char* reason) { ++quality_[reason]; }
  void init_start(std::int64_t event_ns);
  // Shared admission checks; returns false when the event is discarded.
  bool admit(std::int64_t event_ns, std::int64_t arrival_ns, std::string key);

  std::string symbol_;
  std::int64_t width_;
  std::optional<Session> session_;
  std::int64_t late_tolerance_ns_;
  std::size_t max_events_;
  std::optional<std::int64_t> start_;
  std::optional<Quote> last_quote_;
  std::map<std::string, std::int64_t> quality_;
  std::vector<Quote> quotes_;
  std::vector<TradeRecord> trades_;
  std::unordered_set<std::string> seen_;
  std::optional<double> last_price_;
  std::int64_t max_event_ = 0;
  std::int64_t clock_ = 0;
};

}  // namespace hftcore
