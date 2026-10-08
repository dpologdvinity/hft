#pragma once

#include <array>
#include <deque>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "hftcore/bar_aggregator.hpp"
#include "hftcore/features.hpp"

namespace hftcore {

enum class Strategy { Model, EmaCrossover, HoldDay };

struct GapEvent {
  std::int64_t now_ns;
  std::string reason;  // "feed_gap" or "runtime_sleep"
};

struct BarUpdate {
  Bar bar;
  std::int64_t now_ns;
  bool accepted;
  std::optional<std::string> reset;  // "bar_gap"
  bool ready;
  std::optional<std::array<float, kMarketFeatures>> market;
  std::optional<int> action;
};

using Update = std::variant<GapEvent, BarUpdate>;

// Exact port of hft.market_engine.PyMarketEngine.
class MarketEngine {
 public:
  MarketEngine(std::string symbol, Strategy strategy, int bar_seconds = 5);

  void start_session(const Session& session, std::optional<std::int64_t> now_ns);
  void end_session();
  void reset_history() { history_.clear(); }
  // Live feed silence: drop history and restart warmup after now_ns.
  void mark_gap(std::int64_t now_ns) {
    history_.clear();
    warmup_after_ns_ = now_ns;
  }

  // Each call first runs the arrival-gap check, then feeds the aggregator.
  std::vector<Update> on_quote(const Quote& quote, std::optional<std::string_view> identity);
  std::vector<Update> on_trade(std::int64_t event_ns, std::int64_t arrival_ns, double price,
                               double size, bool excluded, std::optional<std::string_view> identity);
  std::vector<Update> on_correction(std::int64_t event_ns, std::int64_t arrival_ns,
                                    bool cancellation);
  std::vector<Update> advance_to(std::int64_t now_ns);

  // Arrival-gap check for events the aggregator ignores (other symbols/kinds).
  std::vector<Update> on_ignored(std::int64_t arrival_ns);

  const std::string& symbol() const noexcept { return symbol_; }
  const std::deque<Bar>& history() const noexcept { return history_; }
  BarAggregator* aggregator() noexcept { return aggregator_ ? &*aggregator_ : nullptr; }
  std::optional<std::int64_t> last_event_ns() const noexcept { return last_event_ns_; }
  std::optional<std::int64_t> last_tick_ns() const noexcept { return last_tick_ns_; }
  std::int64_t warmup_after_ns() const noexcept { return warmup_after_ns_; }

 private:
  void arrival(std::int64_t arrival_ns, std::vector<Update>& out);
  void bars(const std::vector<Bar>& bars, std::int64_t now_ns, std::vector<Update>& out);
  BarAggregator& require_session();

  std::string symbol_;
  Strategy strategy_;
  int bar_seconds_;
  std::optional<BarAggregator> aggregator_;
  std::deque<Bar> history_;
  std::optional<std::int64_t> last_event_ns_, last_tick_ns_;
  std::int64_t warmup_after_ns_ = 0;
};

}  // namespace hftcore
