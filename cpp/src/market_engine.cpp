#include "hftcore/market_engine.hpp"

#include <stdexcept>
#include <utility>

namespace hftcore {
namespace {
constexpr std::int64_t kGapNs = 5 * kNs;
}

MarketEngine::MarketEngine(std::string symbol, Strategy strategy, int bar_seconds)
    : symbol_(std::move(symbol)), strategy_(strategy), bar_seconds_(bar_seconds) {}

void MarketEngine::start_session(const Session& session, std::optional<std::int64_t> now_ns) {
  const std::int64_t width = static_cast<std::int64_t>(bar_seconds_) * kNs;
  aggregator_.emplace(symbol_, bar_seconds_, session);
  if (now_ns && *now_ns > session.open_ns) {
    aggregator_->set_start(std::max(session.open_ns, *now_ns / width * width));
  }
  history_.clear();
  last_event_ns_.reset();
  last_tick_ns_.reset();
  warmup_after_ns_ = session.open_ns;
}

void MarketEngine::end_session() {
  aggregator_.reset();
  history_.clear();
}

BarAggregator& MarketEngine::require_session() {
  if (!aggregator_) throw std::logic_error("start a session before market events");
  return *aggregator_;
}

void MarketEngine::arrival(std::int64_t arrival_ns, std::vector<Update>& out) {
  if (last_event_ns_ && arrival_ns - *last_event_ns_ > kGapNs) {
    history_.clear();
    warmup_after_ns_ = arrival_ns;
    out.emplace_back(GapEvent{arrival_ns, "feed_gap"});
  }
  last_event_ns_ = arrival_ns;
}

std::vector<Update> MarketEngine::on_ignored(std::int64_t arrival_ns) {
  std::vector<Update> out;
  arrival(arrival_ns, out);
  return out;
}

std::vector<Update> MarketEngine::on_quote(const Quote& q, std::optional<std::string_view> id) {
  auto& agg = require_session();
  std::vector<Update> out;
  arrival(q.arrival_ns, out);
  bars(agg.add_quote(q, id), q.arrival_ns, out);
  return out;
}

std::vector<Update> MarketEngine::on_trade(std::int64_t event_ns, std::int64_t arrival_ns,
                                           double price, double size, bool excluded,
                                           std::optional<std::string_view> id) {
  auto& agg = require_session();
  std::vector<Update> out;
  arrival(arrival_ns, out);
  bars(agg.add_trade(event_ns, arrival_ns, price, size, excluded, id), arrival_ns, out);
  return out;
}

std::vector<Update> MarketEngine::on_correction(std::int64_t event_ns, std::int64_t arrival_ns,
                                                bool cancellation) {
  auto& agg = require_session();
  std::vector<Update> out;
  arrival(arrival_ns, out);
  bars(agg.add_correction(event_ns, arrival_ns, cancellation), arrival_ns, out);
  return out;
}

std::vector<Update> MarketEngine::advance_to(std::int64_t now_ns) {
  auto& agg = require_session();
  std::vector<Update> out;
  if (last_tick_ns_ && now_ns - *last_tick_ns_ > kGapNs) {
    history_.clear();
    warmup_after_ns_ = now_ns;
    out.emplace_back(GapEvent{now_ns, "runtime_sleep"});
  }
  last_tick_ns_ = now_ns;
  bars(agg.advance_to(now_ns), now_ns, out);
  return out;
}

void MarketEngine::bars(const std::vector<Bar>& completed, std::int64_t now_ns,
                        std::vector<Update>& out) {
  const std::int64_t width = static_cast<std::int64_t>(bar_seconds_) * kNs;
  for (const Bar& bar : completed) {
    if (bar.start_ns < warmup_after_ns_) {
      out.emplace_back(BarUpdate{bar, now_ns, false, std::nullopt, false, std::nullopt,
                                 std::nullopt});
      continue;
    }
    std::optional<std::string> reset;
    if (!history_.empty() && bar.end_ns - history_.back().end_ns != width) {
      history_.clear();
      reset = "bar_gap";
    }
    history_.push_back(bar);
    if (history_.size() > kLookback + 1) history_.pop_front();
    const bool ready = history_.size() >= kLookback + 1;
    BarUpdate update{bar, now_ns, true, reset, ready, std::nullopt, std::nullopt};
    if (ready) {
      update.market = market_features(history_);
      if (strategy_ == Strategy::HoldDay) update.action = 1;
      if (strategy_ == Strategy::EmaCrossover) {
        update.action = ema_last(history_, 5) > ema_last(history_, 20) ? 1 : 0;
      }
    }
    out.emplace_back(std::move(update));
  }
}

}  // namespace hftcore
