#include "hftcore/bar_aggregator.hpp"

#include "hftcore/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace hftcore {
namespace {

std::string numeric_key(char kind, std::int64_t stamp, std::initializer_list<double> values) {
  std::string key(1, kind);
  key.push_back('n');
  key.append(reinterpret_cast<const char*>(&stamp), sizeof stamp);
  for (double v : values) key.append(reinterpret_cast<const char*>(&v), sizeof v);
  return key;
}

std::string text_key(char kind, std::string_view id) {
  std::string key(1, kind);
  key.push_back('s');
  key.append(id);
  return key;
}

}  // namespace

BarAggregator::BarAggregator(std::string symbol, int bar_seconds, std::optional<Session> session,
                             std::int64_t late_tolerance_ns, std::size_t max_events)
    : symbol_(std::move(symbol)),
      width_(static_cast<std::int64_t>(bar_seconds) * kNs),
      session_(std::move(session)),
      late_tolerance_ns_(late_tolerance_ns),
      max_events_(max_events) {
  if (bar_seconds != 1 && bar_seconds != 5) {
    throw std::invalid_argument("bar_seconds must be 1 or 5");
  }
  if (session_) start_ = session_->open_ns;
}

void BarAggregator::init_start(std::int64_t event_ns) {
  if (!start_) {
    // Python floor division.
    std::int64_t q = event_ns / width_;
    if ((event_ns % width_ != 0) && ((event_ns < 0) != (width_ < 0))) --q;
    start_ = q * width_;
  }
}

bool BarAggregator::admit(std::int64_t stamp, std::int64_t arrival, std::string key) {
  if (session_ && !session_->contains(stamp)) {
    count("outside_session");
    return false;
  }
  if (stamp < *start_) {
    count("late_closed");
    return false;
  }
  if (stamp > arrival + 250'000'000) {
    count("future_event");
    return false;
  }
  if (max_event_ - stamp > late_tolerance_ns_ || arrival - stamp > late_tolerance_ns_) {
    count("late_tolerance");
    return false;
  }
  if (seen_.contains(key)) {
    count("duplicates");
    return false;
  }
  if (quotes_.size() + trades_.size() >= max_events_) {
    count("buffer_overflow");
    throw std::invalid_argument("bounded event buffer exhausted");
  }
  seen_.insert(std::move(key));
  max_event_ = std::max(max_event_, stamp);
  return true;
}

std::vector<Bar> BarAggregator::add_quote(const Quote& q, std::optional<std::string_view> id) {
  if (q.arrival_ns < 0) throw std::invalid_argument("invalid arrival_ns");
  init_start(q.event_ns);
  std::vector<Bar> completed = advance_to(q.arrival_ns);
  std::string key = id ? text_key('q', *id)
                       : numeric_key('q', q.event_ns, {q.bid, q.ask, q.bid_size, q.ask_size});
  if (!admit(q.event_ns, q.arrival_ns, std::move(key))) return completed;
  quotes_.push_back(q);
  if (!last_quote_ || q.event_ns >= last_quote_->event_ns) last_quote_ = q;
  return completed;
}

std::vector<Bar> BarAggregator::add_trade(std::int64_t stamp, std::int64_t arrival, double price,
                                          double size, bool excluded,
                                          std::optional<std::string_view> id) {
  if (arrival < 0) throw std::invalid_argument("invalid arrival_ns");
  init_start(stamp);
  if (!std::isfinite(price) || !std::isfinite(size) || price <= 0 || size < 0) {
    throw std::invalid_argument("invalid trade");
  }
  std::vector<Bar> completed = advance_to(arrival);
  std::string key = id ? text_key('t', *id) : numeric_key('t', stamp, {price, size});
  if (!admit(stamp, arrival, std::move(key))) return completed;
  if (excluded) {
    count("filtered_trades");
  } else {
    trades_.push_back({stamp, trades_.size(), price, size});
  }
  return completed;
}

std::vector<Bar> BarAggregator::add_correction(std::int64_t stamp, std::int64_t arrival,
                                               bool cancellation) {
  if (arrival < 0) throw std::invalid_argument("invalid arrival_ns");
  init_start(stamp);
  count(cancellation ? "cancellations" : "corrections");
  return advance_to(arrival);
}

std::vector<Bar> BarAggregator::advance_to(std::int64_t now_ns) {
  std::vector<Bar> completed;
  if (now_ns < clock_) {
    count("clock_reversal");
    return completed;
  }
  clock_ = now_ns;
  if (!start_) return completed;
  const std::int64_t limit = session_ ? std::min(now_ns, session_->close_ns) : now_ns;
  std::vector<TradeRecord> eligible;
  while (*start_ + width_ <= limit) {
    const std::int64_t start = *start_;
    const std::int64_t end = start + width_;
    eligible.clear();
    for (const auto& t : trades_) {
      if (start <= t.event_ns && t.event_ns < end) eligible.push_back(t);
    }
    std::stable_sort(eligible.begin(), eligible.end(), [](const auto& a, const auto& b) {
      return a.event_ns != b.event_ns ? a.event_ns < b.event_ns : a.seq < b.seq;
    });
    // Latest-stamped quote before the boundary; ties keep the last received.
    std::optional<Quote> quote;
    for (const auto& q : quotes_) {
      if (q.event_ns < end && (!quote || q.event_ns >= quote->event_ns)) quote = q;
    }
    if (!quote && last_quote_ && last_quote_->event_ns < end) quote = last_quote_;
    if (!eligible.empty()) last_price_ = eligible.back().price;
    if (quote && last_price_) {
      Bar bar;
      bar.start_ns = start;
      bar.end_ns = end;
      bar.quote_ns = quote->event_ns;
      bar.session_id = session_ ? session_->id : std::string();
      bar.tradable = end - quote->event_ns <= 2 * kNs && quote->bid_size > 0 && quote->ask_size > 0;
      if (eligible.empty()) {
        bar.open = bar.high = bar.low = bar.close = *last_price_;
      } else {
        bar.open = eligible.front().price;
        bar.close = eligible.back().price;
        bar.high = bar.low = eligible.front().price;
        for (const auto& t : eligible) {
          // Python max()/min() keep the first extreme; equal values are identical.
          if (t.price > bar.high) bar.high = t.price;
          if (t.price < bar.low) bar.low = t.price;
        }
      }
      PySum volume_sum, notional_sum;
      for (const auto& t : eligible) volume_sum.add(t.size);
      for (const auto& t : eligible) notional_sum.add(t.price * t.size);
      const double volume = volume_sum.value(), notional = notional_sum.value();
      bar.volume = volume;
      bar.vwap = volume != 0 ? notional / volume : *last_price_;
      bar.bid = quote->bid;
      bar.ask = quote->ask;
      bar.bid_size = quote->bid_size;
      bar.ask_size = quote->ask_size;
      completed.push_back(std::move(bar));
    } else {
      count("missing_context");
    }
    std::erase_if(trades_, [end](const auto& t) { return t.event_ns < end; });
    std::vector<Quote> kept;
    if (quote) kept.push_back(*quote);
    for (const auto& q : quotes_) {
      if (q.event_ns >= end) kept.push_back(q);
    }
    quotes_.swap(kept);
    seen_.clear();
    start_ = end;
  }
  return completed;
}

}  // namespace hftcore
