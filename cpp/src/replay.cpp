#include "hftcore/replay.hpp"

#include <algorithm>
#include <numeric>
#include <optional>

#include "hftcore/bar_aggregator.hpp"

namespace hftcore {
namespace {

std::vector<std::size_t> arrival_order(std::size_t n, std::span<const std::int64_t> arrival) {
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  if (!arrival.empty()) {
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return arrival[a] < arrival[b]; });
  }
  return order;
}

}  // namespace

ReplayResult replay(const ReplayInput& in) {
  ReplayResult out;
  BarAggregator agg(in.symbol, in.bar_seconds, in.session);
  const std::int64_t width = static_cast<std::int64_t>(in.bar_seconds) * kNs;
  const std::int64_t close = in.session.close_ns;
  std::int64_t timer = in.session.open_ns + width;
  std::int64_t last_event = in.session.open_ns;

  auto emit = [&](const std::vector<Bar>& bars, std::int64_t now) {
    for (const Bar& b : bars) {
      out.bars.push_back(b);
      out.ready_ns.push_back(now);
    }
  };

  const auto qorder = arrival_order(in.quote_ns.size(), in.quote_arrival);
  const auto torder = arrival_order(in.trade_ns.size(), in.trade_arrival);
  auto qtime = [&](std::size_t i) { return in.quote_arrival.empty() ? in.quote_ns[i] : in.quote_arrival[i]; };
  auto ttime = [&](std::size_t i) { return in.trade_arrival.empty() ? in.trade_ns[i] : in.trade_arrival[i]; };

  std::size_t qi = 0, ti = 0;
  bool stopped = false;
  while (!stopped && (qi < qorder.size() || ti < torder.size())) {
    // heapq.merge over (time, kind: quote 0 / trade 1, index).
    bool take_quote;
    if (qi == qorder.size()) {
      take_quote = false;
    } else if (ti == torder.size()) {
      take_quote = true;
    } else {
      const std::int64_t a = qtime(qorder[qi]), b = ttime(torder[ti]);
      take_quote = a < b || (a == b);  // quote kind 0 sorts before trade kind 1
    }
    const std::size_t i = take_quote ? qorder[qi++] : torder[ti++];
    const std::int64_t now = take_quote ? qtime(i) : ttime(i);
    // historical_timeline: timers up to the arrival come first.
    while (timer <= std::min(now, close)) {
      emit(agg.advance_to(timer), timer);
      timer += width;
    }
    if (now >= close) {
      stopped = true;
      break;
    }
    if (now - last_event > 5 * kNs) out.gaps_ns.push_back(now);
    last_event = now;
    if (take_quote) {
      std::optional<std::string_view> id;
      if (in.quote_ids) id = in.quote_id_column.at(i);
      emit(agg.add_quote(Quote{in.quote_ns[i], now, in.bid[i], in.ask[i], in.bid_size[i],
                               in.ask_size[i]},
                         id),
           now);
    } else {
      std::optional<std::string_view> id;
      if (in.trade_ids) id = in.trade_id_column.at(i);
      emit(agg.add_trade(in.trade_ns[i], now, in.price[i], in.size[i],
                         in.trade_excluded[i] != 0, id),
           now);
    }
  }
  while (timer <= close) {
    emit(agg.advance_to(timer), timer);
    timer += width;
  }
  if (close - last_event > 5 * kNs) out.gaps_ns.push_back(close);
  out.quality = agg.quality();
  return out;
}

}  // namespace hftcore
