#include <catch2/catch_test_macros.hpp>

#include "hftcore/bar_aggregator.hpp"

using namespace hftcore;

namespace {
constexpr std::int64_t kOpen = 1736173800LL * kNs;
Session session() { return {"2025-01-06", kOpen, kOpen + 3600 * kNs}; }
Quote quote(std::int64_t t, double bid = 100.0, double ask = 100.02) {
  return {t, t, bid, ask, 100, 200};
}
}  // namespace

TEST_CASE("bars publish at fixed boundaries from earlier information") {
  BarAggregator agg("X", 5, session());
  REQUIRE(agg.add_quote(quote(kOpen + 1 * kNs), "q1").empty());
  REQUIRE(agg.add_trade(kOpen + 2 * kNs, kOpen + 2 * kNs, 100.01, 10, false, "t1").empty());
  REQUIRE(agg.add_trade(kOpen + 3 * kNs, kOpen + 3 * kNs, 100.03, 30, false, "t2").empty());
  auto bars = agg.advance_to(kOpen + 5 * kNs);
  REQUIRE(bars.size() == 1);
  const Bar& b = bars[0];
  REQUIRE(b.start_ns == kOpen);
  REQUIRE(b.end_ns == kOpen + 5 * kNs);
  REQUIRE(b.open == 100.01);
  REQUIRE(b.close == 100.03);
  REQUIRE(b.high == 100.03);
  REQUIRE(b.volume == 40);
  REQUIRE(b.vwap == (100.01 * 10 + 100.03 * 30) / 40);
  REQUIRE(b.tradable == false);  // quote is 4 s old at the boundary
}

TEST_CASE("duplicates, late and future events are counted and dropped") {
  BarAggregator agg("X", 5, session());
  agg.add_quote(quote(kOpen + 1 * kNs), "q1");
  agg.add_quote(quote(kOpen + 1 * kNs), "q1");
  agg.add_trade(kOpen + kNs, kOpen + 2 * kNs, 100, 1, false, std::nullopt);  // 1 s late
  agg.add_trade(kOpen + 3 * kNs, kOpen + 2 * kNs, 100, 1, false, std::nullopt);  // future
  agg.add_trade(kOpen + 2 * kNs, kOpen + 2 * kNs, 100, 1, true, std::nullopt);
  REQUIRE(agg.quality().at("duplicates") == 1);
  REQUIRE(agg.quality().at("late_tolerance") == 1);
  REQUIRE(agg.quality().at("future_event") == 1);
  REQUIRE(agg.quality().at("filtered_trades") == 1);
}

TEST_CASE("bars without a quote or a price are missing context") {
  BarAggregator agg("X", 5, session());
  agg.add_quote(quote(kOpen + 1 * kNs), "q1");
  REQUIRE(agg.advance_to(kOpen + 5 * kNs).empty());
  REQUIRE(agg.quality().at("missing_context") == 1);
  REQUIRE(agg.advance_to(kOpen + 4 * kNs).empty());
  REQUIRE(agg.quality().at("clock_reversal") == 1);
}

TEST_CASE("invalid trades and arrivals raise") {
  BarAggregator agg("X", 5, session());
  REQUIRE_THROWS_AS(agg.add_trade(kOpen, kOpen, -1, 1, false, std::nullopt),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(agg.add_quote(Quote{kOpen, -1, 1, 2, 1, 1}, std::nullopt),
                    std::invalid_argument);
}
