#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "hftcore/market_engine.hpp"

using namespace hftcore;

namespace {
constexpr std::int64_t kOpen = 1736173800LL * kNs;
const Session kSession{"2025-01-06", kOpen, kOpen + 3600 * kNs};

// One quote and one trade per 5 s bar so every bar has context.
std::vector<Update> feed_bars(MarketEngine& e, int count, double start, double step) {
  std::vector<Update> all;
  for (int k = 0; k < count; ++k) {
    const std::int64_t t = kOpen + k * 5 * kNs + kNs;
    const double price = start + step * k;
    auto add = [&](std::vector<Update> v) { all.insert(all.end(), v.begin(), v.end()); };
    add(e.on_quote(Quote{t, t, price - 0.01, price + 0.01, 100, 100}, std::nullopt));
    add(e.on_trade(t + 1, t + 1, price, 10, false, std::nullopt));
    add(e.advance_to(kOpen + (k + 1) * 5 * kNs));
  }
  return all;
}

const BarUpdate* last_ready(const std::vector<Update>& updates) {
  const BarUpdate* found = nullptr;
  for (const auto& u : updates) {
    if (const auto* b = std::get_if<BarUpdate>(&u); b && b->ready) found = b;
  }
  return found;
}
}  // namespace

TEST_CASE("decisions start once 61 contiguous bars exist") {
  MarketEngine e("X", Strategy::HoldDay);
  e.start_session(kSession, std::nullopt);
  const auto warmup = feed_bars(e, 60, 100, 0.01);
  REQUIRE(last_ready(warmup) == nullptr);
  auto more = e.advance_to(kOpen + 61 * 5 * kNs);  // carried bar from the last price
  const BarUpdate* ready = last_ready(more);
  REQUIRE(ready != nullptr);
  REQUIRE(ready->action == 1);
  REQUIRE(e.history().size() == 61);
}

TEST_CASE("EMA crossover follows the trend and features are finite") {
  MarketEngine up("X", Strategy::EmaCrossover), down("X", Strategy::EmaCrossover);
  up.start_session(kSession, std::nullopt);
  down.start_session(kSession, std::nullopt);
  // Keep the update vectors alive: last_ready returns pointers into them.
  const auto up_updates = feed_bars(up, 70, 100, 0.02);
  const auto down_updates = feed_bars(down, 70, 110, -0.02);
  const BarUpdate* rising = last_ready(up_updates);
  const BarUpdate* falling = last_ready(down_updates);
  REQUIRE(rising->action == 1);
  REQUIRE(falling->action == 0);
  for (float v : *rising->market) REQUIRE(std::isfinite(v));
  REQUIRE((*rising->market)[0] > 0);  // one-bar log return of a rising series
}

TEST_CASE("a 6 s arrival gap and a clock jump reset history") {
  MarketEngine e("X", Strategy::Model);
  e.start_session(kSession, std::nullopt);
  feed_bars(e, 3, 100, 0.01);
  REQUIRE(e.history().size() == 3);
  auto gap = e.on_trade(kOpen + 21 * kNs, kOpen + 21 * kNs, 100, 1, false, std::nullopt);
  REQUIRE(std::get<GapEvent>(gap.front()).reason == "feed_gap");
  REQUIRE(e.history().empty());
  auto sleep = e.advance_to(kOpen + 40 * kNs);
  REQUIRE(std::get<GapEvent>(sleep.front()).reason == "runtime_sleep");
}
