#pragma once

#include <array>
#include <deque>

#include "hftcore/types.hpp"

namespace hftcore {

inline constexpr std::size_t kLookback = 60;
inline constexpr std::size_t kMarketFeatures = 10;

// Exact port of hft.features.market_features: each value is computed in double
// in the Python operation order and rounded to float32 last.
std::array<float, kMarketFeatures> market_features(const std::deque<Bar>& history);

// hft.research.ema(closes, span)[-1]: seeded with the first close.
double ema_last(const std::deque<Bar>& history, int span);

}  // namespace hftcore
