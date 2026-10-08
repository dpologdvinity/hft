#include "hftcore/features.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "hftcore/numeric.hpp"

namespace hftcore {

std::array<float, kMarketFeatures> market_features(const std::deque<Bar>& history) {
  if (history.size() < kLookback + 1) throw std::invalid_argument("need 61 completed bars");
  const Bar& b = history.back();
  for (std::size_t i = history.size() - (kLookback + 1); i < history.size(); ++i) {
    if (history[i].session_id != b.session_id) {
      throw std::invalid_argument("history crosses sessions");
    }
  }
  std::array<double, kMarketFeatures> v{};
  const std::size_t n = history.size();
  const int windows[] = {1, 5, 15, 60};
  for (int k = 0; k < 4; ++k) v[k] = std::log(b.close / history[n - 1 - windows[k]].close);
  const double span = std::max(b.high - b.low, 1e-12);
  v[4] = (b.close - b.open) / span;
  v[5] = (b.high - std::max(b.open, b.close)) / span;
  v[6] = (std::min(b.close, b.open) - b.low) / span;
  PySum volume_sum, notional_sum;
  for (std::size_t i = n - kLookback; i < n; ++i) volume_sum.add(history[i].volume);
  for (std::size_t i = n - kLookback; i < n; ++i) {
    notional_sum.add(history[i].vwap * history[i].volume);
  }
  const double volume = volume_sum.value();
  const double vwap = volume != 0 ? notional_sum.value() / volume : b.close;
  const double depth = b.bid_size + b.ask_size;
  const double mid = (b.bid + b.ask) / 2;
  v[7] = (b.ask - b.bid) / mid;
  v[8] = depth != 0 ? (b.bid_size - b.ask_size) / depth : 0.0;
  v[9] = b.close / vwap - 1;
  std::array<float, kMarketFeatures> out{};
  for (std::size_t i = 0; i < kMarketFeatures; ++i) out[i] = static_cast<float>(v[i]);
  return out;
}

double ema_last(const std::deque<Bar>& history, int span) {
  if (history.empty()) throw std::invalid_argument("empty history");
  const double alpha = 2.0 / (span + 1);
  double value = history.front().close;
  for (std::size_t i = 1; i < history.size(); ++i) {
    value = alpha * history[i].close + (1 - alpha) * value;
  }
  return value;
}

}  // namespace hftcore
