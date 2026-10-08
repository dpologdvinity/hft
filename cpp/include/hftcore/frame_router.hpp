#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hftcore/market_engine.hpp"

namespace hftcore {

// `message` is the index of the source message in the frame, so callers can
// interleave updates and quotes in arrival order, as the per-message path does.
struct RoutedUpdate {
  std::string symbol;
  Update update;
  int message;
};

struct ParsedQuote {
  std::string symbol;
  Quote quote;
  int message;
};

struct FrameResult {
  std::vector<RoutedUpdate> updates;
  std::vector<ParsedQuote> quotes;  // for the Python Decimal quote path
  int malformed = 0;
};

// Parses an Alpaca market-data frame (a JSON array of messages) and feeds each
// subscribed symbol's MarketEngine, matching the Python path (json.loads +
// stream filtering + MarketEngine.on_event) exactly. Control messages and other
// symbols are skipped; non-array frames and non-object entries count as
// malformed; "error" messages and invalid fields throw.
class FrameRouter {
 public:
  FrameRouter();
  ~FrameRouter();
  void add(MarketEngine& engine);
  FrameResult on_frame(std::string_view raw, std::int64_t arrival_ns);

 private:
  struct Parser;
  std::unique_ptr<Parser> parser_;
  std::unordered_map<std::string, MarketEngine*> engines_;
};

}  // namespace hftcore
