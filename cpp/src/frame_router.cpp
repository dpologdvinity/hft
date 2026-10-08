#include "hftcore/frame_router.hpp"

#include <simdjson.h>

#include <stdexcept>

#include "hftcore/decimal.hpp"
#include "hftcore/timestamp.hpp"

namespace hftcore {

namespace od = simdjson::ondemand;

struct FrameRouter::Parser {
  od::parser parser;
  std::string buffer;
};

FrameRouter::FrameRouter() : parser_(std::make_unique<Parser>()) {}
FrameRouter::~FrameRouter() = default;

void FrameRouter::add(MarketEngine& engine) { engines_[engine.symbol()] = &engine; }

namespace {

[[noreturn]] void invalid(const char* what) { throw std::invalid_argument(what); }

double number(od::object& object, const char* key) {
  double value;
  if (object[key].get_double().get(value)) invalid("invalid numeric field");
  return value;
}

// Python: float(Decimal(str(float(json))) * 100) for lot sizes.
double lots_to_shares(double lots) {
  char buffer[32];
  return scale_decimal(shortest_repr(lots, buffer), 2);
}

// Python: str(event["i"]) for the identity, when present.
std::optional<std::string> identity(od::object& object) {
  od::value value;
  if (object["i"].get(value)) return std::nullopt;
  od::json_type type;
  if (value.type().get(type)) invalid("invalid identity");
  if (type == od::json_type::string) {
    std::string_view text;
    if (value.get_string().get(text)) invalid("invalid identity");
    return std::string(text);
  }
  if (type == od::json_type::number) {
    std::string_view raw = value.raw_json_token();
    while (!raw.empty() && (raw.back() == ' ' || raw.back() == '\n' || raw.back() == '\r' ||
                            raw.back() == '\t')) {
      raw.remove_suffix(1);
    }
    return std::string(raw);
  }
  invalid("unsupported identity type");
}

bool excluded_trade(od::object& object) {
  std::string_view tape = "C";
  od::value z;
  if (!object["z"].get(z)) {
    if (z.get_string().get(tape)) tape = "";  // non-string tape: not "C"
  }
  const std::string_view base = "479CGHIMNPQRTUVZ";
  const char extra = tape == "C" ? 'W' : 'B';
  od::array conditions;
  if (object["c"].get_array().get(conditions)) return false;
  for (auto entry : conditions) {
    std::string_view code;
    if (entry.get_string().get(code) || code.size() != 1) continue;
    if (code[0] == extra || base.find(code[0]) != std::string_view::npos) return true;
  }
  return false;
}

}  // namespace

FrameResult FrameRouter::on_frame(std::string_view raw, std::int64_t arrival_ns) {
  FrameResult result;
  auto& p = *parser_;
  p.buffer.assign(raw);
  p.buffer.reserve(raw.size() + simdjson::SIMDJSON_PADDING);
  od::document doc;
  od::array messages;
  if (p.parser.iterate(p.buffer.data(), raw.size(), p.buffer.capacity()).get(doc) ||
      doc.get_array().get(messages)) {
    result.malformed = 1;
    return result;
  }
  int message = -1;
  for (auto element : messages) {
    ++message;
    od::object object;
    if (element.get_object().get(object)) {
      ++result.malformed;
      continue;
    }
    std::string_view kind;
    if (object["T"].get_string().get(kind)) continue;  // control messages without T
    if (kind == "error") {
      int64_t code = 0;
      if (object["code"].get_int64().get(code)) code = 0;
      throw std::runtime_error("market data stream error: " + std::to_string(code));
    }
    if (kind != "q" && kind != "t" && kind != "c" && kind != "x") continue;
    std::string_view symbol_view;
    if (object["S"].get_string().get(symbol_view)) continue;
    auto found = engines_.find(std::string(symbol_view));
    if (found == engines_.end()) continue;
    MarketEngine& engine = *found->second;
    std::string symbol(symbol_view);
    std::string_view stamp_text;
    if (object["t"].get_string().get(stamp_text)) invalid("timestamp must be RFC3339 text");
    const std::int64_t stamp = parse_timestamp(stamp_text);
    std::vector<Update> updates;
    if (kind == "c" || kind == "x") {
      updates = engine.on_correction(stamp, arrival_ns, kind == "x");
    } else if (kind == "q") {
      // Field order in Alpaca messages is not guaranteed; on-demand lookups rewind.
      auto id = identity(object);
      Quote q{stamp, arrival_ns, number(object, "bp"), number(object, "ap"),
              lots_to_shares(number(object, "bs")), lots_to_shares(number(object, "as"))};
      std::optional<std::string_view> id_view;
      if (id) id_view = *id;
      updates = engine.on_quote(q, id_view);
      result.quotes.push_back({symbol, q, message});
    } else {
      auto id = identity(object);
      const double price = number(object, "p"), size = number(object, "s");
      const bool excluded = excluded_trade(object);
      std::optional<std::string_view> id_view;
      if (id) id_view = *id;
      updates = engine.on_trade(stamp, arrival_ns, price, size, excluded, id_view);
    }
    for (auto& u : updates) result.updates.push_back({symbol, std::move(u), message});
  }
  return result;
}

}  // namespace hftcore
