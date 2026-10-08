#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <variant>
#include <string>

#include "hftcore/bar_aggregator.hpp"
#include "hftcore/decimal.hpp"
#include "hftcore/market_engine.hpp"
#include "hftcore/timestamp.hpp"
#include "hftcore/version.hpp"

namespace py = pybind11;
using namespace hftcore;

namespace {

constexpr const char* kCloseExcluded = "479CGHIMNPQRTUVZ";

py::tuple bar_tuple(const Bar& b) {
  return py::make_tuple(b.start_ns, b.end_ns, b.quote_ns, b.session_id, b.tradable, b.open,
                        b.high, b.low, b.close, b.volume, b.bid, b.ask, b.bid_size, b.ask_size,
                        b.vwap);
}

py::list bar_list(const std::vector<Bar>& bars) {
  py::list out;
  for (const auto& b : bars) out.append(bar_tuple(b));
  return out;
}

bool is_int(const py::handle& value) { return py::isinstance<py::int_>(value); }

// hft.feed.event_timestamp
std::int64_t event_timestamp(const py::dict& event) {
  py::object stamp = event.attr("get")("event_ns");
  if (stamp.is_none()) {
    py::object text = event.attr("get")("t");
    if (!py::isinstance<py::str>(text)) {
      throw py::value_error("timestamp must be RFC3339 text");
    }
    try {
      return parse_timestamp(text.cast<std::string>());
    } catch (const std::invalid_argument& error) {
      throw py::value_error(error.what());
    }
  }
  if (!is_int(stamp) || py::isinstance<py::bool_>(stamp) || stamp.cast<long long>() <= 0) {
    throw py::value_error("event_ns must be positive integer nanoseconds");
  }
  return stamp.cast<std::int64_t>();
}

std::optional<std::string> identity(const py::dict& event) {
  if (event.contains("i")) return py::str(event["i"]).cast<std::string>();
  if (event.contains("quote_id")) return py::str(event["quote_id"]).cast<std::string>();
  return std::nullopt;
}

double decimal_float(const py::handle& value, int power) {
  std::string text = py::str(value).cast<std::string>();
  if (power == 0) return py::float_(py::str(text)).cast<double>();
  try {
    return scale_decimal(text, power);
  } catch (const std::invalid_argument&) {
    // Defer to Python for forms the fast path does not handle (e.g. "NaN").
    py::object decimal = py::module_::import("decimal").attr("Decimal");
    return py::float_(decimal(text) * py::int_(100)).cast<double>();
  }
}

bool excluded_trade(const py::dict& event) {
  py::object conditions = event.attr("get")("c", py::list());
  py::object tape = event.attr("get")("z", "C");
  std::string excluded = kCloseExcluded;
  excluded += tape.equal(py::str("C")) ? 'W' : 'B';
  for (const auto& c : conditions) {
    if (!py::isinstance<py::str>(c)) continue;
    auto code = c.cast<std::string>();
    if (code.size() == 1 && excluded.find(code[0]) != std::string::npos) return true;
  }
  return false;
}

struct Parsed {
  enum class Kind { Ignored, Quote, Trade, Correction, Cancellation } kind = Kind::Ignored;
  std::int64_t stamp = 0, arrival = 0;
  Quote quote{};
  double price = 0, size = 0;
  bool excluded = false;
  std::optional<std::string> id;
  std::optional<std::string_view> id_view() const {
    return id ? std::optional<std::string_view>(*id) : std::nullopt;
  }
};

// Python-compatible parsing and validation of hft.feed.BarAggregator.add input.
Parsed parse_event(const std::string& symbol, const py::dict& event, const py::object& arrival_ns) {
  Parsed p;
  if (!event.attr("get")("S").equal(py::str(symbol))) return p;
  py::object kind_obj = event.attr("get")("T");
  if (!py::isinstance<py::str>(kind_obj)) return p;
  auto kind = kind_obj.cast<std::string>();
  if (kind != "q" && kind != "t" && kind != "c" && kind != "x") return p;
  p.stamp = event_timestamp(event);
  py::object arrival_obj =
      arrival_ns.is_none() ? event.attr("get")("arrival_ns", p.stamp) : arrival_ns;
  if (!is_int(arrival_obj) || arrival_obj.cast<long long>() < 0) {
    throw py::value_error("invalid arrival_ns");
  }
  p.arrival = arrival_obj.cast<std::int64_t>();
  p.id = identity(event);
  if (kind == "c" || kind == "x") {
    p.kind = kind == "x" ? Parsed::Kind::Cancellation : Parsed::Kind::Correction;
  } else if (kind == "q") {
    bool shares = py::bool_(event.attr("get")("sizes_in_shares"));
    p.kind = Parsed::Kind::Quote;
    p.quote = Quote{p.stamp,
                    p.arrival,
                    decimal_float(event["bp"], 0),
                    decimal_float(event["ap"], 0),
                    decimal_float(event["bs"], shares ? 0 : 2),
                    decimal_float(event["as"], shares ? 0 : 2)};
  } else {
    p.kind = Parsed::Kind::Trade;
    p.price = py::float_(event["p"]).cast<double>();
    p.size = py::float_(event["s"]).cast<double>();
    p.excluded = excluded_trade(event);
  }
  return p;
}

template <class Fn>
auto rethrow_value_errors(Fn&& fn) -> decltype(fn()) {
  try {
    return fn();
  } catch (const std::invalid_argument& error) {
    throw py::value_error(error.what());
  } catch (const std::logic_error& error) {
    throw py::value_error(error.what());
  }
}

py::list add_event(BarAggregator& agg, const py::dict& event, const py::object& arrival_ns) {
  Parsed p = parse_event(agg.symbol(), event, arrival_ns);
  return rethrow_value_errors([&] {
    switch (p.kind) {
      case Parsed::Kind::Ignored: return py::list();
      case Parsed::Kind::Correction:
      case Parsed::Kind::Cancellation:
        return bar_list(agg.add_correction(p.stamp, p.arrival,
                                           p.kind == Parsed::Kind::Cancellation));
      case Parsed::Kind::Quote: return bar_list(agg.add_quote(p.quote, p.id_view()));
      case Parsed::Kind::Trade:
        return bar_list(
            agg.add_trade(p.stamp, p.arrival, p.price, p.size, p.excluded, p.id_view()));
    }
    return py::list();
  });
}

py::object update_object(const Update& update) {
  if (const auto* gap = std::get_if<GapEvent>(&update)) {
    return py::make_tuple("gap", gap->now_ns, gap->reason);
  }
  const auto& u = std::get<BarUpdate>(update);
  py::object market = py::none();
  if (u.market) {
    py::list values;
    for (float v : *u.market) values.append(static_cast<double>(v));
    market = values;
  }
  return py::make_tuple("bar", bar_tuple(u.bar), u.now_ns, u.accepted,
                        u.reset ? py::object(py::str(*u.reset)) : py::object(py::none()), u.ready,
                        market, u.action ? py::object(py::int_(*u.action)) : py::object(py::none()));
}

py::list update_list(const std::vector<Update>& updates) {
  py::list out;
  for (const auto& u : updates) out.append(update_object(u));
  return out;
}

// hft.market_engine.PyMarketEngine.on_event: the arrival-gap check runs for every
// event, including ones the aggregator ignores.
py::list engine_event(MarketEngine& engine, const py::dict& event, std::int64_t arrival_ns) {
  if (!engine.aggregator()) throw py::value_error("start a session before market events");
  Parsed p = parse_event(engine.symbol(), event, py::int_(arrival_ns));
  return rethrow_value_errors([&] {
    switch (p.kind) {
      case Parsed::Kind::Ignored: return update_list(engine.on_ignored(arrival_ns));
      case Parsed::Kind::Correction:
      case Parsed::Kind::Cancellation:
        return update_list(engine.on_correction(p.stamp, p.arrival,
                                                p.kind == Parsed::Kind::Cancellation));
      case Parsed::Kind::Quote: return update_list(engine.on_quote(p.quote, p.id_view()));
      case Parsed::Kind::Trade:
        return update_list(
            engine.on_trade(p.stamp, p.arrival, p.price, p.size, p.excluded, p.id_view()));
    }
    return py::list();
  });
}

Strategy parse_strategy_name(const std::string& name) {
  if (name == "model") return Strategy::Model;
  if (name == "ema-crossover") return Strategy::EmaCrossover;
  if (name == "hold-day") return Strategy::HoldDay;
  throw py::value_error("unknown strategy '" + name + "'");
}

}  // namespace

PYBIND11_MODULE(hftcore, m) {
  m.doc() = "Low-latency market-data replay and decision engine";
  m.def("version", [] { return std::string(hftcore::version()); },
        "Engine version recorded in run identities");

  py::class_<BarAggregator>(m, "BarAggregator")
      .def(py::init([](std::string symbol, int bar_seconds, std::optional<std::string> session_id,
                       std::optional<std::int64_t> open_ns, std::optional<std::int64_t> close_ns,
                       std::int64_t late_tolerance_ns, std::size_t max_events) {
             std::optional<Session> session;
             if (session_id) session = Session{*session_id, open_ns.value(), close_ns.value()};
             try {
               return BarAggregator(std::move(symbol), bar_seconds, session, late_tolerance_ns,
                                    max_events);
             } catch (const std::invalid_argument& error) {
               throw py::value_error(error.what());
             }
           }),
           py::arg("symbol"), py::arg("bar_seconds") = 5, py::arg("session_id") = py::none(),
           py::arg("open_ns") = py::none(), py::arg("close_ns") = py::none(),
           py::arg("late_tolerance_ns") = 250'000'000, py::arg("max_events") = 100'000)
      .def("add", &add_event, py::arg("event"), py::arg("arrival_ns") = py::none())
      .def(
          "advance_to",
          [](BarAggregator& agg, const py::object& now_ns) {
            if (!is_int(now_ns)) throw py::value_error("clock must be integer nanoseconds");
            return bar_list(agg.advance_to(now_ns.cast<std::int64_t>()));
          },
          py::arg("now_ns"))
      .def("quality", [](const BarAggregator& agg) { return agg.quality(); })
      .def_property(
          "start", [](const BarAggregator& agg) { return agg.start(); },
          [](BarAggregator& agg, std::int64_t start) { agg.set_start(start); });

  py::class_<MarketEngine>(m, "MarketEngine")
      .def(py::init([](std::string symbol, const std::string& strategy, int bar_seconds) {
             return MarketEngine(std::move(symbol), parse_strategy_name(strategy), bar_seconds);
           }),
           py::arg("symbol"), py::arg("strategy") = "model", py::arg("bar_seconds") = 5)
      .def(
          "start_session",
          [](MarketEngine& e, std::string session_id, std::int64_t open_ns, std::int64_t close_ns,
             std::optional<std::int64_t> now_ns) {
            e.start_session(Session{std::move(session_id), open_ns, close_ns}, now_ns);
          },
          py::arg("session_id"), py::arg("open_ns"), py::arg("close_ns"),
          py::arg("now_ns") = py::none())
      .def("end_session", &MarketEngine::end_session)
      .def("reset_history", &MarketEngine::reset_history)
      .def("on_event", &engine_event, py::arg("event"), py::arg("arrival_ns"))
      .def(
          "advance_to",
          [](MarketEngine& e, std::int64_t now_ns) {
            return rethrow_value_errors([&] { return update_list(e.advance_to(now_ns)); });
          },
          py::arg("now_ns"))
      .def("history",
           [](const MarketEngine& e) {
             py::list out;
             for (const auto& b : e.history()) out.append(bar_tuple(b));
             return out;
           })
      .def("quality",
           [](MarketEngine& e) {
             return e.aggregator() ? e.aggregator()->quality()
                                   : std::map<std::string, std::int64_t>{};
           })
      .def("aggregator_start",
           [](MarketEngine& e) {
             return e.aggregator() ? e.aggregator()->start() : std::nullopt;
           })
      .def_property_readonly("last_event_ns", &MarketEngine::last_event_ns)
      .def_property_readonly("last_tick_ns", &MarketEngine::last_tick_ns)
      .def_property_readonly("warmup_after_ns", &MarketEngine::warmup_after_ns);
}
