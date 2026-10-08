#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>

#include "hftcore/bar_aggregator.hpp"
#include "hftcore/decimal.hpp"
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

// Python-compatible dict path of hft.feed.BarAggregator.add.
py::list add_event(BarAggregator& agg, const py::dict& event, const py::object& arrival_ns) {
  if (!event.attr("get")("S").equal(py::str(agg.symbol()))) return py::list();
  py::object kind_obj = event.attr("get")("T");
  if (!py::isinstance<py::str>(kind_obj)) return py::list();
  auto kind = kind_obj.cast<std::string>();
  if (kind != "q" && kind != "t" && kind != "c" && kind != "x") return py::list();
  std::int64_t stamp = event_timestamp(event);
  py::object arrival_obj =
      arrival_ns.is_none() ? event.attr("get")("arrival_ns", stamp) : arrival_ns;
  if (!is_int(arrival_obj) || arrival_obj.cast<long long>() < 0) {
    throw py::value_error("invalid arrival_ns");
  }
  auto arrival = arrival_obj.cast<std::int64_t>();
  std::optional<std::string> id = identity(event);
  std::optional<std::string_view> id_view;
  if (id) id_view = *id;
  try {
    if (kind == "c" || kind == "x") return bar_list(agg.add_correction(stamp, arrival, kind == "x"));
    if (kind == "q") {
      bool shares = py::bool_(event.attr("get")("sizes_in_shares"));
      Quote q{stamp,
              arrival,
              decimal_float(event["bp"], 0),
              decimal_float(event["ap"], 0),
              decimal_float(event["bs"], shares ? 0 : 2),
              decimal_float(event["as"], shares ? 0 : 2)};
      return bar_list(agg.add_quote(q, id_view));
    }
    double price = py::float_(event["p"]).cast<double>();
    double size = py::float_(event["s"]).cast<double>();
    return bar_list(agg.add_trade(stamp, arrival, price, size, excluded_trade(event), id_view));
  } catch (const std::invalid_argument& error) {
    throw py::value_error(error.what());
  }
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
}
