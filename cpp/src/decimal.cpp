#include "hftcore/decimal.hpp"

#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <system_error>

namespace hftcore {

std::string_view shortest_repr(double value, char (&buffer)[32]) {
  auto [end, ec] = std::to_chars(buffer, buffer + sizeof buffer, value);
  if (ec != std::errc()) throw std::invalid_argument("unrepresentable number");
  return {buffer, static_cast<std::size_t>(end - buffer)};
}

double scale_decimal(std::string_view text, int power) {
  // Split mantissa and exponent, then shift the exponent: exact in decimal.
  std::string mantissa;
  long exponent = 0;
  auto e = text.find_first_of("eE");
  std::string_view body = text.substr(0, e);
  if (e != std::string_view::npos) {
    std::string tail(text.substr(e + 1));
    char* stop = nullptr;
    exponent = std::strtol(tail.c_str(), &stop, 10);
    if (stop == tail.c_str() || *stop != '\0') throw std::invalid_argument("invalid decimal");
  }
  mantissa.assign(body);
  std::string scaled = mantissa + "e" + std::to_string(exponent + power);
  double result = 0;
  auto [ptr, ec] = std::from_chars(scaled.data(), scaled.data() + scaled.size(), result);
  if (ec != std::errc() || ptr != scaled.data() + scaled.size()) {
    throw std::invalid_argument("invalid decimal");
  }
  return result;
}

}  // namespace hftcore
