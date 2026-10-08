#pragma once

#include <string_view>

namespace hftcore {

// float(Decimal(text) * 10**power), correctly rounded, without binary rounding
// of the intermediate product (e.g. "0.07" lots * 100 == 7.0 exactly).
double scale_decimal(std::string_view text, int power);

// Shortest round-trip text of a double, as Python's repr(float) produces.
std::string_view shortest_repr(double value, char (&buffer)[32]);

}  // namespace hftcore
