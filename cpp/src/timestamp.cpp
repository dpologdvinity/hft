#include "hftcore/timestamp.hpp"

#include <stdexcept>

#include "hftcore/types.hpp"

namespace hftcore {
namespace {

constexpr const char* kFormat = "timestamp must be timezone-aware RFC3339";

bool digits(std::string_view s, std::size_t pos, std::size_t n, int& out) {
  if (pos + n > s.size()) return false;
  int v = 0;
  for (std::size_t i = pos; i < pos + n; ++i) {
    if (s[i] < '0' || s[i] > '9') return false;
    v = v * 10 + (s[i] - '0');
  }
  out = v;
  return true;
}

// Howard Hinnant's days_from_civil.
std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d) {
  y -= m <= 2;
  const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

bool leap(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

}  // namespace

std::int64_t parse_timestamp(std::string_view s) {
  int year, month, day, hour, minute, second;
  if (s.size() < 20 || !digits(s, 0, 4, year) || s[4] != '-' || !digits(s, 5, 2, month) ||
      s[7] != '-' || !digits(s, 8, 2, day) || s[10] != 'T' || !digits(s, 11, 2, hour) ||
      s[13] != ':' || !digits(s, 14, 2, minute) || s[16] != ':' || !digits(s, 17, 2, second)) {
    throw std::invalid_argument(kFormat);
  }
  std::size_t pos = 19;
  std::int64_t fraction = 0;
  if (s[pos] == '.') {
    std::size_t start = ++pos;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
    std::size_t n = pos - start;
    if (n < 1 || n > 9) throw std::invalid_argument(kFormat);
    for (std::size_t i = start; i < pos; ++i) fraction = fraction * 10 + (s[i] - '0');
    for (std::size_t i = n; i < 9; ++i) fraction *= 10;
  }
  std::int64_t offset = 0;
  if (pos < s.size() && s[pos] == 'Z' && pos + 1 == s.size()) {
    offset = 0;
  } else if (pos + 6 == s.size() && (s[pos] == '+' || s[pos] == '-') && s[pos + 3] == ':') {
    int oh, om;
    if (!digits(s, pos + 1, 2, oh) || !digits(s, pos + 4, 2, om)) {
      throw std::invalid_argument(kFormat);
    }
    // datetime.fromisoformat accepts offsets strictly within +-24h.
    if (oh > 23 || om > 59) throw std::invalid_argument("offset out of range");
    offset = (oh * 3600 + om * 60) * (s[pos] == '-' ? -1 : 1);
  } else {
    throw std::invalid_argument(kFormat);
  }
  static constexpr int kDays[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (year < 1 || month < 1 || month > 12 || day < 1 ||
      day > kDays[month - 1] + (month == 2 && leap(year)) || hour > 23 || minute > 59 ||
      second > 59) {
    throw std::invalid_argument("timestamp field out of range");
  }
  std::int64_t seconds = days_from_civil(year, static_cast<unsigned>(month),
                                         static_cast<unsigned>(day)) * 86400 +
                         hour * 3600 + minute * 60 + second - offset;
  return seconds * kNs + fraction;
}

}  // namespace hftcore
