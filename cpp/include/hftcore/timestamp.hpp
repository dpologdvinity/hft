#pragma once

#include <cstdint>
#include <string_view>

namespace hftcore {

// Same contract as hft.feed.parse_timestamp: RFC3339 with Z or +-HH:MM and up
// to nine fraction digits; returns UTC epoch nanoseconds. Throws
// std::invalid_argument on malformed or out-of-range input.
std::int64_t parse_timestamp(std::string_view text);

}  // namespace hftcore
