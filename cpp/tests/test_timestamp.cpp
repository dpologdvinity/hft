#include <catch2/catch_test_macros.hpp>

#include "hftcore/timestamp.hpp"

using hftcore::parse_timestamp;

TEST_CASE("RFC3339 timestamps parse to UTC nanoseconds") {
  REQUIRE(parse_timestamp("1970-01-01T00:00:00Z") == 0);
  REQUIRE(parse_timestamp("2025-01-06T14:30:00.123456789Z") == 1736173800123456789LL);
  REQUIRE(parse_timestamp("2025-01-06T09:30:00.5-05:00") == 1736173800500000000LL);
  REQUIRE(parse_timestamp("2024-02-29T00:00:00+00:00") == 1709164800000000000LL);
}

TEST_CASE("malformed timestamps are rejected") {
  for (const char* bad : {"2025-01-06T14:30:00", "2025-01-06 14:30:00Z", "2025-13-01T00:00:00Z",
                          "2025-02-29T00:00:00Z", "2025-01-06T14:30:00.1234567890Z",
                          "2025-01-06T24:00:00Z"}) {
    REQUIRE_THROWS_AS(parse_timestamp(bad), std::invalid_argument);
  }
}
