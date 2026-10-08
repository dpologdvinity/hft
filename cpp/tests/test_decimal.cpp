#include <catch2/catch_test_macros.hpp>

#include "hftcore/decimal.hpp"

using hftcore::scale_decimal;

TEST_CASE("decimal scaling is exact before rounding") {
  // Binary 0.07 * 100 is 7.000000000000001; Decimal("0.07") * 100 is 7.
  REQUIRE(scale_decimal("0.07", 2) == 7.0);
  REQUIRE(scale_decimal("3", 2) == 300.0);
  REQUIRE(scale_decimal("1.5e-3", 2) == 0.15);
  REQUIRE(scale_decimal("2E2", 0) == 200.0);
  REQUIRE_THROWS_AS(scale_decimal("abc", 2), std::invalid_argument);
}

TEST_CASE("shortest repr round-trips") {
  char buffer[32];
  REQUIRE(hftcore::shortest_repr(0.1, buffer) == "0.1");
  REQUIRE(hftcore::shortest_repr(150.25, buffer) == "150.25");
}

#include "hftcore/numeric.hpp"

TEST_CASE("PySum matches CPython's compensated float sum") {
  hftcore::PySum s;
  for (double x : {1e16, 1.0, -1e16}) s.add(x);
  REQUIRE(s.value() == 1.0);  // a naive running sum gives 0.0
  hftcore::PySum t;
  for (double x : {0.1, 0.1, 0.1}) t.add(x);
  REQUIRE(t.value() == 0.30000000000000004);
}
