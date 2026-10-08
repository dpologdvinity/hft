#include <catch2/catch_test_macros.hpp>

#include "hftcore/version.hpp"

TEST_CASE("engine reports its version") { REQUIRE(hftcore::version() == "0.1.0"); }
