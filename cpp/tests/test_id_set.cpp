#include <catch2/catch_test_macros.hpp>
#include <string>

#include "hftcore/id_set.hpp"

using namespace hftcore;

TEST_CASE("IdSet detects duplicates, grows, and clears in O(1)") {
  IdSet set;
  for (int i = 0; i < 10000; ++i) {
    REQUIRE(set.insert(detail::hash_bytes("id" + std::to_string(i), 1)));
  }
  REQUIRE_FALSE(set.insert(detail::hash_bytes("id42", 1)));
  REQUIRE(set.contains(detail::hash_bytes("id9999", 1)));
  set.clear();
  REQUIRE_FALSE(set.contains(detail::hash_bytes("id42", 1)));
  REQUIRE(set.insert(detail::hash_bytes("id42", 1)));
}

TEST_CASE("hash domains separate identical bytes") {
  REQUIRE_FALSE(detail::hash_bytes("abc", 1) == detail::hash_bytes("abc", 2));
  REQUIRE_FALSE(detail::hash_bytes("abc", 1) == detail::hash_bytes("abd", 1));
  REQUIRE(detail::hash_bytes("", 1) == detail::hash_bytes("", 1));
}
