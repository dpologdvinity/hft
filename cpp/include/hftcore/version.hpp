#pragma once

#include <string_view>

namespace hftcore {

// Bumped whenever engine output could change; recorded in run identities.
std::string_view version() noexcept;

}  // namespace hftcore
