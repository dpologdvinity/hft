#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

namespace hftcore {

struct Key128 {
  std::uint64_t hi = 0, lo = 0;
  bool operator==(const Key128&) const = default;
};

namespace detail {
inline std::uint64_t mix(std::uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Two independently seeded 64-bit hashes of the bytes: a 128-bit identity.
inline Key128 hash_bytes(std::string_view bytes, std::uint64_t domain) {
  std::uint64_t a = 0x9e3779b97f4a7c15ULL ^ domain, b = 0xc2b2ae3d27d4eb4fULL ^ (domain << 1);
  std::size_t i = 0;
  for (; i + 8 <= bytes.size(); i += 8) {
    std::uint64_t word;
    std::memcpy(&word, bytes.data() + i, 8);
    a = mix(a ^ word);
    b = mix(b + word + 0x632be59bd9b4e019ULL);
  }
  std::uint64_t tail = 0;
  std::memcpy(&tail, bytes.data() + i, bytes.size() - i);
  a = mix(a ^ tail ^ (static_cast<std::uint64_t>(bytes.size()) << 56));
  b = mix(b + tail + bytes.size());
  return {a, b};
}
}  // namespace detail

// Allocation-free set of 128-bit keys with O(1) clear (generation stamps).
class IdSet {
 public:
  IdSet() : slots_(1 << 12) {}

  // Inserts the key; returns false if it was already present.
  bool insert(const Key128& key) {
    if ((count_ + 1) * 2 > slots_.size()) grow();
    std::size_t mask = slots_.size() - 1, i = key.lo & mask;
    while (slots_[i].generation == generation_) {
      if (slots_[i].key == key) return false;
      i = (i + 1) & mask;
    }
    slots_[i] = {key, generation_};
    ++count_;
    return true;
  }

  bool contains(const Key128& key) const {
    std::size_t mask = slots_.size() - 1, i = key.lo & mask;
    while (slots_[i].generation == generation_) {
      if (slots_[i].key == key) return true;
      i = (i + 1) & mask;
    }
    return false;
  }

  void clear() {
    ++generation_;
    count_ = 0;
  }

 private:
  struct Slot {
    Key128 key;
    std::uint64_t generation = 0;
  };
  void grow() {
    std::vector<Slot> old;
    old.swap(slots_);
    slots_.assign(old.size() * 2, Slot{});
    const std::uint64_t live = generation_;
    generation_ = 1;
    count_ = 0;
    for (const auto& s : old) {
      if (s.generation == live) insert(s.key);
    }
  }
  std::vector<Slot> slots_;
  std::uint64_t generation_ = 1;
  std::size_t count_ = 0;
};

}  // namespace hftcore
