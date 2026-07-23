/*
  YaneuraOu PackedSfen (32-byte) position key.

  Numeric fields inside a YBB file are little-endian, while PackedSfen itself
  is an opaque byte string sorted with memcmp().
*/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace lczero {

struct PackedSfen {
  std::array<uint8_t, 32> data{};

  bool operator==(const PackedSfen& other) const {
    return data == other.data;
  }

  bool operator!=(const PackedSfen& other) const {
    return !(*this == other);
  }
};

inline int ComparePackedSfen(const PackedSfen& lhs, const PackedSfen& rhs) {
  return std::memcmp(lhs.data.data(), rhs.data.data(), lhs.data.size());
}

static_assert(sizeof(PackedSfen) == 32);

}  // namespace lczero
