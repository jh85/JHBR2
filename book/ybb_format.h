/*
  YaneuraOu Binary Book (YBB) V1 format helpers.
*/

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>

#include "shogi/packed_sfen.h"

namespace jhbr2 {

constexpr std::array<char, 16> kYbbMagic{
    'Y', 'A', 'N', 'E', '-', 'B', 'I', 'N',
    'B', 'O', 'O', 'K', '-', 'V', '1', '\0',
};
constexpr uint64_t kYbbHeaderSize = 32;
constexpr uint64_t kYbbIndexRecordSize = 44;
constexpr uint64_t kYbbFlagMoveDepth = 1;
constexpr uint64_t kYbbKnownFlags = kYbbFlagMoveDepth;

struct YbbIndexEntry {
  lczero::PackedSfen packed_sfen;
  uint64_t moves_offset = 0;
  uint16_t ply = 0;
  uint16_t move_count = 0;
};

struct YbbMoveRecord {
  uint16_t move = 0;
  int16_t eval = 0;
  uint16_t depth = 0;
};

inline uint16_t ReadLe16(const uint8_t* bytes) {
  return static_cast<uint16_t>(bytes[0]) |
         static_cast<uint16_t>(bytes[1]) << 8;
}

inline uint64_t ReadLe64(const uint8_t* bytes) {
  uint64_t value = 0;
  for (int i = 7; i >= 0; --i) {
    value = (value << 8) | bytes[i];
  }
  return value;
}

inline void StoreLe16(uint8_t* bytes, uint16_t value) {
  bytes[0] = static_cast<uint8_t>(value & 0xff);
  bytes[1] = static_cast<uint8_t>((value >> 8) & 0xff);
}

inline void StoreLe64(uint8_t* bytes, uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    bytes[i] = static_cast<uint8_t>(value & 0xff);
    value >>= 8;
  }
}

inline uint64_t YbbMoveRecordSize(uint64_t flags) {
  return (flags & kYbbFlagMoveDepth) ? 6 : 4;
}

inline bool YbbIndexSize(uint64_t record_count, uint64_t* index_size) {
  if (index_size == nullptr ||
      record_count >
          (std::numeric_limits<uint64_t>::max() - kYbbHeaderSize) /
              kYbbIndexRecordSize) {
    return false;
  }
  *index_size = kYbbHeaderSize + record_count * kYbbIndexRecordSize;
  return true;
}

inline void DecodeYbbIndexEntry(const uint8_t* bytes, YbbIndexEntry* entry) {
  std::memcpy(entry->packed_sfen.data.data(), bytes, 32);
  entry->moves_offset = ReadLe64(bytes + 32);
  entry->ply = ReadLe16(bytes + 40);
  entry->move_count = ReadLe16(bytes + 42);
}

inline void EncodeYbbIndexEntry(const YbbIndexEntry& entry,
                                std::array<uint8_t, 44>* bytes) {
  std::memcpy(bytes->data(), entry.packed_sfen.data.data(), 32);
  StoreLe64(bytes->data() + 32, entry.moves_offset);
  StoreLe16(bytes->data() + 40, entry.ply);
  StoreLe16(bytes->data() + 42, entry.move_count);
}

inline void DecodeYbbMoveRecord(const uint8_t* bytes, uint64_t flags,
                                YbbMoveRecord* move) {
  move->move = ReadLe16(bytes);
  move->eval = static_cast<int16_t>(ReadLe16(bytes + 2));
  move->depth =
      (flags & kYbbFlagMoveDepth) ? ReadLe16(bytes + 4) : uint16_t{0};
}

inline void EncodeYbbMoveRecord(const YbbMoveRecord& move,
                                std::array<uint8_t, 6>* bytes) {
  StoreLe16(bytes->data(), move.move);
  StoreLe16(bytes->data() + 2, static_cast<uint16_t>(move.eval));
  StoreLe16(bytes->data() + 4, move.depth);
}

inline bool WriteYbbHeader(std::ostream& output, uint64_t record_count,
                           uint64_t flags) {
  std::array<uint8_t, 16> tail{};
  StoreLe64(tail.data(), record_count);
  StoreLe64(tail.data() + 8, flags);
  output.write(kYbbMagic.data(), kYbbMagic.size());
  output.write(reinterpret_cast<const char*>(tail.data()), tail.size());
  return static_cast<bool>(output);
}

}  // namespace jhbr2
