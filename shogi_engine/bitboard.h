/*
  This file is part of Leela Shogi Zero (adapted from Leela Chess Zero).
  Copyright (C) 2025 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

// 81-bit Shogi bitboard using two uint64_t.
//
// Bit layout (vertical/columnar, following YaneuraOu):
//   p[0] bits  0-62:  squares  0-62  (files 1-7, 9 ranks each = 63 squares)
//   p[0] bit   63:    unused (always 0)
//   p[1] bits  0-17:  squares 63-80  (files 8-9, 9 ranks each = 18 squares)
//   p[1] bits 18-63:  unused (always 0)
//
// Square-to-bit mapping:
//   Square  0-62  → p[0], bit = square
//   Square 63-80  → p[1], bit = square - 63
//
// Direction shifts (within one file, shift by 1 = one rank):
//   Left shift  (<<1) = move one rank down (toward rank i)
//   Right shift (>>1) = move one rank up   (toward rank a)

#pragma once

#include <bit>
#include <cstdint>
#include <functional>
#include <string>

#include "shogi/types.h"

namespace lczero {

// The boundary between p[0] and p[1]: squares 0-62 in p[0], 63-80 in p[1].
constexpr int kBBSplit = 63;

// Masks for valid bits.
constexpr uint64_t kBBMask0 = UINT64_C(0x7FFFFFFFFFFFFFFF);  // bits 0-62
constexpr uint64_t kBBMask1 = UINT64_C(0x000000000003FFFF);  // bits 0-17

class Bitboard {
 public:
  // --- constructors ---

  // Uninitialized.
  Bitboard() = default;

  // Zero bitboard.
  static constexpr Bitboard Zero() {
    return Bitboard(UINT64_C(0), UINT64_C(0));
  }

  // All squares set (81 bits).
  static constexpr Bitboard All() {
    return Bitboard(kBBMask0, kBBMask1);
  }

  // Single square.
  static constexpr Bitboard FromSquare(Square sq) {
    uint8_t i = sq.as_idx();
    if (i < kBBSplit) {
      return Bitboard(UINT64_C(1) << i, 0);
    } else {
      return Bitboard(0, UINT64_C(1) << (i - kBBSplit));
    }
  }

  // From raw values.
  static constexpr Bitboard FromRaw(uint64_t p0, uint64_t p1) {
    return Bitboard(p0, p1);
  }

  // --- bit queries ---

  // Test if a square is set.
  constexpr bool Test(Square sq) const {
    uint8_t i = sq.as_idx();
    if (i < kBBSplit) return p_[0] & (UINT64_C(1) << i);
    return p_[1] & (UINT64_C(1) << (i - kBBSplit));
  }

  // Test if any bit is set.
  constexpr bool Any() const { return p_[0] | p_[1]; }
  constexpr bool Empty() const { return !Any(); }

  // Number of set bits.
  int PopCount() const {
    return std::popcount(p_[0]) + std::popcount(p_[1]);
  }

  // More than one bit set?
  bool MoreThanOne() const {
    // If both halves have bits, or either half has >1 bit.
    if (p_[0] && p_[1]) return true;
    return (p_[0] & (p_[0] - 1)) || (p_[1] & (p_[1] - 1));
  }

  // --- bit manipulation ---

  // Set a square.
  void Set(Square sq) {
    uint8_t i = sq.as_idx();
    if (i < kBBSplit) p_[0] |= UINT64_C(1) << i;
    else              p_[1] |= UINT64_C(1) << (i - kBBSplit);
  }

  // Clear a square.
  void Clear(Square sq) {
    uint8_t i = sq.as_idx();
    if (i < kBBSplit) p_[0] &= ~(UINT64_C(1) << i);
    else              p_[1] &= ~(UINT64_C(1) << (i - kBBSplit));
  }

  // Toggle a square.
  void Toggle(Square sq) {
    uint8_t i = sq.as_idx();
    if (i < kBBSplit) p_[0] ^= UINT64_C(1) << i;
    else              p_[1] ^= UINT64_C(1) << (i - kBBSplit);
  }

  // --- pop (extract and remove lowest set bit) ---

  // Remove and return the lowest set square.  Undefined if empty.
  Square Pop() {
    if (p_[0]) {
      int bit = std::countr_zero(p_[0]);
      p_[0] &= p_[0] - 1;
      return Square::FromIdx(bit);
    }
    int bit = std::countr_zero(p_[1]);
    p_[1] &= p_[1] - 1;
    return Square::FromIdx(bit + kBBSplit);
  }

  // Return the lowest set square without removing it.
  Square Peek() const {
    if (p_[0]) return Square::FromIdx(std::countr_zero(p_[0]));
    return Square::FromIdx(std::countr_zero(p_[1]) + kBBSplit);
  }

  // --- raw access ---

  constexpr uint64_t Lo() const { return p_[0]; }
  constexpr uint64_t Hi() const { return p_[1]; }

  // Which half a square belongs to (0 or 1).
  static constexpr int Part(Square sq) { return sq.as_idx() >= kBBSplit; }

  // Merge both halves (only valid when the halves don't overlap — always
  // true for our layout since they cover disjoint bit ranges conceptually,
  // but as uint64 OR they produce a meaningful combined value only for
  // pop_count or similar aggregate operations).
  constexpr uint64_t Merge() const { return p_[0] | p_[1]; }

  // --- bitwise operators ---

  Bitboard& operator|=(const Bitboard& o) { p_[0] |= o.p_[0]; p_[1] |= o.p_[1]; return *this; }
  Bitboard& operator&=(const Bitboard& o) { p_[0] &= o.p_[0]; p_[1] &= o.p_[1]; return *this; }
  Bitboard& operator^=(const Bitboard& o) { p_[0] ^= o.p_[0]; p_[1] ^= o.p_[1]; return *this; }

  Bitboard operator|(const Bitboard& o) const { return Bitboard(p_[0] | o.p_[0], p_[1] | o.p_[1]); }
  Bitboard operator&(const Bitboard& o) const { return Bitboard(p_[0] & o.p_[0], p_[1] & o.p_[1]); }
  Bitboard operator^(const Bitboard& o) const { return Bitboard(p_[0] ^ o.p_[0], p_[1] ^ o.p_[1]); }
  Bitboard operator~() const { return Bitboard(~p_[0] & kBBMask0, ~p_[1] & kBBMask1); }

  // andnot: (~this) & other.
  Bitboard AndNot(const Bitboard& o) const {
    return Bitboard(~p_[0] & o.p_[0], ~p_[1] & o.p_[1]);
  }

  // Shift operations (for pawn/lance effects — shift by 1 = one rank).
  Bitboard& operator<<=(int s) { p_[0] <<= s; p_[1] <<= s; return *this; }
  Bitboard& operator>>=(int s) { p_[0] >>= s; p_[1] >>= s; return *this; }
  Bitboard operator<<(int s) const { return Bitboard(p_[0] << s, p_[1] << s); }
  Bitboard operator>>(int s) const { return Bitboard(p_[0] >> s, p_[1] >> s); }

  // Comparison.
  bool operator==(const Bitboard& o) const { return p_[0] == o.p_[0] && p_[1] == o.p_[1]; }
  bool operator!=(const Bitboard& o) const { return !(*this == o); }

  // --- iteration ---

  // Call f(Square) for each set bit.
  template <typename F>
  void ForEach(F f) const {
    uint64_t lo = p_[0];
    while (lo) {
      int bit = std::countr_zero(lo);
      lo &= lo - 1;
      f(Square::FromIdx(bit));
    }
    uint64_t hi = p_[1];
    while (hi) {
      int bit = std::countr_zero(hi);
      hi &= hi - 1;
      f(Square::FromIdx(bit + kBBSplit));
    }
  }

  // --- conversion for NN input ---

  // Write this bitboard as a 9×9 float array for neural network input.
  // Layout: plane[rank][file] matches the input tensor convention.
  // The square at (file f, rank r) = f*9+r maps to plane[r][f].
  void ToPlane(float* plane_81) const {
    for (int i = 0; i < 81; ++i) {
      int f = i / 9;
      int r = i % 9;
      plane_81[r * 9 + f] = Test(Square::FromIdx(i)) ? 1.0f : 0.0f;
    }
  }

  // --- debug ---

  // Pretty-print the bitboard as a 9×9 grid.
  std::string DebugString() const;

 private:
  constexpr Bitboard(uint64_t p0, uint64_t p1) : p_{p0, p1} {}

  uint64_t p_[2];
};

// --- pre-computed tables (initialized at startup) ---

namespace ShogiTables {

// Bitboard with a single square set, indexed by square.
extern Bitboard SquareBB[kSquareNB];

// Bitboards for each file (all 9 squares in the file set).
extern Bitboard FileBB[kBoardSize];

// Bitboards for each rank (all 9 squares in the rank set).
extern Bitboard RankBB[kBoardSize];

// Promotion zone bitboards for each color.
extern Bitboard PromotionZoneBB[COLOR_NB];

// --- Precomputed step attack tables ---
// Indexed by [square][color].  For king, color is irrelevant (symmetric).

extern Bitboard PawnEffectBB[kSquareNB][COLOR_NB];
extern Bitboard KnightEffectBB[kSquareNB][COLOR_NB];
extern Bitboard SilverEffectBB[kSquareNB][COLOR_NB];
extern Bitboard GoldEffectBB[kSquareNB][COLOR_NB];
extern Bitboard KingEffectBB[kSquareNB];

// Horse extra steps (4 cardinal: up/down/left/right).
extern Bitboard HorseStepBB[kSquareNB];

// Dragon extra steps (4 diagonal: NE/NW/SE/SW).
extern Bitboard DragonStepBB[kSquareNB];

// Lance ray masks (all squares the lance could reach, ignoring blockers).
// LanceMaskBB[sq][BLACK] = squares on same file with rank < sq's rank.
// LanceMaskBB[sq][WHITE] = squares on same file with rank > sq's rank.
extern Bitboard LanceMaskBB[kSquareNB][COLOR_NB];

// --- Fast sliding attack functions ---

// Lance effect using Qugiy bit-subtraction trick. O(1), no loops.
inline Bitboard LanceEffect(Color c, Square sq, const Bitboard& occ) {
  int i = sq.as_idx();
  int part = Bitboard::Part(sq);

  if (c == WHITE) {
    // WHITE moves toward higher bits (toward rank i).
    if (part == 0) {
      uint64_t mask = LanceMaskBB[i][WHITE].Lo();
      uint64_t mocc = occ.Lo() & mask;
      return Bitboard::FromRaw((mocc ^ (mocc - 1)) & mask, 0);
    } else {
      uint64_t mask = LanceMaskBB[i][WHITE].Hi();
      uint64_t mocc = occ.Hi() & mask;
      return Bitboard::FromRaw(0, (mocc ^ (mocc - 1)) & mask);
    }
  } else {
    // BLACK moves toward lower bits (toward rank a).
    if (part == 0) {
      uint64_t mask = LanceMaskBB[i][BLACK].Lo();
      uint64_t mocc = occ.Lo() & mask;
      // MSB trick: find highest set bit in mocc (first blocker).
      // mocc|1 avoids UB when mocc==0; effect becomes full mask.
      int msb = 63 - __builtin_clzll(mocc | 1);
      return Bitboard::FromRaw((UINT64_C(0xFFFFFFFFFFFFFFFF) << msb) & mask, 0);
    } else {
      uint64_t mask = LanceMaskBB[i][BLACK].Hi();
      uint64_t mocc = occ.Hi() & mask;
      int msb = 63 - __builtin_clzll(mocc | 1);
      return Bitboard::FromRaw(0, (UINT64_C(0xFFFFFFFFFFFFFFFF) << msb) & mask);
    }
  }
}

// Rook file (vertical) effect = BLACK lance + WHITE lance.
inline Bitboard RookFileEffect(Square sq, const Bitboard& occ) {
  return LanceEffect(BLACK, sq, occ) | LanceEffect(WHITE, sq, occ);
}

// Initialize all tables.  Must be called once at startup.
void Init();

}  // namespace ShogiTables

// =====================================================================
// Inline implementation of DebugString
// =====================================================================

inline std::string Bitboard::DebugString() const {
  // Print as a 9×9 grid (rank a on top, file 9 on left, file 1 on right).
  // This matches the standard Shogi board orientation for BLACK.
  std::string s;
  s += "  9 8 7 6 5 4 3 2 1\n";
  for (int r = 0; r < 9; ++r) {
    s += char('a' + r);
    s += ' ';
    for (int f = 8; f >= 0; --f) {
      s += Test(Square(File::FromIdx(f), Rank::FromIdx(r))) ? '*' : '.';
      s += ' ';
    }
    s += '\n';
  }
  return s;
}

}  // namespace lczero
