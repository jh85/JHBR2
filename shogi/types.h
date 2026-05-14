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

// Shogi type definitions adapted from lc0 chess types.
// Board representation follows YaneuraOu conventions:
//   - 9×9 board, 81 squares
//   - Square indexing: file * 9 + rank (file 0 = right, rank 0 = top)
//   - 180° rotation for perspective flip: Flip(sq) = 80 - sq

#pragma once

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace lczero {

// =====================================================================
// Color (side to move)
// =====================================================================
// BLACK = sente (first player), WHITE = gote (second player).

enum Color : uint8_t { BLACK = 0, WHITE = 1, COLOR_NB = 2 };

constexpr Color operator~(Color c) { return Color(c ^ 1); }

// =====================================================================
// PieceType — 14 piece types (8 base + 6 promoted), no color.
// =====================================================================
// Encoding chosen so that promoted = base | kPromoteBit (value 8).
// The first 7 types (PAWN..GOLD) are droppable hand pieces.
// KING (8) is not droppable.  Promoted types are 9..14.

struct PieceType {
  uint8_t idx;

  static constexpr PieceType FromIdx(uint8_t i) { return PieceType{i}; }

  // Can this piece type be held in hand? (PAWN..GOLD, indices 1..7)
  constexpr bool IsHandPiece() const { return idx >= 1 && idx <= 7; }

  // Is this a promoted piece? (idx >= 9)
  constexpr bool IsPromoted() const { return idx >= 9; }

  // Return the promoted version (only valid for PAWN..ROOK, not GOLD/KING).
  constexpr PieceType Promote() const {
    return PieceType{static_cast<uint8_t>(idx | 8)};
  }

  // Return the unpromoted (base) version.
  constexpr PieceType Unpromote() const {
    return PieceType{static_cast<uint8_t>(idx & 7)};
  }

  // Can this piece promote? (PAWN, LANCE, KNIGHT, SILVER, BISHOP, ROOK)
  constexpr bool CanPromote() const {
    return idx >= 1 && idx <= 6;
  }

  bool IsValid() const { return idx >= 1 && idx <= 14 && idx != 15; }

  bool operator==(const PieceType& o) const { return idx == o.idx; }
  bool operator!=(const PieceType& o) const { return idx != o.idx; }

  // USI piece character (uppercase).
  char ToChar() const;

  // Parse USI piece character (case-insensitive for base pieces).
  static PieceType Parse(char c);

 private:
  constexpr explicit PieceType(uint8_t i) : idx(i) {}
};

// Base (unpromotable) pieces.
constexpr PieceType kNoPieceType = PieceType::FromIdx(0);
constexpr PieceType kPawn        = PieceType::FromIdx(1);
constexpr PieceType kLance       = PieceType::FromIdx(2);
constexpr PieceType kKnight      = PieceType::FromIdx(3);
constexpr PieceType kSilver      = PieceType::FromIdx(4);
constexpr PieceType kBishop      = PieceType::FromIdx(5);
constexpr PieceType kRook        = PieceType::FromIdx(6);
constexpr PieceType kGold        = PieceType::FromIdx(7);
constexpr PieceType kKing        = PieceType::FromIdx(8);

// Promoted pieces (base idx | 8).
constexpr PieceType kProPawn     = PieceType::FromIdx(9);   // Tokin
constexpr PieceType kProLance    = PieceType::FromIdx(10);
constexpr PieceType kProKnight   = PieceType::FromIdx(11);
constexpr PieceType kProSilver   = PieceType::FromIdx(12);
constexpr PieceType kHorse       = PieceType::FromIdx(13);  // Promoted bishop
constexpr PieceType kDragon      = PieceType::FromIdx(14);  // Promoted rook

// Number of piece types that can be in hand (PAWN..GOLD = 7).
constexpr int kHandPieceTypes = 7;

// Total piece types including promoted (excluding kNoPieceType).
constexpr int kPieceTypeNB = 15;

// Promotion offset: promoted_idx = base_idx | kPromoteBit.
constexpr uint8_t kPromoteBit = 8;

// =====================================================================
// File (column, 1-9 in Shogi notation, 0-indexed internally)
// =====================================================================
// File 0 = rightmost column from BLACK's perspective (file "1" in USI).
// File 8 = leftmost column (file "9" in USI).

struct File {
  uint8_t idx;

  static constexpr File FromIdx(uint8_t i) { return File{i}; }
  static constexpr File Parse(char c) { return File{static_cast<uint8_t>(c - '1')}; }

  constexpr bool IsValid() const { return idx < 9; }
  std::string ToString() const { return std::string(1, '1' + idx); }

  // Flip file for 180° rotation: file → 8 - file.
  void Flip() { idx = 8 - idx; }

  auto operator<=>(const File& o) const = default;
  void operator++() { ++idx; }
  void operator--() { --idx; }
  File operator+(int d) const { return File{static_cast<uint8_t>(idx + d)}; }
  File operator-(int d) const { return File{static_cast<uint8_t>(idx - d)}; }

 private:
  constexpr explicit File(uint8_t i) : idx(i) {}
};

constexpr File kFile1 = File::FromIdx(0), kFile2 = File::FromIdx(1),
               kFile3 = File::FromIdx(2), kFile4 = File::FromIdx(3),
               kFile5 = File::FromIdx(4), kFile6 = File::FromIdx(5),
               kFile7 = File::FromIdx(6), kFile8 = File::FromIdx(7),
               kFile9 = File::FromIdx(8);

// =====================================================================
// Rank (row, 'a'-'i' in USI notation, 0-indexed internally)
// =====================================================================
// Rank 0 = top row from BLACK's perspective (rank "a" in USI).
// Rank 8 = bottom row (rank "i" in USI).

struct Rank {
  uint8_t idx;

  static constexpr Rank FromIdx(uint8_t i) { return Rank{i}; }
  static constexpr Rank Parse(char c) { return Rank{static_cast<uint8_t>(c - 'a')}; }

  constexpr bool IsValid() const { return idx < 9; }
  std::string ToString() const { return std::string(1, 'a' + idx); }

  // Flip rank for 180° rotation: rank → 8 - rank.
  void Flip() { idx = 8 - idx; }

  // Is this rank in the promotion zone for the given color?
  // BLACK promotes in ranks 0,1,2 (enemy territory = top 3 rows).
  // WHITE promotes in ranks 6,7,8.
  constexpr bool InPromotionZone(Color c) const {
    return c == BLACK ? idx <= 2 : idx >= 6;
  }

  auto operator<=>(const Rank& o) const = default;
  void operator++() { ++idx; }
  void operator--() { --idx; }
  Rank operator+(int d) const { return Rank{static_cast<uint8_t>(idx + d)}; }
  Rank operator-(int d) const { return Rank{static_cast<uint8_t>(idx - d)}; }

 private:
  constexpr explicit Rank(uint8_t i) : idx(i) {}
};

constexpr Rank kRank1 = Rank::FromIdx(0), kRank2 = Rank::FromIdx(1),
               kRank3 = Rank::FromIdx(2), kRank4 = Rank::FromIdx(3),
               kRank5 = Rank::FromIdx(4), kRank6 = Rank::FromIdx(5),
               kRank7 = Rank::FromIdx(6), kRank8 = Rank::FromIdx(7),
               kRank9 = Rank::FromIdx(8);

// =====================================================================
// Square — a single board position (0-80)
// =====================================================================
// Layout: square = file * 9 + rank.
// This matches YaneuraOu's convention.
//   Square 0  = File 1, Rank a (1a, top-right from BLACK's view)
//   Square 80 = File 9, Rank i (9i, bottom-left)
//
// 180° rotation (perspective flip): Flip(sq) = 80 - sq.

class Square {
 public:
  constexpr Square() : idx_(81) {}
  constexpr Square(File f, Rank r) : idx_(f.idx * 9 + r.idx) {}

  static constexpr Square FromIdx(uint8_t i) { return Square{i}; }

  static constexpr Square Parse(std::string_view s) {
    return Square(File::Parse(s[0]), Rank::Parse(s[1]));
  }

  constexpr File file() const { return File::FromIdx(idx_ / 9); }
  constexpr Rank rank() const { return Rank::FromIdx(idx_ % 9); }

  // 180° rotation for BLACK ↔ WHITE perspective flip.
  void Flip() { idx_ = 80 - idx_; }
  constexpr Square Flipped() const { return Square{static_cast<uint8_t>(80 - idx_)}; }

  // Is this square in the promotion zone for the given color?
  constexpr bool InPromotionZone(Color c) const {
    return rank().InPromotionZone(c);
  }

  std::string ToString() const { return file().ToString() + rank().ToString(); }

  constexpr bool IsValid() const { return idx_ < 81; }
  constexpr uint8_t as_idx() const { return idx_; }

  constexpr bool operator==(const Square& o) const { return idx_ == o.idx_; }
  constexpr bool operator!=(const Square& o) const { return idx_ != o.idx_; }

 private:
  constexpr explicit Square(uint8_t i) : idx_(i) {}
  uint8_t idx_;
};

constexpr int kBoardSize = 9;
constexpr int kSquareNB = 81;

// A few notable squares.
constexpr Square kSquare1a = Square(kFile1, kRank1);  // 0
constexpr Square kSquare5e = Square(kFile5, kRank5);  // center
constexpr Square kSquare9i = Square(kFile9, kRank9);  // 80
constexpr Square kSquareNone = Square();               // 81 (invalid)

// Direction offsets for square arithmetic (matching YaneuraOu).
// In file*9+rank layout:
constexpr int kDirUp    = -1;   // toward rank a (top)
constexpr int kDirDown  = +1;   // toward rank i (bottom)
constexpr int kDirRight = -9;   // toward file 1 (right)
constexpr int kDirLeft  = +9;   // toward file 9 (left)
constexpr int kDirRU    = kDirRight + kDirUp;    // -10
constexpr int kDirRD    = kDirRight + kDirDown;   // -8
constexpr int kDirLU    = kDirLeft  + kDirUp;     // +8
constexpr int kDirLD    = kDirLeft  + kDirDown;   // +10
constexpr int kDirRUU   = kDirRU + kDirUp;        // -11 (knight right-up)
constexpr int kDirLUU   = kDirLU + kDirUp;        // +7  (knight left-up)

// =====================================================================
// Hand — captured pieces available for dropping (bit-packed uint32_t)
// =====================================================================
// Bit layout (follows YaneuraOu):
//   Pawn:   bits  0-4  (5 bits, max 18 but 5 bits = max 31)
//   Lance:  bits  8-10 (3 bits, max 4)
//   Knight: bits 12-14 (3 bits, max 4)
//   Silver: bits 16-18 (3 bits, max 4)
//   Bishop: bits 20-21 (2 bits, max 2)
//   Rook:   bits 24-25 (2 bits, max 2)
//   Gold:   bits 28-30 (3 bits, max 4)
//
// This encoding allows count queries and add/sub with simple shifts.

class Hand {
 public:
  constexpr Hand() : data_(0) {}
  constexpr explicit Hand(uint32_t raw) : data_(raw) {}

  // Number of pieces of the given type in hand.
  constexpr int Count(PieceType pt) const {
    return (data_ >> kBits[pt.idx]) & kMask[pt.idx];
  }

  // Whether at least one piece of the given type is held.
  constexpr bool Has(PieceType pt) const {
    return data_ & kMask2[pt.idx];
  }

  // Whether any piece at all is held.
  constexpr bool IsEmpty() const { return data_ == 0; }

  // Add one piece to hand.
  void Add(PieceType pt) { data_ += kOne[pt.idx]; }

  // Remove one piece from hand.
  void Sub(PieceType pt) { data_ -= kOne[pt.idx]; }

  // Raw representation for hashing / comparison.
  constexpr uint32_t raw() const { return data_; }

  bool operator==(const Hand& o) const { return data_ == o.data_; }
  bool operator!=(const Hand& o) const { return data_ != o.data_; }

  // Does this hand have at least as many of every piece type as other?
  constexpr bool Dominates(const Hand& other) const;

  // USI string for hand pieces (e.g. "2P1L3S" or "-" if empty).
  std::string ToString(Color c) const;

  // Parse hand part of SFEN string.
  static Hand Parse(const std::string& sfen_hand, Color c);

 private:
  uint32_t data_;

  // Bit positions for each piece type (indexed by PieceType::idx, 0 unused).
  static constexpr int kBits[kPieceTypeNB] = {
    0, 0/*P*/, 8/*L*/, 12/*N*/, 16/*S*/, 20/*B*/, 24/*R*/, 28/*G*/,
    0,0,0,0,0,0,0  // promoted pieces & king: not in hand
  };

  // Bit masks for count extraction.
  static constexpr int kMask[kPieceTypeNB] = {
    0, 31/*P*/, 7/*L*/, 7/*N*/, 7/*S*/, 3/*B*/, 3/*R*/, 7/*G*/,
    0,0,0,0,0,0,0
  };

  // Bit masks shifted to position (for existence check).
  static constexpr uint32_t kMask2[kPieceTypeNB] = {
    0,
    uint32_t(31) << 0,   // P
    uint32_t(7)  << 8,   // L
    uint32_t(7)  << 12,  // N
    uint32_t(7)  << 16,  // S
    uint32_t(3)  << 20,  // B
    uint32_t(3)  << 24,  // R
    uint32_t(7)  << 28,  // G
    0,0,0,0,0,0,0
  };

  // Value of one piece at the correct bit position.
  static constexpr uint32_t kOne[kPieceTypeNB] = {
    0,
    uint32_t(1) << 0,   // P
    uint32_t(1) << 8,   // L
    uint32_t(1) << 12,  // N
    uint32_t(1) << 16,  // S
    uint32_t(1) << 20,  // B
    uint32_t(1) << 24,  // R
    uint32_t(1) << 28,  // G
    0,0,0,0,0,0,0
  };
};

// =====================================================================
// Move — a single Shogi move (32 bits)
// =====================================================================
// Encoding (16 active bits, compatible with YaneuraOu's Move16):
//   bits  0-6:  to square (0-80, 7 bits)
//   bits  7-13: from square (0-80) for board moves,
//               or piece type (1-7) for drop moves
//   bit   14:   drop flag
//   bit   15:   promotion flag
//
// A null/invalid move has data_ == 0.

class Move {
 public:
  Move() = default;

  // Normal board move (from → to), no promotion.
  static constexpr Move Normal(Square from, Square to) {
    return Move(to.as_idx() | (from.as_idx() << 7));
  }

  // Board move with promotion.
  static constexpr Move Promotion(Square from, Square to) {
    return Move(to.as_idx() | (from.as_idx() << 7) | kPromoteFlag);
  }

  // Drop move: place a hand piece on the board.
  static constexpr Move Drop(PieceType pt, Square to) {
    return Move(to.as_idx() | (pt.idx << 7) | kDropFlag);
  }

  constexpr Square to() const { return Square::FromIdx(data_ & 0x7F); }
  constexpr Square from() const { return Square::FromIdx((data_ >> 7) & 0x7F); }

  constexpr bool is_drop() const { return data_ & kDropFlag; }
  constexpr bool is_promotion() const { return data_ & kPromoteFlag; }
  constexpr bool is_null() const { return data_ == 0; }

  // For drop moves: the piece type being dropped.
  constexpr PieceType drop_piece() const {
    return PieceType::FromIdx((data_ >> 7) & 0x7F);
  }

  // Flip the move for perspective change (180° rotation).
  void Flip() {
    if (!is_drop()) {
      uint8_t f = 80 - ((data_ >> 7) & 0x7F);  // flip from
      uint8_t t = 80 - (data_ & 0x7F);           // flip to
      data_ = (data_ & kFlagMask) | t | (f << 7);
    } else {
      uint8_t t = 80 - (data_ & 0x7F);           // flip to only
      data_ = (data_ & ~uint16_t(0x7F)) | t;
    }
  }

  // USI format string (e.g. "7g7f", "2d2c+", "P*5e").
  std::string ToString() const;

  // Parse USI move string.
  static Move Parse(const std::string& usi);

  bool operator==(const Move& o) const { return data_ == o.data_; }
  bool operator!=(const Move& o) const { return data_ != o.data_; }

  uint16_t raw() const { return data_; }

 private:
  constexpr explicit Move(uint16_t d) : data_(d) {}

  uint16_t data_ = 0;

  static constexpr uint16_t kDropFlag    = 1 << 14;
  static constexpr uint16_t kPromoteFlag = 1 << 15;
  static constexpr uint16_t kFlagMask    = kDropFlag | kPromoteFlag;
};

// Stack-allocated move list. Max legal moves in shogi is ~593.
class MoveList {
 public:
  MoveList() = default;
  void push_back(Move m) { assert(count_ < kMaxMoves); moves_[count_++] = m; }
  int size() const { return count_; }
  bool empty() const { return count_ == 0; }
  void reserve(int) {}  // no-op, for compatibility
  Move& operator[](int i) { return moves_[i]; }
  const Move& operator[](int i) const { return moves_[i]; }
  Move* begin() { return moves_; }
  Move* end() { return moves_ + count_; }
  const Move* begin() const { return moves_; }
  const Move* end() const { return moves_ + count_; }

 private:
  static constexpr int kMaxMoves = 600;
  Move moves_[kMaxMoves];
  int count_ = 0;
};

// =====================================================================
// Inline implementations
// =====================================================================

inline char PieceType::ToChar() const {
  constexpr const char* chars = ".PLNSBRGK+l+n+s+h+d";
  // This is approximate; promoted pieces need multi-char in USI.
  if (idx <= 8) return ".PLNSBRGK"[idx];
  return '+';  // Promoted pieces start with '+' in USI
}

inline PieceType PieceType::Parse(char c) {
  switch (c) {
    case 'P': case 'p': return kPawn;
    case 'L': case 'l': return kLance;
    case 'N': case 'n': return kKnight;
    case 'S': case 's': return kSilver;
    case 'B': case 'b': return kBishop;
    case 'R': case 'r': return kRook;
    case 'G': case 'g': return kGold;
    case 'K': case 'k': return kKing;
    default: return kNoPieceType;
  }
}

inline std::string Move::ToString() const {
  if (is_null()) return "0000";
  if (is_drop()) {
    // USI drop: "P*5e"
    return std::string(1, drop_piece().ToChar()) + "*" + to().ToString();
  }
  // USI board move: "7g7f" or "7g7f+" (promotion)
  return from().ToString() + to().ToString() + (is_promotion() ? "+" : "");
}

inline Move Move::Parse(const std::string& usi) {
  if (usi.size() < 4) return Move();
  if (usi[1] == '*') {
    // Drop: "P*5e"
    PieceType pt = PieceType::Parse(usi[0]);
    Square to = Square::Parse(usi.substr(2, 2));
    return Move::Drop(pt, to);
  }
  Square from = Square::Parse(usi.substr(0, 2));
  Square to = Square::Parse(usi.substr(2, 2));
  if (usi.size() >= 5 && usi[4] == '+') {
    return Move::Promotion(from, to);
  }
  return Move::Normal(from, to);
}

inline int operator-(File a, File b) { return int(a.idx) - b.idx; }
inline int operator-(Rank a, Rank b) { return int(a.idx) - b.idx; }

}  // namespace lczero
