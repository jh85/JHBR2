/*
  This file is part of Leela Shogi Zero (adapted from Leela Chess Zero).
  Copyright (C) 2025 The LCZero Authors
*/

#include "shogi/encoder.h"

#include <algorithm>
#include <cassert>
#include <set>

namespace lczero {

// =====================================================================
// Direction-based policy mapping (v2: 2187 outputs)
// =====================================================================

namespace {

// Direction vectors: (df, dr) describing movement from source to destination.
// Must match DIRECTION_VECTORS in shogi_model_v2.py exactly.
constexpr int kDirVectors[10][2] = {
  { 0, -1},  // 0: UP
  {-1, -1},  // 1: UP_LEFT
  { 1, -1},  // 2: UP_RIGHT
  {-1,  0},  // 3: LEFT
  { 1,  0},  // 4: RIGHT
  { 0,  1},  // 5: DOWN
  {-1,  1},  // 6: DOWN_LEFT
  { 1,  1},  // 7: DOWN_RIGHT
  {-1, -2},  // 8: UP2_LEFT (knight)
  { 1, -2},  // 9: UP2_RIGHT (knight)
};

// Sliding directions (can move multiple squares).
constexpr bool kIsSliding[10] = {
  true, true, true, true, true, true, true, true, false, false
};

// Lookup table: g_direction[from * 81 + to] = direction index (0-9), or -1.
int g_direction[81 * 81];

void InitDirectionTable() {
  std::fill(g_direction, g_direction + 81 * 81, -1);

  for (int from_sq = 0; from_sq < 81; ++from_sq) {
    int from_f = from_sq / 9;
    int from_r = from_sq % 9;

    for (int dir = 0; dir < 10; ++dir) {
      int df = kDirVectors[dir][0];
      int dr = kDirVectors[dir][1];

      if (kIsSliding[dir]) {
        // Sliding: all distances
        for (int dist = 1; dist < 9; ++dist) {
          int to_f = from_f + df * dist;
          int to_r = from_r + dr * dist;
          if (to_f < 0 || to_f >= 9 || to_r < 0 || to_r >= 9) break;
          g_direction[from_sq * 81 + to_f * 9 + to_r] = dir;
        }
      } else {
        // Step (knight): exact distance
        int to_f = from_f + df;
        int to_r = from_r + dr;
        if (to_f >= 0 && to_f < 9 && to_r >= 0 && to_r < 9) {
          g_direction[from_sq * 81 + to_f * 9 + to_r] = dir;
        }
      }
    }
  }
}

}  // anonymous namespace

// =====================================================================
// Public API
// =====================================================================

namespace ShogiEncoderTables {
void Init() {
  InitDirectionTable();
}
}  // namespace ShogiEncoderTables

// --- Input encoding ---

ShogiInputPlanes EncodeShogiPosition(const ShogiBoard& board) {
  ShogiInputPlanes planes;

  // Flip board if WHITE to move (always encode from mover's perspective).
  ShogiBoard b = (board.side_to_move() == WHITE) ? board.Flipped() : board;
  // After flipping, side_to_move is always effectively BLACK.

  Color us = BLACK;
  Color them = WHITE;

  // Planes 0-13: Our 14 piece types.
  const PieceType piece_types[] = {
    kPawn, kLance, kKnight, kSilver, kBishop, kRook, kGold, kKing,
    kProPawn, kProLance, kProKnight, kProSilver, kHorse, kDragon
  };

  for (int i = 0; i < 14; ++i) {
    planes[i].SetFromBitboard(b.pieces(us, piece_types[i]));
  }

  // Planes 14-27: Their 14 piece types.
  for (int i = 0; i < 14; ++i) {
    planes[14 + i].SetFromBitboard(b.pieces(them, piece_types[i]));
  }

  // Plane 28: Repetition flag (1 if current position has occurred before).
  if (b.IsRepetition()) {
    planes[28].SetAll(1.0f);
  } else {
    planes[28].Clear();
  }

  // Planes 29-35: Our hand piece counts.
  const PieceType hand_types[] = {
    kPawn, kLance, kKnight, kSilver, kBishop, kRook, kGold
  };
  for (int i = 0; i < 7; ++i) {
    float count = static_cast<float>(b.hand(us).Count(hand_types[i]));
    planes[29 + i].SetAll(count);
  }

  // Planes 36-42: Their hand piece counts.
  for (int i = 0; i < 7; ++i) {
    float count = static_cast<float>(b.hand(them).Count(hand_types[i]));
    planes[36 + i].SetAll(count);
  }

  // Plane 43: All ones.
  planes[43].SetAll(1.0f);

  // Planes 44-47: Entering-king (nyugyoku) progress features.
  auto our_ek = b.ComputeEnteringKingInfo(us);
  auto their_ek = b.ComputeEnteringKingInfo(them);
  planes[44].SetAll(static_cast<float>(our_ek.points) / 28.0f);
  planes[45].SetAll(static_cast<float>(their_ek.points) / 28.0f);
  planes[46].SetAll(static_cast<float>(our_ek.pieces_in_camp) / 10.0f);
  planes[47].SetAll(static_cast<float>(their_ek.pieces_in_camp) / 10.0f);

  return planes;
}

// --- Policy mapping (v2: direction-based) ---

int ShogiMoveToNNIndex(Move move) {
  if (move.is_drop()) {
    int pt = move.drop_piece().idx - 1;  // PieceType idx 1-7 → 0-6
    int to = move.to().as_idx();
    return (kNumDirections + kNumDirections + pt) * 81 + to;  // directions 20-26
  }

  int from = move.from().as_idx();
  int to = move.to().as_idx();
  int dir = g_direction[from * 81 + to];
  if (dir < 0) return -1;

  if (move.is_promotion()) {
    dir += kNumDirections;  // 0-9 → 10-19
  }

  return dir * 81 + to;
}

}  // namespace lczero
