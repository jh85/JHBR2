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

Bitboard DlshogiPieceAttacks(PieceType pt, Color c, Square sq,
                             const Bitboard& occ) {
  const int i = sq.as_idx();
  switch (pt.idx) {
    case kPawn.idx:
      return ShogiTables::PawnEffectBB[i][c];
    case kLance.idx:
      return ShogiTables::LanceEffect(c, sq, occ);
    case kKnight.idx:
      return ShogiTables::KnightEffectBB[i][c];
    case kSilver.idx:
      return ShogiTables::SilverEffectBB[i][c];
    case kBishop.idx:
      return ShogiTables::BishopEffect(sq, occ);
    case kRook.idx:
      return ShogiTables::RookEffect(sq, occ);
    case kGold.idx:
    case kProPawn.idx:
    case kProLance.idx:
    case kProKnight.idx:
    case kProSilver.idx:
      return ShogiTables::GoldEffectBB[i][c];
    case kKing.idx:
      return ShogiTables::KingEffectBB[i];
    case kHorse.idx:
      return ShogiTables::BishopEffect(sq, occ) | ShogiTables::HorseStepBB[i];
    case kDragon.idx:
      return ShogiTables::RookEffect(sq, occ) | ShogiTables::DragonStepBB[i];
    default:
      return Bitboard::Zero();
  }
}

void SetDlshogiPlane(float* planes, int plane, Square sq, float value = 1.0f) {
  planes[plane * 81 + sq.as_idx()] = value;
}

void SetDlshogiPlaneAll(float* planes, int plane, float value = 1.0f) {
  std::fill(planes + plane * 81, planes + (plane + 1) * 81, value);
}

int DlshogiPiecePlane(PieceType pt) {
  if (pt.idx >= kPawn.idx && pt.idx <= kDragon.idx) return pt.idx - 1;
  return -1;
}

int DlshogiDropPieceLabel(PieceType pt) {
  switch (pt.idx) {
    case kPawn.idx:
    case kLance.idx:
    case kKnight.idx:
    case kSilver.idx:
    case kBishop.idx:
    case kRook.idx:
    case kGold.idx:
      return pt.idx - 1;
    default:
      return -1;
  }
}

int DlshogiMoveDirection(int dir_x, int dir_y) {
  if (dir_y < 0 && dir_x == 0) return 0;       // UP
  if (dir_y == -2 && dir_x == -1) return 8;    // UP2_LEFT
  if (dir_y == -2 && dir_x == 1) return 9;     // UP2_RIGHT
  if (dir_y < 0 && dir_x < 0) return 1;        // UP_LEFT
  if (dir_y < 0 && dir_x > 0) return 2;        // UP_RIGHT
  if (dir_y == 0 && dir_x < 0) return 3;       // LEFT
  if (dir_y == 0 && dir_x > 0) return 4;       // RIGHT
  if (dir_y > 0 && dir_x == 0) return 5;       // DOWN
  if (dir_y > 0 && dir_x < 0) return 6;        // DOWN_LEFT
  return 7;                                    // DOWN_RIGHT
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
  ShogiInputPlanes planes{};

  // Flip board if WHITE to move (always encode from mover's perspective).
  ShogiBoard b = (board.side_to_move() == WHITE) ? board.Flipped() : board;

  Color us = BLACK;
  Color them = WHITE;

  // Planes 0-13: Our 14 piece types.
  const PieceType piece_types[] = {
    kPawn, kLance, kKnight, kSilver, kBishop, kRook, kGold, kKing,
    kProPawn, kProLance, kProKnight, kProSilver, kHorse, kDragon
  };

  for (int i = 0; i < 14; ++i) {
    planes[i].SetBitsFromBitboard(b.pieces(us, piece_types[i]));
  }

  // Planes 14-27: Their 14 piece types.
  for (int i = 0; i < 14; ++i) {
    planes[14 + i].SetBitsFromBitboard(b.pieces(them, piece_types[i]));
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

void EncodeDlshogiPosition(const ShogiBoard& board, float* input1,
                           float* input2) {
  std::fill(input1, input1 + kDlshogiInput1Planes * 81, 0.0f);
  std::fill(input2, input2 + kDlshogiInput2Planes * 81, 0.0f);

  ShogiBoard b = (board.side_to_move() == WHITE) ? board.Flipped() : board;
  const Bitboard occ = b.occupied();
  int attack_num[COLOR_NB][kSquareNB] = {};

  for (int sq_idx = 0; sq_idx < kSquareNB; ++sq_idx) {
    Square sq = Square::FromIdx(sq_idx);
    Piece pc = b.piece_on(sq);
    if (pc.IsNone()) continue;

    Color c = pc.GetColor();
    PieceType pt = pc.GetType();
    int piece_plane = DlshogiPiecePlane(pt);
    if (piece_plane < 0) continue;

    const int color_offset = static_cast<int>(c) * 31;
    SetDlshogiPlane(input1, color_offset + piece_plane, sq);

    Bitboard attacks = DlshogiPieceAttacks(pt, c, sq, occ);
    attacks.ForEach([&](Square to) {
      SetDlshogiPlane(input1, color_offset + 14 + piece_plane, to);
      int& num = attack_num[c][to.as_idx()];
      if (num < 3) {
        SetDlshogiPlane(input1, color_offset + 28 + num, to);
        ++num;
      }
    });
  }

  const PieceType hand_order[] = {
      kPawn, kLance, kKnight, kSilver, kGold, kBishop, kRook};
  const int hand_max[] = {8, 4, 4, 4, 4, 2, 2};
  for (Color c : {BLACK, WHITE}) {
    int plane = static_cast<int>(c) * 28;
    for (int i = 0; i < 7; ++i) {
      int count = std::min(b.hand(c).Count(hand_order[i]), hand_max[i]);
      for (int n = 0; n < count; ++n) {
        SetDlshogiPlaneAll(input2, plane + n);
      }
      plane += hand_max[i];
    }
  }

  if (b.InCheck()) {
    SetDlshogiPlaneAll(input2, 56);
  }
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

int DlshogiMoveToNNIndex(Move move, Color side_to_move) {
  if (side_to_move == WHITE) move.Flip();

  const int to_sq = move.to().as_idx();
  if (move.is_drop()) {
    const int hand_piece = DlshogiDropPieceLabel(move.drop_piece());
    if (hand_piece < 0) return -1;
    return (20 + hand_piece) * 81 + to_sq;
  }

  const int from_sq = move.from().as_idx();
  const int to_x = to_sq / 9;
  const int to_y = to_sq % 9;
  const int from_x = from_sq / 9;
  const int from_y = from_sq % 9;
  const int dir_x = from_x - to_x;
  const int dir_y = to_y - from_y;

  int direction = DlshogiMoveDirection(dir_x, dir_y);
  if (move.is_promotion()) direction += 10;
  return direction * 81 + to_sq;
}

}  // namespace lczero
