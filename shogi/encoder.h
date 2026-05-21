/*
  This file is part of Leela Shogi Zero (adapted from Leela Chess Zero).
  Copyright (C) 2025 The LCZero Authors
*/

// Neural network input/output encoding for Shogi.
//
// INPUT PLANES (148 binary channels, 9×9 each):
//   Planes  0-13:  Our 14 piece types on board (P,L,N,S,B,R,G,K,+P,+L,+N,+S,+H,+D)
//   Planes 14-27:  Their 14 piece types on board
//   Plane   28:    Repetition flag (all 1s if position has occurred before)
//   Planes 29-56:  Our unary hand pieces (P8,L4,N4,S4,G4,B2,R2)
//   Planes 57-84:  Their unary hand pieces (P8,L4,N4,S4,G4,B2,R2)
//   Plane   85:    All ones (board edge helper)
//   Planes 86-116: Our dlshogi-style nyugyoku features
//   Planes 117-147: Their dlshogi-style nyugyoku features
//
// NYUGYOKU FEATURES PER SIDE (31 planes):
//   +0:       King is in opponent camp
//   +1..+10: Remaining pieces needed in opponent camp, buckets 0..9.
//             10+ remaining pieces is represented by all zero planes.
//   +11..+30: Remaining points needed, buckets 0..19.
//             20+ remaining points is represented by all zero planes.
//
// POLICY OUTPUT (dlshogi-compatible, 2187 moves):
//   Encoding: direction * 81 + destination square
//   Directions  0-9:  non-promotion board moves
//   Directions 10-19: promotion board moves
//   Directions 20-26: drops (P,L,N,S,B,R,G)

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include "shogi/board.h"

namespace lczero {

// --- Constants ---

constexpr int kShogiPieceInputPlanes = 28;
constexpr int kShogiRepetitionPlane = 28;
constexpr int kShogiHandBasePlane = 29;
constexpr int kShogiHandPlanesPerSide = 28;
constexpr int kShogiHandInputPlanes = 2 * kShogiHandPlanesPerSide;
constexpr int kShogiAllOnesPlane = kShogiHandBasePlane + kShogiHandInputPlanes;
constexpr int kShogiNyugyokuBasePlane = kShogiAllOnesPlane + 1;
constexpr int kShogiNyugyokuPlanesPerSide = 31;
constexpr int kShogiNyugyokuOppFieldBuckets = 10;
constexpr int kShogiNyugyokuScoreBuckets = 20;
constexpr int kShogiInputPlanes =
    kShogiNyugyokuBasePlane + 2 * kShogiNyugyokuPlanesPerSide;
constexpr int kShogiPackedInputBytes = (kShogiInputPlanes * 81 + 7) / 8;
constexpr int kShogiBoardSize = 9;
constexpr int kDlshogiInput1Planes = 62;
constexpr int kDlshogiInput2Planes = 57;

// --- Input Plane ---

struct ShogiInputPlane {
  // For Shogi, we use a simple 81-element float array (not a bitmask)
  // since the board is 9×9, not 8×8.
  float data[81] = {};

  void SetAll(float val = 1.0f) {
    std::fill(data, data + 81, val);
  }
  void Clear() { SetAll(0.0f); }

  // Set from a Bitboard (1.0 where set, 0.0 elsewhere).
  // Layout: data[rank * 9 + file] to match the training convention.
  void SetFromBitboard(const Bitboard& bb) {
    Clear();
    SetBitsFromBitboard(bb);
  }

  // Set 1.0 where bb has bits. Existing zero values are preserved, so callers
  // must only use this on a freshly-cleared plane.
  void SetBitsFromBitboard(const Bitboard& bb) {
    uint64_t lo = bb.Lo();
    while (lo) {
      int bit = std::countr_zero(lo);
      lo &= lo - 1;
      data[(bit % 9) * 9 + bit / 9] = 1.0f;
    }
    uint64_t hi = bb.Hi();
    while (hi) {
      int bit = std::countr_zero(hi) + kBBSplit;
      hi &= hi - 1;
      data[(bit % 9) * 9 + bit / 9] = 1.0f;
    }
  }
};

using ShogiInputPlanes = std::array<ShogiInputPlane, kShogiInputPlanes>;

// --- Encoding ---

// Encode a ShogiBoard as input planes for the neural network.
// The board is always encoded from the side-to-move's perspective:
// if it's WHITE's turn, the board is flipped 180° before encoding.
ShogiInputPlanes EncodeShogiPosition(const ShogiBoard& board);

// Encode the dlshogi model inputs:
//   input1: 2 colors * (14 piece + 14 attack + 3 attack-count) planes.
//   input2: 28 hand-count planes per color plus one side-in-check plane.
// Both buffers must have room for planes * 81 floats. They are zeroed here.
void EncodeDlshogiPosition(const ShogiBoard& board, float* input1,
                           float* input2);

// --- Policy mapping (v2: direction-based, 2187 outputs) ---
//
// Encoding: direction * 81 + to_sq
//   Directions 0-9:   non-promotion board moves
//   Directions 10-19: promotion board moves
//   Directions 20-26: drops (P, L, N, S, B, R, G)
//   Total: 27 * 81 = 2187

constexpr int kPolicySize = 2187;
constexpr int kNumDirections = 10;
constexpr int kNumDropTypes = 7;

// Convert a Move to its index in the 2187-element dlshogi policy vector.
// The move must be from BLACK's perspective (flip for WHITE before calling).
// Returns -1 if the move direction is not recognized.
int ShogiMoveToNNIndex(Move move);

// Convert a move to dlshogi's 27*81 policy label. The move is passed in the
// board's native coordinates; side_to_move controls the 180-degree flip used
// by dlshogi when WHITE is to move.
int DlshogiMoveToNNIndex(Move move, Color side_to_move);

// --- Tables (initialized at startup) ---

namespace ShogiEncoderTables {
  void Init();
}

}  // namespace lczero
