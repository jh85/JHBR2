/*
  This file is part of Leela Shogi Zero (adapted from Leela Chess Zero).
  Copyright (C) 2025 The LCZero Authors
*/

// Neural network input/output encoding for Shogi.
//
// INPUT PLANES (48 channels, 9×9 each):
//   Planes  0-13:  Our 14 piece types on board (P,L,N,S,B,R,G,K,+P,+L,+N,+S,+H,+D)
//   Planes 14-27:  Their 14 piece types on board
//   Plane   28:    Repetition flag (all 1s if position has occurred before)
//   Planes 29-35:  Our hand piece counts (P,L,N,S,B,R,G), value = count
//   Planes 36-42:  Their hand piece counts
//   Plane   43:    All ones (board edge helper)
//   Plane   44:    Our entering-king points / 28.0 (nyugyoku progress)
//   Plane   45:    Their entering-king points / 28.0
//   Plane   46:    Our pieces in enemy camp / 10.0
//   Plane   47:    Their pieces in enemy camp / 10.0
//
// POLICY OUTPUT (3849 moves):
//   Indices    0-2223:  Board moves (from×to, non-promotion)
//   Indices 2224-3281:  Board moves (from×to, promotion)
//   Indices 3282-3848:  Drop moves (7 piece types × 81 squares)
//
// ATTENTION POLICY RAW OUTPUT (13689 values):
//   Section 0:  81×81 = 6561 (board from×to, non-promotion)
//   Section 1:  81×81 = 6561 (board from×to, promotion)
//   Section 2:  7×81  = 567  (drop type×to)
//   Mapped to 3849 policy indices via kShogiAttnPolicyMap.

#pragma once

#include <array>
#include <string>
#include <vector>

#include "shogi/board.h"

namespace lczero {

// --- Constants ---

constexpr int kShogiInputPlanes = 48;
constexpr int kShogiBoardSize = 9;

// --- Input Plane ---

struct ShogiInputPlane {
  // For Shogi, we use a simple 81-element float array (not a bitmask)
  // since the board is 9×9, not 8×8.
  float data[81] = {};

  void SetAll(float val = 1.0f) {
    for (int i = 0; i < 81; ++i) data[i] = val;
  }
  void Clear() { SetAll(0.0f); }

  // Set from a Bitboard (1.0 where set, 0.0 elsewhere).
  void SetFromBitboard(const Bitboard& bb) {
    Clear();
    Bitboard tmp = bb;
    while (tmp.Any()) {
      Square sq = tmp.Pop();
      data[sq.as_idx()] = 1.0f;
    }
  }
};

using ShogiInputPlanes = std::array<ShogiInputPlane, kShogiInputPlanes>;

// --- Encoding ---

// Encode a ShogiBoard as input planes for the neural network.
// The board is always encoded from the side-to-move's perspective:
// if it's WHITE's turn, the board is flipped 180° before encoding.
ShogiInputPlanes EncodeShogiPosition(const ShogiBoard& board);

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

// Convert a Move to its index in the 2187-element policy vector.
// The move must be from BLACK's perspective (flip for WHITE before calling).
// Returns -1 if the move direction is not recognized.
int ShogiMoveToNNIndex(Move move);

// --- Tables (initialized at startup) ---

namespace ShogiEncoderTables {
  void Init();
}

}  // namespace lczero
