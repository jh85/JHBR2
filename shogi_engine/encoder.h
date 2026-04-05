/*
  This file is part of Leela Shogi Zero (adapted from Leela Chess Zero).
  Copyright (C) 2025 The LCZero Authors
*/

// Neural network input/output encoding for Shogi.
//
// INPUT PLANES (44 channels, 9×9 each):
//   Planes  0-13:  Our 14 piece types on board (P,L,N,S,B,R,G,K,+P,+L,+N,+S,+H,+D)
//   Planes 14-27:  Their 14 piece types on board
//   Plane   28:    Repetition flag (all 1s if position has occurred before)
//   Planes 29-35:  Our hand piece counts (P,L,N,S,B,R,G), value = count
//   Planes 36-42:  Their hand piece counts
//   Plane   43:    All ones (board edge helper)
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

constexpr int kShogiInputPlanes = 44;
constexpr int kShogiPolicySize = 3849;
constexpr int kShogiBoardSize = 9;

// Planes per board position (pieces + repetition).
constexpr int kShogiPlanesPerPosition = 29;  // 14 ours + 14 theirs + 1 rep

// Hand piece plane indices (relative to hand section start).
constexpr int kShogiHandPieces = 7;  // P, L, N, S, B, R, G

// Raw attention output sections.
constexpr int kShogiRawBoard = 81 * 81;   // 6561
constexpr int kShogiRawPromo = 81 * 81;   // 6561
constexpr int kShogiRawDrop  = 7 * 81;    // 567
constexpr int kShogiRawTotal = kShogiRawBoard + kShogiRawPromo + kShogiRawDrop;

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

// --- Policy mapping ---

// Convert a Move to its index in the 3849-element policy vector.
// The move must be from the side-to-move's perspective (BLACK after flip).
// Returns -1 if the move is not found.
int ShogiMoveToNNIndex(Move move);

// Convert a policy index back to a Move (from BLACK's perspective).
Move ShogiMoveFromNNIndex(int idx);

// Convert a Move to its index in the raw attention output (13689 elements).
int ShogiMoveToRawIndex(Move move);

// --- Tables (initialized at startup) ---

namespace ShogiEncoderTables {
  void Init();
}

}  // namespace lczero
