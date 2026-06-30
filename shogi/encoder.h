/*
  This file is part of Leela Shogi Zero (adapted from Leela Chess Zero).
  Copyright (C) 2025 The LCZero Authors
*/

// Neural network input/output encoding for Shogi.
//
// INPUT PLANES (148 channels, 9×9 each) — dlshogi-style, see
// docs/nyugyoku_dlshogi_features.md for the authoritative spec.
//
//   features1 (positional bitmaps, 1 bit/square):
//     Planes  0-13:  Our 14 piece types (P,L,N,S,B,R,G,K,+P,+L,+N,+S,+H,+D)
//     Planes 14-27:  Their 14 piece types
//   features2 (uniform one-hot planes, 1 bit/plane, broadcast to 81 squares):
//     Planes 28-55:  Our hand, thermometer (P×8,L×4,N×4,S×4,G×4,B×2,R×2)
//     Planes 56-83:  Their hand, thermometer
//     Plane  84:     Check (side-to-move in check)
//     Planes 85-115: Our nyugyoku block (king-in-camp, opp-field, score)
//     Planes 116-146:Their nyugyoku block
//     Plane  147:    Repetition flag (position has occurred before)
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

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include "shogi/board.h"

namespace lczero {

// --- Constants ---

// Plane families (see docs/nyugyoku_dlshogi_features.md).
constexpr int kShogiNumF1Planes = 28;   // positional bitmaps (1 bit / square)
constexpr int kShogiNumF2Planes = 120;  // uniform one-hot (1 bit / plane)
constexpr int kShogiInputPlanes = kShogiNumF1Planes + kShogiNumF2Planes;  // 148
constexpr int kShogiBoardSize = 9;
constexpr int kShogiSquares = 81;

// dlshogi-model inputs (the external-net validation path; see encoder.cc).
constexpr int kDlshogiInput1Planes = 62;
constexpr int kDlshogiInput2Planes = 57;

// Packed (bit) buffer sizes, in bytes, per position.
constexpr int kPackedF1Bytes = (kShogiNumF1Planes * kShogiSquares + 7) / 8;  // 284
constexpr int kPackedF2Bytes = (kShogiNumF2Planes + 7) / 8;                  // 15

// features2 sub-layout (local indices within the 120-plane block).
constexpr int kF2HandPerColor = 28;  // P8+L4+N4+S4+G4+B2+R2
constexpr int kF2OurHand   = 0;
constexpr int kF2TheirHand = kF2OurHand + kF2HandPerColor;        // 28
constexpr int kF2Check     = kF2TheirHand + kF2HandPerColor;      // 56
constexpr int kF2NyugyokuPerColor = 1 + 10 + 20;                 // 31
constexpr int kF2OurNyugyoku   = kF2Check + 1;                    // 57
constexpr int kF2TheirNyugyoku = kF2OurNyugyoku + kF2NyugyokuPerColor;  // 88
constexpr int kF2Repetition    = kF2TheirNyugyoku + kF2NyugyokuPerColor;  // 119

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

// Pack a ShogiBoard into the bit-packed feature buffers (the primary,
// authoritative encoder). features1 = 1 bit/square, features2 = 1 bit/plane.
// Caller provides zero-initialized buffers of kPackedF1Bytes / kPackedF2Bytes.
// The board is encoded from the side-to-move's perspective (flipped 180° for
// WHITE). This is what the GPU unpack kernel expands; see encoder_unpack.cu.
void PackShogiPosition(const ShogiBoard& board,
                       uint8_t* packed_f1, uint8_t* packed_f2);

// Encode a ShogiBoard as full float input planes. Defined as a CPU unpack of
// PackShogiPosition(), so it is bit-identical to the GPU path. Used by the
// ONNX Runtime fallback and as the correctness oracle for the kernel.
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

// Convert a Move to its index in the 2187-element policy vector.
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
