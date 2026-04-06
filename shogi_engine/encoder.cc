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
// Move list & policy map (generated at startup)
// =====================================================================

namespace {

// A move in the policy list.
struct PolicyMove {
  enum Type : uint8_t { kBoard, kDrop };
  Type type;
  uint8_t from_or_pt;  // from_sq for board, piece_type idx for drop
  uint8_t to;
  bool promote;

  PolicyMove() = default;
  PolicyMove(Type t, uint8_t fp, uint8_t to_, bool p)
      : type(t), from_or_pt(fp), to(to_), promote(p) {}
};

// All 3849 valid moves.
std::vector<PolicyMove> g_move_list;

// Reverse lookup: policy_idx from a board move or drop.
// Board move: board_move_idx[from * 81 + to] for non-promote,
//             board_promo_idx[from * 81 + to] for promote.
int g_board_move_idx[81 * 81];   // non-promote → policy idx, -1 if invalid
int g_board_promo_idx[81 * 81];  // promote → policy idx, -1 if invalid
int g_drop_idx[7 * 81];          // drop → policy idx

// Raw attention map: raw_idx → policy_idx.
int g_attn_policy_map[kShogiRawTotal];

// Piece movement definitions for BLACK.
struct MoveDir { int df, dr, max_dist; };

// All possible moves for each piece type (BLACK's perspective).
const MoveDir kPawnMoves[]   = {{0,-1,1}};
const MoveDir kLanceMoves[]  = {{0,-1,8}};
const MoveDir kKnightMoves[] = {{-1,-2,1},{1,-2,1}};
const MoveDir kSilverMoves[] = {{0,-1,1},{-1,-1,1},{1,-1,1},{-1,1,1},{1,1,1}};
const MoveDir kGoldMoves[]   = {{0,-1,1},{-1,-1,1},{1,-1,1},{-1,0,1},{1,0,1},{0,1,1}};
const MoveDir kBishopMoves[] = {{-1,-1,8},{-1,1,8},{1,-1,8},{1,1,8}};
const MoveDir kRookMoves[]   = {{0,-1,8},{0,1,8},{-1,0,8},{1,0,8}};
const MoveDir kKingMoves[]   = {
  {-1,-1,1},{0,-1,1},{1,-1,1},{-1,0,1},{1,0,1},{-1,1,1},{0,1,1},{1,1,1}
};
const MoveDir kHorseMoves[]  = {  // bishop + cardinal steps
  {-1,-1,8},{-1,1,8},{1,-1,8},{1,1,8},
  {0,-1,1},{0,1,1},{-1,0,1},{1,0,1}
};
const MoveDir kDragonMoves[] = {  // rook + diagonal steps
  {0,-1,8},{0,1,8},{-1,0,8},{1,0,8},
  {-1,-1,1},{-1,1,1},{1,-1,1},{1,1,1}
};

struct PieceMoveDef {
  const MoveDir* dirs;
  int n_dirs;
};

// Indexed by PieceType values for BLACK's pieces.
// Only need piece types that appear on the board (1..14 excluding 0 and 15).
PieceMoveDef g_piece_moves[16];

void InitPieceMoves() {
  auto set = [](int pt, const MoveDir* d, int n) {
    g_piece_moves[pt] = {d, n};
  };
  set(1,  kPawnMoves,   1);   // PAWN
  set(2,  kLanceMoves,  1);   // LANCE
  set(3,  kKnightMoves, 2);   // KNIGHT
  set(4,  kSilverMoves, 5);   // SILVER
  set(5,  kBishopMoves, 4);   // BISHOP
  set(6,  kRookMoves,   4);   // ROOK
  set(7,  kGoldMoves,   6);   // GOLD
  set(8,  kKingMoves,   8);   // KING
  set(9,  kGoldMoves,   6);   // PRO_PAWN (moves like gold)
  set(10, kGoldMoves,   6);   // PRO_LANCE
  set(11, kGoldMoves,   6);   // PRO_KNIGHT
  set(12, kGoldMoves,   6);   // PRO_SILVER
  set(13, kHorseMoves,  8);   // HORSE (promoted bishop)
  set(14, kDragonMoves, 8);   // DRAGON (promoted rook)
}

bool InBounds(int f, int r) { return f >= 0 && f < 9 && r >= 0 && r < 9; }

void BuildMoveList() {
  InitPieceMoves();

  // 1. Collect all valid (from, to) board move pairs.
  std::set<std::pair<int,int>> valid_pairs;
  for (int pt = 1; pt <= 14; ++pt) {
    auto& pm = g_piece_moves[pt];
    if (!pm.dirs) continue;
    for (int f = 0; f < 9; ++f) {
      for (int r = 0; r < 9; ++r) {
        int from_sq = f * 9 + r;
        for (int d = 0; d < pm.n_dirs; ++d) {
          auto& dir = pm.dirs[d];
          for (int dist = 1; dist <= dir.max_dist; ++dist) {
            int nf = f + dir.df * dist;
            int nr = r + dir.dr * dist;
            if (!InBounds(nf, nr)) break;
            valid_pairs.insert({from_sq, nf * 9 + nr});
          }
        }
      }
    }
  }

  // 2. Determine promotion-eligible pairs.
  // Promotion zone (BLACK): ranks 0, 1, 2.
  auto canPromote = [](int from_sq, int to_sq) -> bool {
    int from_r = from_sq % 9;
    int to_r = to_sq % 9;
    return from_r <= 2 || to_r <= 2;
  };

  // 3. Build move list.
  g_move_list.clear();
  std::fill(g_board_move_idx, g_board_move_idx + 81*81, -1);
  std::fill(g_board_promo_idx, g_board_promo_idx + 81*81, -1);
  std::fill(g_drop_idx, g_drop_idx + 7*81, -1);

  // Non-promotion board moves (sorted).
  for (auto& [f, t] : valid_pairs) {
    int idx = static_cast<int>(g_move_list.size());
    g_move_list.push_back({PolicyMove::kBoard,
                           static_cast<uint8_t>(f),
                           static_cast<uint8_t>(t), false});
    g_board_move_idx[f * 81 + t] = idx;
  }

  // Promotion board moves.
  for (auto& [f, t] : valid_pairs) {
    if (!canPromote(f, t)) continue;
    int idx = static_cast<int>(g_move_list.size());
    g_move_list.push_back({PolicyMove::kBoard,
                           static_cast<uint8_t>(f),
                           static_cast<uint8_t>(t), true});
    g_board_promo_idx[f * 81 + t] = idx;
  }

  // Drop moves.
  for (int pt = 0; pt < 7; ++pt) {
    for (int sq = 0; sq < 81; ++sq) {
      int idx = static_cast<int>(g_move_list.size());
      g_move_list.push_back({PolicyMove::kDrop,
                             static_cast<uint8_t>(pt),
                             static_cast<uint8_t>(sq), false});
      g_drop_idx[pt * 81 + sq] = idx;
    }
  }

  // 4. Build attention policy map.
  std::fill(g_attn_policy_map, g_attn_policy_map + kShogiRawTotal, -1);

  // Section 0: non-promotion board moves (81×81).
  for (int f = 0; f < 81; ++f) {
    for (int t = 0; t < 81; ++t) {
      int raw = f * 81 + t;
      int pol = g_board_move_idx[f * 81 + t];
      if (pol >= 0) g_attn_policy_map[raw] = pol;
    }
  }

  // Section 1: promotion board moves (81×81).
  for (int f = 0; f < 81; ++f) {
    for (int t = 0; t < 81; ++t) {
      int raw = kShogiRawBoard + f * 81 + t;
      int pol = g_board_promo_idx[f * 81 + t];
      if (pol >= 0) g_attn_policy_map[raw] = pol;
    }
  }

  // Section 2: drop moves (7×81).
  for (int pt = 0; pt < 7; ++pt) {
    for (int sq = 0; sq < 81; ++sq) {
      int raw = kShogiRawBoard + kShogiRawPromo + pt * 81 + sq;
      g_attn_policy_map[raw] = g_drop_idx[pt * 81 + sq];
    }
  }
}

}  // anonymous namespace

// =====================================================================
// Public API
// =====================================================================

namespace ShogiEncoderTables {
void Init() {
  BuildMoveList();
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

// --- Policy mapping ---

int ShogiMoveToNNIndex(Move move) {
  if (move.is_drop()) {
    // Drop: piece type index 0-6 (P=0, L=1, N=2, S=3, B=4, R=5, G=6).
    int pt = move.drop_piece().idx - 1;  // PieceType idx 1-7 → 0-6
    int to = move.to().as_idx();
    return g_drop_idx[pt * 81 + to];
  }

  int from = move.from().as_idx();
  int to = move.to().as_idx();
  int key = from * 81 + to;

  if (move.is_promotion()) {
    return g_board_promo_idx[key];
  }
  return g_board_move_idx[key];
}

Move ShogiMoveFromNNIndex(int idx) {
  if (idx < 0 || idx >= static_cast<int>(g_move_list.size())) {
    return Move();  // null move
  }

  const PolicyMove& pm = g_move_list[idx];
  if (pm.type == PolicyMove::kDrop) {
    PieceType pt = PieceType::FromIdx(pm.from_or_pt + 1);  // 0-6 → 1-7
    return Move::Drop(pt, Square::FromIdx(pm.to));
  }

  Square from = Square::FromIdx(pm.from_or_pt);
  Square to = Square::FromIdx(pm.to);
  if (pm.promote) {
    return Move::Promotion(from, to);
  }
  return Move::Normal(from, to);
}

int ShogiMoveToRawIndex(Move move) {
  if (move.is_drop()) {
    int pt = move.drop_piece().idx - 1;
    int to = move.to().as_idx();
    return kShogiRawBoard + kShogiRawPromo + pt * 81 + to;
  }

  int from = move.from().as_idx();
  int to = move.to().as_idx();

  if (move.is_promotion()) {
    return kShogiRawBoard + from * 81 + to;
  }
  return from * 81 + to;
}

}  // namespace lczero
