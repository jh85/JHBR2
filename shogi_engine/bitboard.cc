/*
  This file is part of Leela Shogi Zero (adapted from Leela Chess Zero).
  Copyright (C) 2025 The LCZero Authors
*/

#include "shogi/bitboard.h"

#include <initializer_list>
#include <utility>

namespace lczero {
namespace ShogiTables {

Bitboard SquareBB[kSquareNB];
Bitboard FileBB[kBoardSize];
Bitboard RankBB[kBoardSize];
Bitboard PromotionZoneBB[COLOR_NB];

Bitboard PawnEffectBB[kSquareNB][COLOR_NB];
Bitboard KnightEffectBB[kSquareNB][COLOR_NB];
Bitboard SilverEffectBB[kSquareNB][COLOR_NB];
Bitboard GoldEffectBB[kSquareNB][COLOR_NB];
Bitboard KingEffectBB[kSquareNB];
Bitboard HorseStepBB[kSquareNB];
Bitboard DragonStepBB[kSquareNB];
Bitboard LanceMaskBB[kSquareNB][COLOR_NB];
Bitboard QugiyRookMask[kSquareNB][2];
Bitboard256 QugiyBishopMask[kSquareNB][2];

// Helper: build a step attack bitboard from a list of (df, dr) deltas.
static Bitboard MakeStepBB(int f, int r,
                           const std::initializer_list<std::pair<int,int>>& deltas) {
  Bitboard bb = Bitboard::Zero();
  for (auto [df, dr] : deltas) {
    int nf = f + df, nr = r + dr;
    if (nf >= 0 && nf < 9 && nr >= 0 && nr < 9)
      bb.Set(Square(File::FromIdx(nf), Rank::FromIdx(nr)));
  }
  return bb;
}

static void InitStepAttacks() {
  for (int f = 0; f < 9; ++f) {
    for (int r = 0; r < 9; ++r) {
      int sq = f * 9 + r;

      // Pawn: one step forward.
      // BLACK forward = rank-1, WHITE forward = rank+1.
      PawnEffectBB[sq][BLACK] = MakeStepBB(f, r, {{0, -1}});
      PawnEffectBB[sq][WHITE] = MakeStepBB(f, r, {{0, +1}});

      // Knight: 2 forward + 1 sideways.
      KnightEffectBB[sq][BLACK] = MakeStepBB(f, r, {{-1, -2}, {+1, -2}});
      KnightEffectBB[sq][WHITE] = MakeStepBB(f, r, {{-1, +2}, {+1, +2}});

      // Silver: forward + 2 forward-diagonals + 2 backward-diagonals.
      SilverEffectBB[sq][BLACK] = MakeStepBB(f, r,
          {{0,-1}, {-1,-1}, {+1,-1}, {-1,+1}, {+1,+1}});
      SilverEffectBB[sq][WHITE] = MakeStepBB(f, r,
          {{0,+1}, {-1,+1}, {+1,+1}, {-1,-1}, {+1,-1}});

      // Gold (also used for promoted pawn/lance/knight/silver):
      // forward + 2 forward-diagonals + left + right + backward.
      GoldEffectBB[sq][BLACK] = MakeStepBB(f, r,
          {{0,-1}, {-1,-1}, {+1,-1}, {-1,0}, {+1,0}, {0,+1}});
      GoldEffectBB[sq][WHITE] = MakeStepBB(f, r,
          {{0,+1}, {-1,+1}, {+1,+1}, {-1,0}, {+1,0}, {0,-1}});

      // King: all 8 neighbors.
      KingEffectBB[sq] = MakeStepBB(f, r,
          {{-1,-1}, {-1,0}, {-1,+1}, {0,-1}, {0,+1}, {+1,-1}, {+1,0}, {+1,+1}});

      // Horse extra steps: 4 cardinal directions.
      HorseStepBB[sq] = MakeStepBB(f, r,
          {{0,-1}, {0,+1}, {-1,0}, {+1,0}});

      // Dragon extra steps: 4 diagonal directions.
      DragonStepBB[sq] = MakeStepBB(f, r,
          {{-1,-1}, {-1,+1}, {+1,-1}, {+1,+1}});

      // Lance masks: all squares on the same file in the forward direction.
      // BLACK moves toward rank 0 (lower bits), WHITE toward rank 8 (higher bits).
      {
        Bitboard black_mask = Bitboard::Zero();
        Bitboard white_mask = Bitboard::Zero();
        for (int rr = 0; rr < r; ++rr)  // ranks above (BLACK direction)
          black_mask.Set(Square(File::FromIdx(f), Rank::FromIdx(rr)));
        for (int rr = r + 1; rr < 9; ++rr)  // ranks below (WHITE direction)
          white_mask.Set(Square(File::FromIdx(f), Rank::FromIdx(rr)));
        LanceMaskBB[sq][BLACK] = black_mask;
        LanceMaskBB[sq][WHITE] = white_mask;
      }
    }
  }
}

// Build unobstructed ray in one diagonal/horizontal direction from (f,r).
static Bitboard MakeRayBB(int f, int r, int df, int dr) {
  Bitboard bb = Bitboard::Zero();
  int nf = f + df, nr = r + dr;
  while (nf >= 0 && nf < 9 && nr >= 0 && nr < 9) {
    bb.Set(Square(File::FromIdx(nf), Rank::FromIdx(nr)));
    nf += df;
    nr += dr;
  }
  return bb;
}

static void InitQugiyMasks() {
  for (int f = 0; f < 9; ++f) {
    for (int r = 0; r < 9; ++r) {
      int sq = f * 9 + r;

      // --- Rook horizontal masks ---
      // Left direction: increasing file (higher bit positions in same rank).
      Bitboard left = MakeRayBB(f, r, +1, 0);
      // Right direction: decreasing file.
      Bitboard right = MakeRayBB(f, r, -1, 0);

      Bitboard right_rev = right.byte_reverse();
      Bitboard hi, lo;
      Bitboard::Unpack(right_rev, left, hi, lo);
      QugiyRookMask[sq][0] = lo;
      QugiyRookMask[sq][1] = hi;

      // --- Bishop diagonal masks ---
      // 4 diagonals: LU (left-up), LD (left-down), RU (right-up), RD (right-down).
      // "Left" = increasing file direction, "Right" = decreasing file direction.
      // "Up" = decreasing rank (toward rank a), "Down" = increasing rank (toward rank i).
      Bitboard lu = MakeRayBB(f, r, +1, -1);  // left-up
      Bitboard ld = MakeRayBB(f, r, +1, +1);  // left-down
      Bitboard ru = MakeRayBB(f, r, -1, -1);  // right-up
      Bitboard rd = MakeRayBB(f, r, -1, +1);  // right-down

      // Byte-reverse the right diagonals.
      Bitboard ru_rev = ru.byte_reverse();
      Bitboard rd_rev = rd.byte_reverse();

      // Pack into Bitboard256 after unpack.
      // We want two Bitboard256s (lo and hi) such that after
      // Unpack(reversed_occ256, occ256, hi256, lo256), the masks align.
      //
      // After Unpack on Bitboard256:
      //   lo_out.p[0] = occ.p[0]          (left diag, lower 64-bit)
      //   lo_out.p[1] = rev_occ.p[0]      (right diag rev, lower 64-bit)
      //   lo_out.p[2] = occ.p[0]          (second copy, left diag)
      //   lo_out.p[3] = rev_occ.p[0]      (second copy, right diag rev)
      //   hi_out.p[0] = occ.p[1]          (left diag, upper 64-bit)
      //   hi_out.p[1] = rev_occ.p[1]      (right diag rev, upper 64-bit)
      //   hi_out.p[2] = occ.p[1]          (second copy)
      //   hi_out.p[3] = rev_occ.p[1]      (second copy)
      //
      // So the masks should be arranged in the same order:
      //   lo_mask.p[0] = lu.p[0],  lo_mask.p[1] = ru_rev.p[0]
      //   lo_mask.p[2] = ld.p[0],  lo_mask.p[3] = rd_rev.p[0]
      //   hi_mask.p[0] = lu.p[1],  hi_mask.p[1] = ru_rev.p[1]
      //   hi_mask.p[2] = ld.p[1],  hi_mask.p[3] = rd_rev.p[1]

      // Construct the mask Bitboard256s:
      // lo_mask = Bitboard256(Bitboard(lu.lo, ru_rev.lo), Bitboard(ld.lo, rd_rev.lo))
      // hi_mask = Bitboard256(Bitboard(lu.hi, ru_rev.hi), Bitboard(ld.hi, rd_rev.hi))
      QugiyBishopMask[sq][0] = Bitboard256(
          Bitboard::FromRaw(lu.Lo(), ru_rev.Lo()),
          Bitboard::FromRaw(ld.Lo(), rd_rev.Lo()));
      QugiyBishopMask[sq][1] = Bitboard256(
          Bitboard::FromRaw(lu.Hi(), ru_rev.Hi()),
          Bitboard::FromRaw(ld.Hi(), rd_rev.Hi()));
    }
  }
}

void Init() {
  // Square bitboards.
  for (int sq = 0; sq < kSquareNB; ++sq) {
    SquareBB[sq] = Bitboard::FromSquare(Square::FromIdx(sq));
  }

  // File bitboards (each file has 9 squares).
  for (int f = 0; f < kBoardSize; ++f) {
    FileBB[f] = Bitboard::Zero();
    for (int r = 0; r < kBoardSize; ++r) {
      FileBB[f].Set(Square(File::FromIdx(f), Rank::FromIdx(r)));
    }
  }

  // Rank bitboards (each rank has 9 squares).
  for (int r = 0; r < kBoardSize; ++r) {
    RankBB[r] = Bitboard::Zero();
    for (int f = 0; f < kBoardSize; ++f) {
      RankBB[r].Set(Square(File::FromIdx(f), Rank::FromIdx(r)));
    }
  }

  // Promotion zone: BLACK = ranks 0,1,2 (top 3 rows).
  //                 WHITE = ranks 6,7,8 (bottom 3 rows).
  PromotionZoneBB[BLACK] = RankBB[0] | RankBB[1] | RankBB[2];
  PromotionZoneBB[WHITE] = RankBB[6] | RankBB[7] | RankBB[8];

  // Step attack tables.
  InitStepAttacks();

  // Qugiy sliding attack masks.
  InitQugiyMasks();
}

}  // namespace ShogiTables
}  // namespace lczero
