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
Bitboard PawnCheckBB[kSquareNB][COLOR_NB];
Bitboard KnightCheckBB[kSquareNB][COLOR_NB];
Bitboard SilverCheckBB[kSquareNB][COLOR_NB];
Bitboard GoldCheckBB[kSquareNB][COLOR_NB];
Bitboard LanceCheckBB[kSquareNB][COLOR_NB];
Bitboard BishopCheckBB[kSquareNB];
Bitboard HorseCheckBB[kSquareNB];

// Move-check tables (Phase 7d, promotion-aware).
Bitboard PawnMoveCheckBB  [kSquareNB][COLOR_NB];
Bitboard KnightMoveCheckBB[kSquareNB][COLOR_NB];
Bitboard SilverMoveCheckBB[kSquareNB][COLOR_NB];
Bitboard LanceMoveCheckBB [kSquareNB][COLOR_NB];
Bitboard BishopMoveCheckBB[kSquareNB][COLOR_NB];
Bitboard GoldMoveCheckBB  [kSquareNB][COLOR_NB];
Bitboard HorseMoveCheckBB [kSquareNB];

Bitboard BetweenBB[kSquareNB][kSquareNB];
Bitboard LineBB[kSquareNB][kSquareNB];
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

static void InitBetweenAndLine() {
  // 8 directions: (df, dr)
  constexpr int dirs[8][2] = {
    {0,-1}, {0,+1}, {-1,0}, {+1,0},   // rook directions
    {-1,-1}, {-1,+1}, {+1,-1}, {+1,+1} // bishop directions
  };

  for (int sq1 = 0; sq1 < kSquareNB; ++sq1) {
    for (int sq2 = 0; sq2 < kSquareNB; ++sq2) {
      BetweenBB[sq1][sq2] = Bitboard::Zero();
      LineBB[sq1][sq2] = Bitboard::Zero();
    }
  }

  for (int f1 = 0; f1 < 9; ++f1) {
    for (int r1 = 0; r1 < 9; ++r1) {
      int sq1 = f1 * 9 + r1;
      for (auto [df, dr] : dirs) {
        // Walk from sq1 in this direction, building the full ray.
        Bitboard ray = Bitboard::Zero();
        int f = f1 + df, r = r1 + dr;
        while (f >= 0 && f < 9 && r >= 0 && r < 9) {
          int sq2 = f * 9 + r;
          // BetweenBB[sq1][sq2] = all squares between sq1 and sq2 (exclusive).
          BetweenBB[sq1][sq2] = ray;
          ray.Set(Square::FromIdx(sq2));
          f += df;
          r += dr;
        }
        // LineBB: for every pair on this ray, the full line through both.
        // Full ray from sq1 in this direction (both ways).
        Bitboard full_line = MakeRayBB(f1, r1, df, dr) |
                             MakeRayBB(f1, r1, -df, -dr);
        full_line.Set(Square::FromIdx(sq1));
        // Assign to every sq2 on the forward ray.
        f = f1 + df; r = r1 + dr;
        while (f >= 0 && f < 9 && r >= 0 && r < 9) {
          int sq2 = f * 9 + r;
          LineBB[sq1][sq2] = full_line;
          f += df;
          r += dr;
        }
      }
    }
  }
}

// Build "direct attack" check tables: squares S from which a piece of
// (pt, c) attacks ksq directly (with empty occupancy for sliders).
//
// Used for DROP classification: when dropping piece pt at dst, check
// if dst is in CheckBB[ksq][c] — if so, the drop attacks ksq.
//
// Must be called AFTER InitStepAttacks() and InitQugiyMasks().
static void InitDirectAttackTables() {
  for (int ksq_idx = 0; ksq_idx < kSquareNB; ++ksq_idx) {
    Square ksq = Square::FromIdx(ksq_idx);

    PawnCheckBB[ksq_idx][BLACK]   = Bitboard::Zero();
    PawnCheckBB[ksq_idx][WHITE]   = Bitboard::Zero();
    KnightCheckBB[ksq_idx][BLACK] = Bitboard::Zero();
    KnightCheckBB[ksq_idx][WHITE] = Bitboard::Zero();
    SilverCheckBB[ksq_idx][BLACK] = Bitboard::Zero();
    SilverCheckBB[ksq_idx][WHITE] = Bitboard::Zero();
    GoldCheckBB[ksq_idx][BLACK]   = Bitboard::Zero();
    GoldCheckBB[ksq_idx][WHITE]   = Bitboard::Zero();
    LanceCheckBB[ksq_idx][BLACK]  = Bitboard::Zero();
    LanceCheckBB[ksq_idx][WHITE]  = Bitboard::Zero();
    BishopCheckBB[ksq_idx]        = Bitboard::Zero();
    HorseCheckBB[ksq_idx]         = Bitboard::Zero();

    for (int s_idx = 0; s_idx < kSquareNB; ++s_idx) {
      Square s = Square::FromIdx(s_idx);
      if (PawnEffectBB[s_idx][BLACK].Test(ksq))   PawnCheckBB[ksq_idx][BLACK].Set(s);
      if (PawnEffectBB[s_idx][WHITE].Test(ksq))   PawnCheckBB[ksq_idx][WHITE].Set(s);
      if (KnightEffectBB[s_idx][BLACK].Test(ksq)) KnightCheckBB[ksq_idx][BLACK].Set(s);
      if (KnightEffectBB[s_idx][WHITE].Test(ksq)) KnightCheckBB[ksq_idx][WHITE].Set(s);
      if (SilverEffectBB[s_idx][BLACK].Test(ksq)) SilverCheckBB[ksq_idx][BLACK].Set(s);
      if (SilverEffectBB[s_idx][WHITE].Test(ksq)) SilverCheckBB[ksq_idx][WHITE].Set(s);
      if (GoldEffectBB[s_idx][BLACK].Test(ksq))   GoldCheckBB[ksq_idx][BLACK].Set(s);
      if (GoldEffectBB[s_idx][WHITE].Test(ksq))   GoldCheckBB[ksq_idx][WHITE].Set(s);
      if (LanceEffect(BLACK, s, Bitboard::Zero()).Test(ksq))
        LanceCheckBB[ksq_idx][BLACK].Set(s);
      if (LanceEffect(WHITE, s, Bitboard::Zero()).Test(ksq))
        LanceCheckBB[ksq_idx][WHITE].Set(s);
      if (BishopEffect(s, Bitboard::Zero()).Test(ksq))
        BishopCheckBB[ksq_idx].Set(s);
    }
    HorseCheckBB[ksq_idx] = BishopCheckBB[ksq_idx] | HorseStepBB[ksq_idx];
  }
}

// Build promotion-aware "move-check" tables (Phase 7d).
//
// MoveCheckBB[ksq][c] = squares S such that a c-color piece of the
// given type at S has at least one MOVE that gives check to ksq,
// INCLUDING promotion variants. Used for the BOARD-MOVE candidate
// filter in GenerateCheckingMoves.
//
// Algorithm mirrors YaneuraOu/cppshogi/init.cpp:initCheckTable.
//
// Must be called AFTER InitDirectAttackTables (uses Effect/Slider tables).
static void InitMoveCheckTables() {
  for (int ksq_idx = 0; ksq_idx < kSquareNB; ++ksq_idx) {
    Square ksq = Square::FromIdx(ksq_idx);
    Bitboard ksq_bit = Bitboard::FromSquare(ksq);

    for (int c_int = 0; c_int < COLOR_NB; ++c_int) {
      Color c   = Color(c_int);
      Color opp = ~c;
      // Promotion zone for c: ranks 0-2 for BLACK, 6-8 for WHITE.
      Bitboard promo_zone = PromotionZoneBB[c];
      // "Rank 4" from c's perspective: the rank just outside promo
      // zone (rank 3 for BLACK, rank 5 for WHITE).
      Bitboard rank4 = (c == BLACK) ? RankBB[3] : RankBB[5];

      // -------- Gold (no promotion — single part) --------
      // Used for gold + all promoted-stepper variants (they all move
      // like gold).
      {
        Bitboard tbl = Bitboard::Zero();
        Bitboard candidates = GoldEffectBB[ksq_idx][opp];
        while (candidates.Any()) {
          Square checkSq = candidates.Pop();
          tbl |= GoldEffectBB[checkSq.as_idx()][opp];
        }
        tbl = tbl & ~ksq_bit & ~GoldEffectBB[ksq_idx][opp];
        GoldMoveCheckBB[ksq_idx][c_int] = tbl;
      }

      // -------- Silver (Parts 1+2+3) --------
      {
        Bitboard tbl = Bitboard::Zero();

        // Part 1: silver moves (no promotion) to a square attacking ksq.
        {
          Bitboard candidates = SilverEffectBB[ksq_idx][opp];
          while (candidates.Any()) {
            Square checkSq = candidates.Pop();
            tbl |= SilverEffectBB[checkSq.as_idx()][opp];
          }
        }
        // Part 2: silver IN promo zone moves and promotes to gold-attack-ksq.
        {
          Bitboard candidates = GoldEffectBB[ksq_idx][opp];
          while (candidates.Any()) {
            Square checkSq = candidates.Pop();
            tbl |= (SilverEffectBB[checkSq.as_idx()][opp] & promo_zone);
          }
        }
        // Part 3: silver at rank 4 enters promo zone with promotion → gold attacks ksq.
        {
          Bitboard candidates = GoldEffectBB[ksq_idx][opp] & promo_zone;
          while (candidates.Any()) {
            Square checkSq = candidates.Pop();
            tbl |= (SilverEffectBB[checkSq.as_idx()][opp] & rank4);
          }
        }
        tbl = tbl & ~ksq_bit & ~SilverEffectBB[ksq_idx][opp];
        SilverMoveCheckBB[ksq_idx][c_int] = tbl;
      }

      // -------- Knight --------
      {
        Bitboard tbl = Bitboard::Zero();
        // Part 1: knight moves (no promotion) to a square attacking ksq.
        {
          Bitboard candidates = KnightEffectBB[ksq_idx][opp];
          while (candidates.Any()) {
            Square checkSq = candidates.Pop();
            tbl |= KnightEffectBB[checkSq.as_idx()][opp];
          }
        }
        // Part 2: knight moves into promo zone and promotes to gold.
        // YaneuraOu only has one part: gold-attack-ksq squares INSIDE
        // promo zone, with knight's "from" squares.
        {
          Bitboard candidates = GoldEffectBB[ksq_idx][opp] & promo_zone;
          while (candidates.Any()) {
            Square checkSq = candidates.Pop();
            tbl |= KnightEffectBB[checkSq.as_idx()][opp];
          }
        }
        // No "subtract attacks" for knight (rare currently-checking case).
        KnightMoveCheckBB[ksq_idx][c_int] = tbl;
      }

      // -------- Lance --------
      {
        // Start with the file ray (the natural lance reach to ksq).
        Bitboard tbl = LanceEffect(opp, ksq, Bitboard::Zero());
        // Add: squares from which lance moves to ksq's gold-attack-zone
        // INSIDE promo zone (lance promotes to gold there).
        Bitboard candidates = GoldEffectBB[ksq_idx][opp] & promo_zone;
        while (candidates.Any()) {
          Square checkSq = candidates.Pop();
          tbl |= LanceEffect(opp, checkSq, Bitboard::Zero());
        }
        // Subtract: ksq itself and pawn-attack-ksq squares (already-checking).
        tbl = tbl & ~ksq_bit & ~PawnEffectBB[ksq_idx][opp];
        LanceMoveCheckBB[ksq_idx][c_int] = tbl;
      }

      // -------- Pawn --------
      {
        Bitboard tbl = Bitboard::Zero();
        // Part 1: pawn moves to a square attacking ksq (non-promoting).
        {
          Bitboard candidates = PawnEffectBB[ksq_idx][opp];
          while (candidates.Any()) {
            Square checkSq = candidates.Pop();
            tbl |= PawnEffectBB[checkSq.as_idx()][opp];
          }
        }
        // Part 2: pawn moves to a square inside promo zone where (as
        // gold) it attacks ksq.
        {
          Bitboard candidates = GoldEffectBB[ksq_idx][opp] & promo_zone;
          while (candidates.Any()) {
            Square checkSq = candidates.Pop();
            tbl |= PawnEffectBB[checkSq.as_idx()][opp];
          }
        }
        tbl = tbl & ~ksq_bit;
        PawnMoveCheckBB[ksq_idx][c_int] = tbl;
      }

      // -------- Bishop --------
      {
        Bitboard tbl = Bitboard::Zero();
        // Part 1: bishop moves to a square attacking ksq (non-promoting).
        {
          Bitboard candidates = BishopEffect(ksq, Bitboard::Zero());
          while (candidates.Any()) {
            Square checkSq = candidates.Pop();
            tbl |= BishopEffect(checkSq, Bitboard::Zero());
          }
        }
        // Part 2: bishop promotes (target in promo zone) — promoted
        // bishop = horse, which adds king-step around it. So checkSq
        // is around ksq IN promo zone, and our bishop reaches it.
        {
          Bitboard candidates = KingEffectBB[ksq_idx] & promo_zone;
          while (candidates.Any()) {
            Square checkSq = candidates.Pop();
            tbl |= BishopEffect(checkSq, Bitboard::Zero());
          }
        }
        // Part 3: bishop in promo zone moves (and promotes — horse
        // step). Our bishop is in promo zone, target is around ksq.
        {
          Bitboard candidates = KingEffectBB[ksq_idx];
          while (candidates.Any()) {
            Square checkSq = candidates.Pop();
            tbl |= BishopEffect(checkSq, Bitboard::Zero()) & promo_zone;
          }
        }
        tbl = tbl & ~ksq_bit;
        BishopMoveCheckBB[ksq_idx][c_int] = tbl;
      }

      // -------- Horse (color-symmetric, no promotion) --------
      // Compute once per ksq (when c_int == 0) since it's symmetric.
      if (c_int == 0) {
        Bitboard tbl = Bitboard::Zero();
        // Horse attack pattern = bishop diagonals + 4-cardinal step.
        Bitboard candidates = BishopEffect(ksq, Bitboard::Zero()) | HorseStepBB[ksq_idx];
        while (candidates.Any()) {
          Square checkSq = candidates.Pop();
          tbl |= BishopEffect(checkSq, Bitboard::Zero()) | HorseStepBB[checkSq.as_idx()];
        }
        tbl = tbl & ~ksq_bit;
        HorseMoveCheckBB[ksq_idx] = tbl;
      }
    }
  }
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

  // Line and between tables (needed for pin detection).
  InitBetweenAndLine();

  // Qugiy sliding attack masks.
  InitQugiyMasks();

  // Direct-attack check tables (used for drop classification).
  InitDirectAttackTables();

  // Promotion-aware move-check tables (used for board-move filtering).
  InitMoveCheckTables();
}

}  // namespace ShogiTables
}  // namespace lczero
