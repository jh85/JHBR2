/*
  This file is part of Leela Shogi Zero (adapted from Leela Chess Zero).
  Copyright (C) 2025 The LCZero Authors
*/

#include "shogi/bitboard.h"

namespace lczero {
namespace ShogiTables {

Bitboard SquareBB[kSquareNB];
Bitboard FileBB[kBoardSize];
Bitboard RankBB[kBoardSize];
Bitboard PromotionZoneBB[COLOR_NB];

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
}

}  // namespace ShogiTables
}  // namespace lczero
