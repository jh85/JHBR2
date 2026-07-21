#include <iostream>

#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/types.h"

int main() {
  using namespace lczero;

  ShogiTables::Init();

  constexpr char kMaxLegalMovesSfen[] =
      "R8/2K1S1SSk/4B4/9/9/9/9/9/1L1L1L3 b PLNSGBR17p3n3g 1";
  ShogiBoard board;
  if (!board.SetFromSfen(kMaxLegalMovesSfen)) {
    std::cerr << "Failed to parse maximum-legal-moves position\n";
    return 1;
  }

  const MoveList moves = board.GenerateLegalMoves();
  if (moves.size() != kMaxLegalMoves) {
    std::cerr << "Expected " << kMaxLegalMoves << " legal moves, got "
              << moves.size() << '\n';
    return 1;
  }

  std::cout << "Maximum-legal-moves position: " << moves.size()
            << " moves\n";
  return 0;
}
