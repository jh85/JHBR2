#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "mate/dfpn.h"
#include "shogi/board.h"

namespace {

using jhbr2::MateDfpnSolver;
using lczero::Move;
using lczero::ShogiBoard;

int failures = 0;

void Check(const std::string& name, bool condition) {
  if (condition) {
    std::cout << "  OK    " << name << '\n';
  } else {
    std::cout << "  FAIL  " << name << '\n';
    ++failures;
  }
}

bool IsLegal(ShogiBoard& board, Move move) {
  const auto legal_moves = board.GenerateLegalMoves();
  return std::find(legal_moves.begin(), legal_moves.end(), move) !=
         legal_moves.end();
}

void TestGameHistoryPerpetualCheck() {
  // Position after ply 155 from the game that JHBR2 lost by
  // OUTE_SENNICHITE on 2026-07-25. The following six plies retain the first
  // occurrence of the checking position in board history while placing Gote
  // at the root where the old, root-independent df-pn search chose 6e7d.
  ShogiBoard board;
  Check("perpetual-check SFEN parses",
        board.SetFromSfen(
            "5gknl/p+S1+L2g2/2G1pp1s1/1K4P2/Pn3P3/1S2+b2pp/9/7+b1/"
            "LN7 w RNrgl9p 156"));

  const std::vector<std::string> history = {
      "2h7c", "8d8e", "7c7d", "8e7f", "7d6e", "7f8e"};
  for (const auto& usi : history) {
    const Move move = Move::Parse(usi);
    Check("history move " + usi + " is legal", IsLegal(board, move));
    board.DoMove(move);
  }

  const Move losing_check = Move::Parse("6e7d");
  Check("historical losing check is legal", IsLegal(board, losing_check));
  ShogiBoard repeated = board;
  repeated.DoMove(losing_check);
  Check("historical losing check gives the defender a repetition win",
        repeated.CheckRepetition(MateDfpnSolver::kRepetitionLookbackPly) ==
            ShogiBoard::RepetitionResult::kWin);

  MateDfpnSolver solver(100000);
  const Move result = solver.search(board, 100000);
  bool safe = result.is_null() || MateDfpnSolver::IsNoMate(result);
  if (!safe) {
    ShogiBoard child = board;
    child.DoMove(result);
    const auto repetition =
        child.CheckRepetition(MateDfpnSolver::kRepetitionLookbackPly);
    safe = repetition != ShogiBoard::RepetitionResult::kWin &&
           repetition != ShogiBoard::RepetitionResult::kDraw;
  }
  Check("df-pn does not prove mate through an attacker-losing repetition",
        safe);
}

}  // namespace

int main() {
  lczero::ShogiTables::Init();
  TestGameHistoryPerpetualCheck();

  if (failures != 0) {
    std::cout << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "All df-pn repetition tests passed\n";
  return 0;
}
