#include <cstdio>

#include "dlshogi_mcts/search_repetition.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/types.h"

namespace {

using dlshogi_mcts::GetSearchRepetitionResult;
using lczero::Move;
using lczero::ShogiBoard;

int failures = 0;

void Check(const char* name, bool condition) {
  if (!condition) {
    std::printf("  FAIL  %s\n", name);
    ++failures;
  }
}

void TestRepeatedRootIsNotTerminal() {
  ShogiBoard board;
  board.SetStartPos();

  // Return to startpos after one reversible cycle.  The live game is only at
  // its second occurrence, so it remains a position that MCTS must search.
  board.DoMove(Move::Parse("5i5h"));
  board.DoMove(Move::Parse("5a5b"));
  board.DoMove(Move::Parse("5h5i"));
  board.DoMove(Move::Parse("5b5a"));

  Check("fixture is a repeated position",
        board.CheckRepetition() == ShogiBoard::RepetitionResult::kDraw);
  Check("repeated root is not search-terminal",
        GetSearchRepetitionResult(board, true) ==
            ShogiBoard::RepetitionResult::kNone);
}

void TestRepeatedDescendantIsTerminal() {
  ShogiBoard board;
  board.SetStartPos();
  board.DoMove(Move::Parse("5i5h"));
  board.DoMove(Move::Parse("5a5b"));
  board.DoMove(Move::Parse("5h5i"));
  board.DoMove(Move::Parse("5b5a"));

  Check("repeated descendant remains search-terminal",
        GetSearchRepetitionResult(board, false) ==
            ShogiBoard::RepetitionResult::kDraw);
}

void TestFreshDescendantIsNotTerminal() {
  ShogiBoard board;
  board.SetStartPos();
  board.DoMove(Move::Parse("7g7f"));

  Check("fresh descendant is not search-terminal",
        GetSearchRepetitionResult(board, false) ==
            ShogiBoard::RepetitionResult::kNone);
}

}  // namespace

int main() {
  lczero::ShogiTables::Init();

  TestRepeatedRootIsNotTerminal();
  TestRepeatedDescendantIsTerminal();
  TestFreshDescendantIsNotTerminal();

  std::printf("\n=== Search repetition: %d failed ===\n", failures);
  return failures == 0 ? 0 : 1;
}
