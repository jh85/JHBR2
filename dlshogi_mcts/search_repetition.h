#pragma once

#include "shogi/board.h"

namespace dlshogi_mcts {

// Search uses an early repetition adjudication below the root to avoid
// spending playouts on cycles.  The root, however, represents the live game
// position and may legally be a second or third occurrence; the game
// controller owns adjudication of an actual fourfold repetition there.
inline lczero::ShogiBoard::RepetitionResult GetSearchRepetitionResult(
    const lczero::ShogiBoard& board, bool at_root) {
  return at_root ? lczero::ShogiBoard::RepetitionResult::kNone
                 : board.CheckRepetition();
}

}  // namespace dlshogi_mcts
