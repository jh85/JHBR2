// Shallow (depth-bounded) mate search for use at MCTS leaf nodes.
//
// This is a port of dlshogi's mateMoveInOddPly / mateMoveInEvenPly /
// mateMoveIn3Ply templates from DeepLearningShogi/usi/mate.h, adapted
// to jhbr2's ShogiBoard API.
//
// vs. df-pn at the leaf:
//   - Per-call cost: ~100 ns – 5 μs (vs df-pn at 5–20 ms)
//   - Coverage: mate-in-1, -3, -5 (vs df-pn which can find longer
//     mates given enough budget)
//   - Trade-off: doesn't prove "no mate" (only finds mate or times out
//     by depth)
//
// See:
//   - docs/architecture_improvements_research.md
//   - docs/port_5ply_mate_check_plan.md
//
// PHASE 0 status: only the MoveGivesCheck helper is implemented so far.
// Templates (mateMoveInOddPly etc.) come in Phase 1.
//
#pragma once

#include "shogi/board.h"
#include "shogi/types.h"

namespace jhbr2 {

using lczero::ShogiBoard;
using lczero::Move;
using lczero::UndoInfo;

namespace shallow_mate {

// Does playing `m` from `board` give check to the opponent?
//
// "Do/undo" implementation: apply the move, ask whether the new
// side-to-move is in check, then undo. Slower than dlshogi's
// CheckInfo-based moveGivesCheck() but simple and correct using only
// existing ShogiBoard primitives.
//
// `m` MUST be a legal move from `board`. Behavior is undefined for
// illegal moves (caller must filter via GenerateLegalMoves).
inline bool MoveGivesCheck(ShogiBoard& board, Move m) {
    UndoInfo undo = board.DoMove(m);
    // After DoMove, the opponent (who didn't make the move) is the new
    // side-to-move. board.InCheck() with no arg checks the current
    // side-to-move's king — which is exactly what we want: "does the
    // opponent have their king attacked?"
    bool gives_check = board.InCheck();
    board.UndoMove(m, undo);
    return gives_check;
}

}  // namespace shallow_mate
}  // namespace jhbr2
