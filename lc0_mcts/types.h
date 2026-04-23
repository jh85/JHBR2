/*
  JHBR2 Shogi Engine — lc0 MCTS Adapter Types

  Maps lc0's chess types to our shogi types so the MCTS core
  (node.h/cc, search.cc/h) can be used with minimal changes.

  lc0's MCTS uses these chess types:
    - Move          → our lczero::Move (shogi/types.h)
    - MoveList      → our lczero::MoveList (shogi/types.h)
    - GameResult    → defined here
    - Position      → wrapper around ShogiBoard snapshot
    - PositionHistory → lightweight history for repetition

  The MCTS code itself is game-agnostic — it only stores Moves in
  edges and passes Position/PositionHistory to the encoder and
  terminal detection.
*/

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "shogi/board.h"
#include "shogi/types.h"

namespace lc0_shogi {

// Re-export shogi types that lc0's MCTS will use directly.
using Move = lczero::Move;
using MoveList = lczero::MoveList;
using ShogiBoard = lczero::ShogiBoard;

// =====================================================================
// GameResult — matches lc0's enum ordering
// =====================================================================

enum class GameResult : uint8_t {
  UNDECIDED = 0,
  BLACK_WON = 1,  // Side-to-move lost (from lc0's perspective: "black" = second player)
  DRAW = 2,
  WHITE_WON = 3,  // Side-to-move won
};

// Negate result (flip perspective).
inline GameResult operator-(const GameResult& res) {
  switch (res) {
    case GameResult::WHITE_WON: return GameResult::BLACK_WON;
    case GameResult::BLACK_WON: return GameResult::WHITE_WON;
    default: return res;
  }
}

// =====================================================================
// Position — snapshot of a board state (immutable after creation)
// =====================================================================

class Position {
 public:
  Position() = default;

  // Create from a board (copies the board state).
  explicit Position(const ShogiBoard& board)
      : board_(board), ply_count_(0), repetitions_(0) {}

  // Create from parent + move.
  Position(const Position& parent, Move m)
      : board_(parent.board_), ply_count_(parent.ply_count_ + 1) {
    board_.DoMove(m);
    // Repetition detection uses the board's internal history.
    auto rep = board_.CheckRepetition();
    if (rep == ShogiBoard::RepetitionResult::kDraw ||
        rep == ShogiBoard::RepetitionResult::kWin ||
        rep == ShogiBoard::RepetitionResult::kLoss) {
      repetitions_ = 1;  // At least one prior occurrence
    }
  }

  const ShogiBoard& GetBoard() const { return board_; }
  int GetGamePly() const { return ply_count_; }
  int GetRepetitions() const { return repetitions_; }
  uint64_t Hash() const { return board_.Hash(); }

  bool IsBlackToMove() const {
    // In lc0 convention: "black" = the side that moves second.
    // In shogi: BLACK = sente (first mover), WHITE = gote.
    // lc0's "IsBlackToMove" returns true when it's the second player's turn.
    // For our purposes, we just track side_to_move.
    return board_.side_to_move() == lczero::WHITE;
  }

 private:
  ShogiBoard board_;
  int ply_count_ = 0;
  int repetitions_ = 0;
};

// =====================================================================
// PositionHistory — tracks game state for repetition detection
// =====================================================================
// Simplified version: lc0 uses this for NN history planes and repetition.
// We only need it for repetition since our encoder doesn't use history planes.

class PositionHistory {
 public:
  PositionHistory() = default;

  void Reset(const ShogiBoard& board) {
    positions_.clear();
    positions_.emplace_back(board);
  }

  void Append(Move m) {
    positions_.emplace_back(positions_.back(), m);
  }

  void Pop() {
    if (positions_.size() > 1) positions_.pop_back();
  }

  const Position& Last() const { return positions_.back(); }
  const Position& Starting() const { return positions_.front(); }
  int GetLength() const { return static_cast<int>(positions_.size()); }
  bool IsBlackToMove() const { return Last().IsBlackToMove(); }

  // Compute game result from the current position.
  GameResult ComputeGameResult() const {
    const auto& board = Last().GetBoard();

    // Check repetition first (uses board's internal history).
    auto rep = const_cast<ShogiBoard&>(board).CheckRepetition();
    if (rep == ShogiBoard::RepetitionResult::kDraw) return GameResult::DRAW;
    if (rep == ShogiBoard::RepetitionResult::kWin) return GameResult::WHITE_WON;
    if (rep == ShogiBoard::RepetitionResult::kLoss) return GameResult::BLACK_WON;

    // Check declare win.
    if (const_cast<ShogiBoard&>(board).CanDeclareWin()) return GameResult::WHITE_WON;

    // Generate moves to check for checkmate.
    MoveList moves = const_cast<ShogiBoard&>(board).GenerateLegalMoves();
    if (moves.empty()) {
      // No legal moves = checkmate. Side to move loses.
      return GameResult::BLACK_WON;
    }

    return GameResult::UNDECIDED;
  }

  const ShogiBoard& GetBoardAtHead() const { return Last().GetBoard(); }

 private:
  std::vector<Position> positions_;
};

// =====================================================================
// Eval — matches lc0's Eval struct
// =====================================================================

struct Eval {
  float wl = 0.0f;   // Win minus Loss (-1 to +1)
  float d = 0.0f;    // Draw probability
  float ml = 0.0f;   // Moves left estimate
};

}  // namespace lc0_shogi
