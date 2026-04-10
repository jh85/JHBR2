/*
  This file is part of Leela Shogi Zero (adapted from Leela Chess Zero).
  Copyright (C) 2025 The LCZero Authors
*/

// ShogiBoard — the core board state for Shogi.
//
// Maintains a dual representation:
//   1. board_[81] array: square → Piece for O(1) lookup
//   2. Bitboards by color and piece type for bulk operations
//
// Provides legal move generation, SFEN parsing, check detection,
// and perspective flipping for neural network encoding.

#pragma once

#include <array>
#include <string>
#include <vector>

#include "shogi/bitboard.h"
#include "shogi/types.h"

namespace lczero {

// Piece with color (stored in board_ array).
// Encoding: 0 = empty, 1-14 = BLACK pieces, 17-30 = WHITE pieces.
//   Piece = color_offset + PieceType::idx
//   color_offset: BLACK=0, WHITE=16
struct Piece {
  uint8_t val;

  static constexpr Piece None() { return Piece{0}; }
  static constexpr Piece Make(Color c, PieceType pt) {
    return Piece{static_cast<uint8_t>((c == WHITE ? 16 : 0) + pt.idx)};
  }

  constexpr bool IsNone() const { return val == 0; }
  constexpr Color GetColor() const { return Color(val >> 4); }
  constexpr PieceType GetType() const {
    return PieceType::FromIdx(val & 0x0F);
  }
  constexpr Piece Promote() const {
    return Piece{static_cast<uint8_t>(val | kPromoteBit)};
  }
  constexpr Piece Unpromote() const {
    return Piece{static_cast<uint8_t>(val & ~kPromoteBit)};
  }

  bool operator==(const Piece& o) const { return val == o.val; }
  bool operator!=(const Piece& o) const { return val != o.val; }

  // USI character (uppercase = BLACK, lowercase = WHITE).
  char ToChar() const;
};

// Undo information stored per move for undo_move().
struct UndoInfo {
  Piece captured;           // Piece captured (Piece::None() if no capture)
  Hand prev_hand;           // Hand of the side that moved, before the move
  uint64_t prev_hash;       // Position hash before the move
  int prev_continuous_check; // Continuous check counter for the moving side
};

// =====================================================================
// ShogiBoard
// =====================================================================

class ShogiBoard {
 public:
  ShogiBoard();

  // --- Setup ---

  // Set to the standard starting position (hirate).
  void SetStartPos();

  // Set from SFEN string (Shogi's FEN equivalent).
  // Format: "piece_placement side_to_move hand_pieces move_count"
  // Example: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
  bool SetFromSfen(const std::string& sfen);

  // Export as SFEN string.
  std::string ToSfen() const;

  // --- Queries ---

  Color side_to_move() const { return side_to_move_; }
  Piece piece_on(Square sq) const { return board_[sq.as_idx()]; }
  bool empty(Square sq) const { return board_[sq.as_idx()].IsNone(); }
  Hand hand(Color c) const { return hand_[c]; }
  Square king_square(Color c) const { return king_sq_[c]; }

  // Bitboard queries.
  Bitboard pieces() const { return by_color_[BLACK] | by_color_[WHITE]; }
  Bitboard pieces(Color c) const { return by_color_[c]; }
  Bitboard pieces(PieceType pt) const { return by_type_[pt.idx]; }
  Bitboard pieces(Color c, PieceType pt) const {
    return by_color_[c] & by_type_[pt.idx];
  }

  // All occupied squares.
  Bitboard occupied() const { return pieces(); }

  // --- Check detection ---

  // Is the given color's king in check?
  bool InCheck(Color c) const;
  bool InCheck() const { return InCheck(side_to_move_); }

  // Bitboard of pieces attacking a given square.
  Bitboard AttackersTo(Square sq, const Bitboard& occ) const;
  Bitboard AttackersTo(Square sq) const { return AttackersTo(sq, occupied()); }

  // --- Move generation ---

  // Generate all legal moves for the side to move.
  MoveList GenerateLegalMoves();

  // Is the given move legal in the current position?
  // (Does not check if the move is well-formed — only checks legality
  // assuming a valid from/to/flags encoding.)
  bool IsLegal(Move m);

  // --- Move application ---

  // Apply a move. Returns undo information for undo_move().
  UndoInfo DoMove(Move m);

  // Undo the last move using saved undo information.
  void UndoMove(Move m, const UndoInfo& undo);

  // --- Game result ---

  enum class GameResult {
    kUndecided,
    kCheckmate,        // Side to move is checkmated (loses)
    kDeclarationWin,   // Side to move wins by entering-king declaration
    // In Shogi there's no stalemate — if you can't move, you lose.
    // Repetition (sennichite) handling is left to the search layer.
  };

  GameResult ComputeGameResult();

  // --- Entering-king declaration (入玉宣言) ---

  // Check if the side to move can declare a win by entering king.
  // Implements CSA 27-point rule:
  //   (1) Side to move's turn (implicit)
  //   (2) King is in opponent's camp (last 3 ranks)
  //   (3) Points >= 28 (BLACK) or >= 27 (WHITE)
  //       Major pieces (R,B,+R,+B) = 5 pts, others = 1 pt
  //       Counted: hand pieces + pieces in opponent's camp
  //   (4) 10+ pieces (excluding king) in opponent's camp
  //   (5) King is not in check
  //   (6) Time remaining (not checked here — engine-level concern)
  bool CanDeclareWin() const;

  // Compute entering-king point and piece counts for a given color.
  // Used for both declaration checking and NN input features.
  struct EnteringKingInfo {
    int points;            // Total points (major=5, minor=1)
    int pieces_in_camp;    // Pieces in enemy camp (excluding king)
    bool king_in_camp;     // King is in enemy camp
  };
  EnteringKingInfo ComputeEnteringKingInfo(Color c) const;

  // --- Position hashing ---

  // Hash of the current position (board + hand + side to move).
  // Two positions are the same if and only if they have the same hash,
  // same hand pieces, and same side to move.
  uint64_t Hash() const { return hash_; }

  // --- Sennichite (repetition) detection ---

  // Repetition result.
  enum class RepetitionResult {
    kNone,       // No repetition
    kDraw,       // Normal sennichite (4th occurrence) → draw
    kWin,        // Opponent was giving perpetual check → we win
    kLoss,       // We were giving perpetual check → we lose
  };

  // Check for sennichite by examining position history.
  // Call this after DoMove() to check if the new position is a repetition.
  // The history is tracked internally.
  RepetitionResult CheckRepetition() const;

  // Get the repetition count (how many times current position has occurred).
  int RepetitionCount() const;

  // Is the current position a repetition (occurred at least once before)?
  bool IsRepetition() const { return RepetitionCount() >= 1; }

  // Clear position history (call when starting a new game or setting position).
  void ClearHistory();

  // --- Perspective flip ---

  // Return a copy of the board rotated 180° with colors swapped.
  // After flipping, BLACK's pieces become WHITE's and vice versa,
  // and all squares are mirrored. The side to move becomes the opponent.
  // Used for neural network encoding (always encode from BLACK's perspective).
  ShogiBoard Flipped() const;

  // --- Debug ---

  std::string DebugString() const;

 private:
  // Place a piece on the board (updates both array and bitboards).
  void PutPiece(Square sq, Piece pc);

  // Remove a piece from the board.
  Piece RemovePiece(Square sq);

  // Move a piece (remove from src, place at dst). Does not handle captures.
  void MovePiece(Square from, Square to);

  // Generate pseudo-legal board moves (non-drop).
  void GenerateBoardMoves(MoveList& moves) const;

  // Generate pseudo-legal drop moves.
  void GenerateDropMoves(MoveList& moves) const;

  // Attack bitboards for non-sliding pieces at a given square.
  static Bitboard StepAttacks(PieceType pt, Color c, Square sq);

  // Attack bitboard for sliding pieces (lance, bishop, rook, horse, dragon).
  Bitboard SlidingAttacks(PieceType pt, Color c, Square sq,
                          const Bitboard& occ) const;

  // Combined attacks for a piece (step + sliding as applicable).
  Bitboard PieceAttacks(PieceType pt, Color c, Square sq,
                        const Bitboard& occ) const;

  // Compute hash from scratch (called after SetFromSfen).
  void ComputeHash();

  // --- Data members ---

  std::array<Piece, kSquareNB> board_;      // Square → Piece
  std::array<Bitboard, 16> by_type_;        // Indexed by PieceType::idx
  Bitboard by_color_[COLOR_NB];             // BLACK / WHITE occupancy
  Hand hand_[COLOR_NB];                     // Pieces in hand
  Square king_sq_[COLOR_NB];                // Cached king positions
  Color side_to_move_ = BLACK;
  int ply_ = 0;
  uint64_t hash_ = 0;                       // Position hash

  // Continuous check counters (number of consecutive plies giving check).
  // Reset to 0 when a non-checking move is made.
  int continuous_check_[COLOR_NB] = {0, 0};

  // Position history for sennichite detection.
  // Stores (hash, hand[BLACK], hand[WHITE]) for each position in the game.
  struct HistoryEntry {
    uint64_t hash;
    uint32_t hand_black;
    uint32_t hand_white;
  };
  std::vector<HistoryEntry> history_;
};

// =====================================================================
// Piece inline helpers
// =====================================================================

// Standard starting SFEN (hirate).
constexpr const char* kStartingSfen =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

}  // namespace lczero
