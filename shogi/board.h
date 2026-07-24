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
#include "shogi/packed_sfen.h"
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
  bool gave_check = false;  // Whether the move checked the new side to move
  bool updated_auxiliary = true; // Hash/history/check counters were updated
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

  // Import/export YaneuraOu's canonical 32-byte PackedSfen. PackedSfen does
  // not contain the game ply, so callers supply it separately when decoding.
  bool SetFromPackedSfen(const PackedSfen& packed, int game_ply = 1);
  bool ToPackedSfen(PackedSfen* packed) const;

  // --- Queries ---

  Color side_to_move() const { return side_to_move_; }
  // Plies played from the SetStartPos / SetFromSfen origin. DoMove
  // increments by 1, undo_move decrements by 1. Used for draw-by-
  // move-limit detection (e.g. floodgate's 320-ply cap).
  int ply() const { return ply_; }
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

  // --- Pin detection ---

  // Compute bitboard of pieces pinned to the king of the given color.
  // A piece is "pinned" if removing it would expose its king to a sliding attack.
  // Returns both own and enemy pinned pieces (caller filters by color if needed).
  Bitboard ComputeBlockersForKing(Color king_color) const;

  // --- Move generation ---

  // Generate all legal moves for the side to move.
  MoveList GenerateLegalMoves();

  // Generate legal evasions when the side to move is in check.
  // The caller must guarantee InCheck(). This avoids generating and
  // testing unrelated moves at shallow-mate AND nodes.
  MoveList GenerateEvasionMoves();

  // Return as soon as one legal evasion is found. The caller must
  // guarantee InCheck().
  bool HasLegalEvasion();

  // Return a legal mate-in-1 move, or a null move if none exists.
  // FindMateInOneNonCheck uses the hand-specialized bitboard path and
  // requires that the side to move is not in check.
  Move FindMateInOne();
  Move FindMateInOneNonCheck();

  // Generate the subset of legal moves that give check to the
  // opponent. Equivalent to filter(GenerateLegalMoves, MoveGivesCheck).
  //
  // Two implementations:
  //   - GenerateCheckingMovesViaFilter: simple oracle (slow).
  //     Used as a property-test reference.
  //   - GenerateCheckingMoves: production version. Uses precomputed
  //     CheckBB tables (Phase 7a) to identify candidate pieces and
  //     skip moves of pieces that cannot possibly give check.
  //
  // Used by mate/shallow_mate.h. See docs/port_5ply_mate_check_plan.md
  // (Phase 6) and docs/port_yaneuraou_check_generator_plan.md (Phase 7).
  MoveList GenerateCheckingMoves();
  MoveList GenerateCheckingMovesNonCheck();  // caller guarantees !InCheck()
  MoveList GenerateCheckingMovesViaFilter();  // oracle; slower but obviously correct

  // Is the given move legal in the current position?
  bool IsLegal(Move m, const Bitboard& pinned);

  // --- Move application ---

  // Apply a move. Returns undo information for undo_move().
  UndoInfo DoMove(Move m);

  // Apply a move whose check status is already known.
  UndoInfo DoMove(Move m, bool gives_check);

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
    kDraw,       // Search-time normal repetition → draw
    kWin,        // Opponent was giving perpetual check → we win
    kLoss,       // We were giving perpetual check → we lose
  };

  // Search-time repetition adjudication. This deliberately returns a result
  // at the first prior occurrence so the search avoids cycles; the game
  // controller remains responsible for the official fourth occurrence.
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

  // Internal move application. Move generation uses the lightweight mode
  // because temporary legality checks do not need hash/history/check counters.
  UndoInfo DoMoveInternal(Move m, bool update_auxiliary, int gives_check);
  UndoInfo DoMoveForMovegen(Move m) {
    return DoMoveInternal(m, false, -1);
  }

  class MateProbeSourceRemoval {
   public:
    MateProbeSourceRemoval(ShogiBoard& board, Color color, PieceType type,
                           Square square)
        : board_(board), color_(color), type_(type), square_(square) {
      assert(board_.by_color_[color_].Test(square_));
      assert(board_.by_type_[type_.idx].Test(square_));
      board_.by_color_[color_].Clear(square_);
      board_.by_type_[type_.idx].Clear(square_);
    }

    ~MateProbeSourceRemoval() {
      board_.by_color_[color_].Set(square_);
      board_.by_type_[type_.idx].Set(square_);
    }

    MateProbeSourceRemoval(const MateProbeSourceRemoval&) = delete;
    MateProbeSourceRemoval& operator=(const MateProbeSourceRemoval&) = delete;

   private:
    ShogiBoard& board_;
    Color color_;
    PieceType type_;
    Square square_;
  };

  class MateProbeCaptureRemoval {
   public:
    MateProbeCaptureRemoval(ShogiBoard& board, Square square)
        : board_(board), piece_(board.piece_on(square)), square_(square) {
      if (piece_.IsNone()) return;
      board_.by_color_[piece_.GetColor()].Clear(square_);
      board_.by_type_[piece_.GetType().idx].Clear(square_);
    }

    ~MateProbeCaptureRemoval() {
      if (piece_.IsNone()) return;
      board_.by_color_[piece_.GetColor()].Set(square_);
      board_.by_type_[piece_.GetType().idx].Set(square_);
    }

    MateProbeCaptureRemoval(const MateProbeCaptureRemoval&) = delete;
    MateProbeCaptureRemoval& operator=(const MateProbeCaptureRemoval&) =
        delete;

   private:
    ShogiBoard& board_;
    Piece piece_;
    Square square_;
  };

  template <Color Us>
  Move FindMateInOneNonCheckImpl();

  template <Color Us>
  bool IsMateAfterMateProbe(PieceType moved_type, Square checker_square);

  bool CanKingEscapeAfterMateProbe(
      Color attacker, Square checker_square, const Bitboard& occupied_after,
      const Bitboard& moved_checker_attacks) const;
  bool CanDefenderCaptureMateChecker(
      Color defender, Square checker_square,
      const Bitboard& occupied_after) const;
  bool CanDefenderInterposeMateCheck(
      Color defender, const Bitboard& between,
      const Bitboard& occupied_after) const;

  MoveList GenerateEvasionMovesImpl(bool stop_after_one);

  // Generate pseudo-legal board moves (non-drop).
  void GenerateBoardMoves(MoveList& moves) const;

  // Generate legal board moves when the side to move is not in check.
  void GenerateBoardMovesNonCheck(MoveList& moves, const Bitboard& pinned) const;

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

  // Fast yes/no attack query. Avoids building a full attackers bitboard.
  bool IsSquareAttacked(Square sq, const Bitboard& occ, Color attacker) const;

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
