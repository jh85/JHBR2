// Shallow (depth-bounded) mate search for use at MCTS leaf nodes.
//
// This is a port of dlshogi's mateMoveInOddPly / mateMoveInEvenPly /
// mateMoveIn3Ply templates from DeepLearningShogi/usi/mate.h, adapted
// to jhbr2's ShogiBoard API.
//
// Cost vs. df-pn at the leaf:
//   - Per-call cost: ~100 ns – 5 μs (vs df-pn at 5–20 ms)
//   - Coverage: mate-in-1, -3, -5 (vs df-pn which can find longer
//     mates given enough budget)
//   - Trade-off: doesn't prove "no mate" (only finds mate or times out
//     by depth)
//
// Implementation notes vs. dlshogi:
//   - dlshogi has specialized generators (`generateMoves<CheckAll>()`,
//     `generateMoves<Evasion>()`); jhbr2 doesn't, so we use
//     `GenerateLegalMoves()` + filter for OR nodes. AND nodes (which
//     are always reached when in check) get evasions automatically
//     because all legal moves under check are evasions.
//   - dlshogi has a 6-state RepetitionResult; jhbr2 has 4. The
//     superior/inferior material variants don't apply here.
//   - dlshogi's `mateMoveIn1Ply()` optimization is replaced by an
//     inline mate-in-1 detector inside our MateIn3Ply.
//   - We omit the `gamePly() + N > draw_ply` cap (dlshogi optimization
//     for very long games) — not exposed cleanly on ShogiBoard.
//
// See:
//   - docs/architecture_improvements_research.md
//   - docs/port_5ply_mate_check_plan.md
//
#pragma once

#include "shogi/board.h"
#include "shogi/types.h"

namespace jhbr2 {

using lczero::ShogiBoard;
using lczero::Move;
using lczero::MoveList;
using lczero::UndoInfo;

namespace shallow_mate {

// =============================================================
// Helper: does playing m give check to the opponent?
// =============================================================

inline bool MoveGivesCheck(ShogiBoard& board, Move m) {
    UndoInfo undo = board.DoMove(m);
    bool gives_check = board.InCheck();
    board.UndoMove(m, undo);
    return gives_check;
}

// =============================================================
// Repetition handling
// =============================================================
//
// jhbr2's `CheckRepetition()` returns from the side-to-move's
// perspective at the position being checked:
//   kNone : no repetition
//   kDraw : 4-fold repetition without perpetual check → draw
//   kWin  : opponent was giving perpetual check → side-to-move wins
//   kLoss : we were giving perpetual check → side-to-move loses
//
// At an OR node (attacker to move), after attacker's doMove, the
// side-to-move flips to defender, and CheckRepetition returns from
// the defender's perspective:
//   kLoss → defender loses → mate found, attacker returns true
//   kWin / kDraw → defender survives → attacker continues with next move
//
// At an AND node (defender to move, in check), after defender's
// doMove, the side-to-move flips to attacker, and CheckRepetition
// returns from the attacker's perspective:
//   kWin → attacker wins → defender's evasion failed, try next evasion
//   kLoss / kDraw → defender escaped → no mate, return false
//

// =============================================================
// Forward declarations
// =============================================================

template <int depth>             bool MateInEvenPly(ShogiBoard& board);
template <int depth, bool INCHECK = false> bool MateInOddPly(ShogiBoard& board);

// =============================================================
// Hand-tuned 3-ply mate (base case for MateInOddPly<3>).
// =============================================================
//
// Mirrors the structure of dlshogi's mateMoveIn3Ply: iterate checking
// moves (depth 1), iterate evasions (depth 2), inline mate-in-1
// detection (depth 3).
//
// INCHECK template param selects the safe in-check path at OR nodes.
// The common false path uses GenerateCheckingMovesNonCheck() to avoid
// the runtime in-check check and the legal-move fallback.
template <bool INCHECK = false>
inline bool MateIn3Ply(ShogiBoard& board) {
    // OR node (depth 1): try each checking move.
    // Use specialized GenerateCheckingMoves (Phase 6) — direct
    // bitboard-based enumeration of moves that cause check, avoiding
    // full legal-move generation on the common non-check path.
    auto checking_moves = INCHECK ? board.GenerateCheckingMoves()
                                  : board.GenerateCheckingMovesNonCheck();
    for (size_t i = 0; i < checking_moves.size(); ++i) {
        Move m1 = checking_moves[i];

        UndoInfo undo1 = board.DoMove(m1);

        // Repetition check from defender's perspective:
        // kWin/kDraw → defender survives → skip this attacker move.
        auto rep1 = board.CheckRepetition();
        if (rep1 == ShogiBoard::RepetitionResult::kWin ||
            rep1 == ShogiBoard::RepetitionResult::kDraw) {
            board.UndoMove(m1, undo1);
            continue;
        }
        // kLoss → defender loses → mate found.
        if (rep1 == ShogiBoard::RepetitionResult::kLoss) {
            board.UndoMove(m1, undo1);
            return true;
        }

        // AND node (depth 2): defender tries to escape.
        auto evasions = board.GenerateLegalMoves();
        if (evasions.empty()) {
            // Mate-in-1 — defender has no legal evasion.
            board.UndoMove(m1, undo1);
            return true;
        }

        // For each evasion, check if attacker has mate-in-1 after.
        bool all_evasions_lose = true;
        for (size_t j = 0; j < evasions.size(); ++j) {
            Move m2 = evasions[j];

            // dlshogi simplification: if defender's evasion is itself
            // a counter-check, treat as defender escape (we don't
            // try to find mate-in-1 against an in-check attacker
            // here — would require INCHECK-aware mate-in-1).
            if (MoveGivesCheck(board, m2)) {
                all_evasions_lose = false;
                break;
            }

            UndoInfo undo2 = board.DoMove(m2);

            auto rep2 = board.CheckRepetition();
            // After defender's evasion, side-to-move = attacker.
            // kWin → attacker wins → this evasion fails, try next.
            // kLoss/kDraw → attacker doesn't mate → defender escaped.
            if (rep2 == ShogiBoard::RepetitionResult::kLoss ||
                rep2 == ShogiBoard::RepetitionResult::kDraw) {
                board.UndoMove(m2, undo2);
                all_evasions_lose = false;
                break;
            }
            if (rep2 == ShogiBoard::RepetitionResult::kWin) {
                board.UndoMove(m2, undo2);
                continue;  // this evasion fails, try next
            }

            // OR node (depth 3): attacker plays mate-in-1.
            // Specialized check-only generator (Phase 6).
            auto attacker_moves = board.GenerateCheckingMovesNonCheck();
            bool found_mate1 = false;
            for (size_t k = 0; k < attacker_moves.size(); ++k) {
                Move m3 = attacker_moves[k];
                UndoInfo undo3 = board.DoMove(m3);
                if (board.GenerateLegalMoves().empty()) {
                    found_mate1 = true;
                }
                board.UndoMove(m3, undo3);
                if (found_mate1) break;
            }

            board.UndoMove(m2, undo2);
            if (!found_mate1) {
                all_evasions_lose = false;
                break;
            }
        }

        board.UndoMove(m1, undo1);
        if (all_evasions_lose) return true;
    }
    return false;
}

// =============================================================
// Generic OR-node template: MateInOddPly<depth>
// =============================================================
//
// Side-to-move (attacker) is searching for a forced mate within
// `depth` plies (must be odd). Returns true if at least one
// checking move leads to a position where MateInEvenPly<depth-1>
// returns true.
template <int depth, bool INCHECK>
inline bool MateInOddPly(ShogiBoard& board) {
    static_assert(depth >= 1 && (depth % 2) == 1,
                  "MateInOddPly: depth must be positive odd");

    // Specialized check-only generator (Phase 6).
    auto moves = INCHECK ? board.GenerateCheckingMoves()
                         : board.GenerateCheckingMovesNonCheck();
    for (size_t i = 0; i < moves.size(); ++i) {
        Move m = moves[i];

        UndoInfo undo = board.DoMove(m);

        auto rep = board.CheckRepetition();
        if (rep == ShogiBoard::RepetitionResult::kLoss) {
            // Defender loses by repetition → mate found.
            board.UndoMove(m, undo);
            return true;
        }
        if (rep == ShogiBoard::RepetitionResult::kWin ||
            rep == ShogiBoard::RepetitionResult::kDraw) {
            // Defender survives by repetition → skip this move.
            board.UndoMove(m, undo);
            continue;
        }

        // Recurse into AND node at depth-1.
        if (MateInEvenPly<depth - 1>(board)) {
            board.UndoMove(m, undo);
            return true;
        }

        board.UndoMove(m, undo);
    }
    return false;
}

// =============================================================
// Generic AND-node template: MateInEvenPly<depth>
// =============================================================
//
// Side-to-move (defender) is in check (entered here from MateInOddPly
// by attacker's checking move). We're checking if EVERY evasion leads
// to a position where MateInOddPly<depth-1> returns true.
template <int depth>
inline bool MateInEvenPly(ShogiBoard& board) {
    static_assert(depth >= 0 && (depth % 2) == 0,
                  "MateInEvenPly: depth must be non-negative even");

    auto moves = board.GenerateLegalMoves();
    if (moves.empty()) {
        // No legal moves — defender is mated.
        return true;
    }
    if (depth == 0) {
        // Out of depth and defender has at least one move → no mate.
        return false;
    }

    for (size_t i = 0; i < moves.size(); ++i) {
        Move m = moves[i];
        UndoInfo undo = board.DoMove(m);

        auto rep = board.CheckRepetition();
        // After defender's doMove, side-to-move = attacker.
        if (rep == ShogiBoard::RepetitionResult::kWin) {
            // Attacker wins by repetition → this evasion fails defender.
            board.UndoMove(m, undo);
            continue;
        }
        if (rep == ShogiBoard::RepetitionResult::kLoss ||
            rep == ShogiBoard::RepetitionResult::kDraw) {
            // Attacker loses or draws → defender escaped.
            board.UndoMove(m, undo);
            return false;
        }

        // Recurse into OR node at depth-1.
        bool attacker_in_check = board.InCheck();
        bool sub;
        if (attacker_in_check) {
            sub = MateInOddPly<depth - 1, true>(board);
        } else {
            sub = MateInOddPly<depth - 1, false>(board);
        }
        board.UndoMove(m, undo);

        if (!sub) {
            // Defender's evasion led to a non-mate position → escaped.
            return false;
        }
    }
    return true;
}

// =============================================================
// Template specializations
// =============================================================

// MateInOddPly<3> uses the hand-tuned MateIn3Ply
template <>
inline bool MateInOddPly<3, false>(ShogiBoard& board) {
    return MateIn3Ply<false>(board);
}
template <>
inline bool MateInOddPly<3, true>(ShogiBoard& board) {
    return MateIn3Ply<true>(board);
}

// MateInOddPly<1> — direct mate-in-1 detection
template <>
inline bool MateInOddPly<1, false>(ShogiBoard& board) {
    auto moves = board.GenerateCheckingMovesNonCheck();
    for (size_t i = 0; i < moves.size(); ++i) {
        Move m = moves[i];
        UndoInfo undo = board.DoMove(m);
        bool no_escape = board.GenerateLegalMoves().empty();
        board.UndoMove(m, undo);
        if (no_escape) return true;
    }
    return false;
}
template <>
inline bool MateInOddPly<1, true>(ShogiBoard& board) {
    auto moves = board.GenerateCheckingMoves();
    for (size_t i = 0; i < moves.size(); ++i) {
        Move m = moves[i];
        UndoInfo undo = board.DoMove(m);
        bool no_escape = board.GenerateLegalMoves().empty();
        board.UndoMove(m, undo);
        if (no_escape) return true;
    }
    return false;
}

// =============================================================
// Public interface
// =============================================================

// Check for a forced mate within `depth` plies. `depth` must be odd
// and one of {1, 3, 5, 7}. Returns true if side-to-move has a forced
// mate within `depth` plies; false if no such mate is found within
// the depth limit (which doesn't prove no mate exists — could just
// be deeper than `depth`).
inline bool HasMateWithin(ShogiBoard& board, int depth) {
    bool in_check = board.InCheck();
    switch (depth) {
        case 1: return in_check ? MateInOddPly<1, true>(board)
                                : MateInOddPly<1, false>(board);
        case 3: return in_check ? MateInOddPly<3, true>(board)
                                : MateInOddPly<3, false>(board);
        case 5: return in_check ? MateInOddPly<5, true>(board)
                                : MateInOddPly<5, false>(board);
        case 7: return in_check ? MateInOddPly<7, true>(board)
                                : MateInOddPly<7, false>(board);
        default: return false;
    }
}

}  // namespace shallow_mate
}  // namespace jhbr2
