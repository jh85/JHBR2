// Phase 0 test: verify MoveGivesCheck() helper.
//
// As we port the shallow mate templates in subsequent phases, this
// file will grow to cover mate-in-1/3/5 fixtures and cross-validation
// against df-pn.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/types.h"
#include "mate/shallow_mate.h"

using lczero::ShogiBoard;
using lczero::Move;
using jhbr2::shallow_mate::MoveGivesCheck;

namespace {

int passed = 0;
int failed = 0;

void check(const std::string& name, bool cond, const std::string& detail = "") {
    if (cond) {
        ++passed;
        std::printf("  OK    %s\n", name.c_str());
    } else {
        ++failed;
        std::printf("  FAIL  %s   %s\n", name.c_str(), detail.c_str());
    }
}

// Count the number of legal moves from `board` that give check.
int CountCheckingMoves(ShogiBoard& board) {
    auto moves = board.GenerateLegalMoves();
    int n = 0;
    for (size_t i = 0; i < moves.size(); ++i) {
        if (MoveGivesCheck(board, moves[i])) ++n;
    }
    return n;
}

// Cross-check: MoveGivesCheck must agree with manual do-undo for every
// legal move (this is technically tautological since MoveGivesCheck IS
// do-undo, but it confirms the API plumbing is wired correctly and
// catches accidental mutation of board state).
void test_idempotence(const std::string& name, const std::string& sfen) {
    ShogiBoard b;
    if (!b.SetFromSfen(sfen)) {
        check(name + " [setup]", false, "SetFromSfen failed");
        return;
    }
    uint64_t hash_before = b.Hash();
    auto moves = b.GenerateLegalMoves();
    bool any_failure = false;
    for (size_t i = 0; i < moves.size(); ++i) {
        Move m = moves[i];
        bool gc = MoveGivesCheck(b, m);
        // Manually do/undo and verify board is restored to same hash
        auto undo = b.DoMove(m);
        bool manual_check = b.InCheck();
        b.UndoMove(m, undo);
        if (gc != manual_check) {
            any_failure = true;
            break;
        }
        if (b.Hash() != hash_before) {
            any_failure = true;
            break;
        }
    }
    check(name, !any_failure);
}

}  // namespace


int main() {
    // Initialize the bitboard / move tables. Without this, move generation
    // silently returns 0 moves on every position.
    lczero::ShogiTables::Init();

    std::printf("=== Phase 0: MoveGivesCheck tests ===\n\n");

    // 1. Starting position: NO move gives check
    {
        ShogiBoard b;
        b.SetStartPos();
        int n = CountCheckingMoves(b);
        check("Starting position: 0 checking moves",
              n == 0,
              "got " + std::to_string(n));
    }

    // 2. Position with known checking and non-checking moves:
    //    Black rook on 5g, white king on 5a, 5-file otherwise empty.
    //    SFEN: 4k4/9/9/9/9/9/4R4/9/4K4 b - 1
    //    Spot-check specific moves rather than total count (which
    //    depends on promotion variants and king-capture conventions).
    {
        ShogiBoard b;
        bool ok = b.SetFromSfen("4k4/9/9/9/9/9/4R4/9/4K4 b - 1");
        if (!ok) {
            check("Rook-5g position: SFEN parse", false);
        } else {
            auto moves = b.GenerateLegalMoves();
            // Helper: find the move with this USI-string-like spelling.
            auto find_move = [&](const std::string& spelling) -> Move {
                for (size_t i = 0; i < moves.size(); ++i) {
                    if (moves[i].ToString() == spelling) return moves[i];
                }
                return Move();  // null move
            };
            // Forward rook moves on the 5-file should give check
            for (const std::string& s : {"5g5f", "5g5e", "5g5d"}) {
                Move m = find_move(s);
                if (m.is_null()) {
                    check("Rook-5g find " + s, false, "move not in legal list");
                } else {
                    check("Rook-5g " + s + " gives check",
                          MoveGivesCheck(b, m));
                }
            }
            // Sideways rook move on rank g should NOT give check
            for (const std::string& s : {"5g1g", "5g9g"}) {
                Move m = find_move(s);
                if (m.is_null()) {
                    check("Rook-5g find " + s, false, "move not in legal list");
                } else {
                    check("Rook-5g " + s + " does NOT give check",
                          !MoveGivesCheck(b, m));
                }
            }
            // Backward rook move 5g5h: rook still on 5-file attacking
            // 5a (black king at 5i is BEHIND the rook, doesn't block).
            // So this DOES give check.
            Move back = find_move("5g5h");
            if (!back.is_null()) {
                check("Rook-5g 5g5h still gives check (king behind rook)",
                      MoveGivesCheck(b, back));
            }
            // Total sanity: there should be SOME checking moves (>0)
            // and some non-checking (king moves, sideways).
            int n_check = 0;
            for (size_t i = 0; i < moves.size(); ++i) {
                if (MoveGivesCheck(b, moves[i])) ++n_check;
            }
            check("Rook-5g: at least 3 checking moves",
                  n_check >= 3,
                  "got " + std::to_string(n_check));
            check("Rook-5g: not all moves give check",
                  n_check < (int)moves.size(),
                  "all " + std::to_string(moves.size()) + " moves check?!");
        }
    }

    // 3. Idempotence: every call to MoveGivesCheck must leave the board
    //    in exactly the same state as before. Test on multiple positions.
    test_idempotence("Idempotence: starting position",
                     "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");

    test_idempotence("Idempotence: rook-on-5g position",
                     "4k4/9/9/9/9/9/4R4/9/4K4 b - 1");

    test_idempotence("Idempotence: middlegame",
                     "lnsgk2nl/1r2g1sb1/ppppppppp/9/9/2P6/PP1PPPPPP/1BG2S1R1/LNS1KG1NL w - 1");

    // 4. In-check position: side to move (black) is in check by white
    //    rook on 5e. Black has only evasion moves. After making any
    //    legal move, black is no longer in check (by definition of
    //    evasion). So no move can ALSO give check to white unless it
    //    happens to attack the white king.
    //
    //    Simple position: black king 5i, white rook 5e gives check.
    //    Black has the white rook covered by black pawn at 5h? No, then
    //    not in check. Let me use a simpler in-check setup:
    //    Black king 5i in check from white rook 5e, black has no other
    //    pieces. Legal moves are king moves only (4i, 6i, 4h, 5h, 6h).
    //    None of these checking moves should give check to white.
    {
        ShogiBoard b;
        bool ok = b.SetFromSfen("4k4/9/9/9/4r4/9/9/9/4K4 b - 1");
        if (!ok) {
            check("In-check position: SFEN parse", false);
        } else {
            check("In-check: side to move (black) is in check",
                  b.InCheck());
            int n_check = CountCheckingMoves(b);
            // Black king must escape rook attack. None of the king
            // escape squares give check to white king on 5a — they're
            // all far away.
            check("In-check evasions: 0 checking moves",
                  n_check == 0,
                  "got " + std::to_string(n_check));
        }
    }

    std::printf("\n=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
