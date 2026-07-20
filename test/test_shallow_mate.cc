// Phase 0 test: verify MoveGivesCheck() helper.
//
// As we port the shallow mate templates in subsequent phases, this
// file will grow to cover mate-in-1/3/5 fixtures and cross-validation
// against df-pn.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/types.h"
#include "mate/dfpn.h"
#include "mate/shallow_mate.h"

using lczero::ShogiBoard;
using lczero::Move;
using jhbr2::shallow_mate::MoveGivesCheck;
using jhbr2::shallow_mate::HasMateWithin;
using jhbr2::shallow_mate::MateIn3Ply;
using jhbr2::shallow_mate::MateInOddPly;

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

bool IsValidMatePv(ShogiBoard board, const std::vector<Move>& pv) {
    if (pv.empty()) return false;
    for (size_t ply = 0; ply < pv.size(); ++ply) {
        auto legal = board.GenerateLegalMoves();
        bool found = false;
        for (const Move& move : legal) {
            if (move == pv[ply]) {
                found = true;
                break;
            }
        }
        if (!found) return false;
        board.DoMove(pv[ply]);
        if (ply % 2 == 0 && !board.InCheck()) return false;
    }
    return board.InCheck() && board.GenerateLegalMoves().empty();
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

    // -------------------------------------------------------------
    // Phase 1 tests: mate-in-N detection
    // -------------------------------------------------------------

    std::printf("\n=== Phase 1: Mate-in-N tests ===\n\n");

    // Helper: assert mate-in-N detection on a given SFEN
    auto assert_mate = [&](const std::string& name, const std::string& sfen,
                           int actual_mate_depth) {
        ShogiBoard b;
        if (!b.SetFromSfen(sfen)) {
            check(name + " [setup]", false, "SetFromSfen failed");
            return;
        }
        // For each odd depth 1, 3, 5, 7:
        //   HasMateWithin should return true iff actual_mate_depth <= depth.
        for (int d : {1, 3, 5, 7}) {
            bool found = HasMateWithin(b, d);
            bool expected = actual_mate_depth <= d;
            check(name + " @ depth=" + std::to_string(d),
                  found == expected,
                  std::string("expected ") + (expected ? "true" : "false")
                  + " got " + (found ? "true" : "false"));
        }
    };

    // Mate-in-1: black rook on 5b, white king on 5a, no escape.
    // Black to move plays Rxa (or any rook move along the file that
    // checkmates).
    //
    // Wait, we need a setup where black can deliver mate-in-1.
    // Use: black rook on 5b, with white king on 5a, plus pieces
    // blocking the king's escape squares (4a, 6a).
    // SFEN: 3SkS3/4R4/9/9/9/9/9/9/4K4 b - 1
    // Black: K at 5i, R at 5b, S at 6a (silver), S at 4a (silver). White: K at 5a.
    // The black rook on 5b currently gives check. Wait, but if black
    // ALREADY gives check, white is the side to move, not black.
    //
    // Let me set up properly: position where black just gives check next.
    // Black to move, black can play Rxa or similar.
    //
    // Simplest: black rook on 5g, white king on 5a, no other pieces.
    // Black plays 5g5b: rook to 5b → check (white can move to 4a, 6a, 4b, 6b).
    // Not mate.
    //
    // For genuine mate-in-1, need king's escape squares blocked.
    // Position: white K on 5a, white pieces around (no escape).
    // Simplest: Sente plays Gold drop on 4a or similar.
    //
    // Use a classic mate-in-1 puzzle:
    //   Black has G in hand. White king on 5a is restricted by:
    //   white pawn at 4b (blocks 4b), white pawn at 6b (blocks 6b),
    //   white pawn at 5b (blocks 5b — wait, that's same file as king).
    //   Black plays G*5b (drop gold at 5b giving check, white K can't escape).
    //
    // SFEN setup:
    //   white king at 5a, white pawns at 4b, 5b? No — 5b would block our drop.
    //   Better: white king at 5a, white pieces at 4a (blocks 4a),
    //   6a (blocks 6a), 4b (blocks 4b/escape), 6b (blocks 6b).
    //   Black plays G*5b: gold attacks 5a from 5b. 4b/6b are blocked.
    //   White's only escape: 4a (blocked), 6a (blocked), 4b (blocked),
    //   6b (blocked), 5b — capture the gold? Gold on 5b is supported by what?
    //
    //   Let me make it simpler: white king on 5a fully surrounded.
    //   White pieces: 4a, 5b, 6a, 4b, 6b all blocked by white pieces.
    //   Black plays "Anastasia's mate" pattern.
    //
    //   I'll use a hand-constructed position with simple verification:
    //   "9/4P4/9/9/9/9/9/9/4K4 w - 1" then any check mates white?
    //   Wait that's white to move with white king at 5i. Not what I want.
    //
    // Let's use a known mate-in-1 from a tsume puzzle.
    // Simplest working mate-in-1:
    //   White K on 5a, with own pieces blocking escape on 4a/6a.
    //   Black has rook in hand, drops at 5b.
    //   SFEN: pkp6/9/9/9/9/9/9/9/4K4 b R 1
    //         (white pawn 9a, white K 8a, white pawn 7a — wait that puts K on 8a)
    //
    // Let me try: 4k4/9/4P4/9/9/9/9/9/4K4 b R 1 — black has rook in hand,
    // black pawn at 5c. Black plays "R*5b" — rook drops at 5b giving check.
    // White king at 5a can go to 4a or 6a (no other pieces blocking).
    // So this is NOT mate.
    //
    // Try: SFEN ngkgn4/9/9/9/9/9/9/9/4K4 b R 1
    //   white king at 7a, knight at 9a, gold at 8a, gold at 6a, knight at 5a — wait
    //   read from left to right in SFEN: file 9 first.
    //   "ngkgn4" means file9=n, file8=g, file7=k, file6=g, file5=n, then 4 empty.
    //   So white king at 7a, surrounded by knight (9a), gold (8a), gold (6a), knight (5a).
    //   Black plays R*7b: drops rook at 7b giving check. White king at 7a can:
    //     - move to 8a/6a (blocked by own gold)
    //     - move to 8b/6b (need to check) - empty squares
    //     - capture rook at 7b? Rook is supported by... nothing.
    //   So the king escapes to 8b or 6b. Not mate.
    //
    // For a clean mate-in-1, I'll use a tsume problem that's known.
    // Simple "smother mate" pattern:
    //   White king at 5a. White pieces fully surround:
    //     4a (e.g. white silver), 6a (e.g. white gold),
    //     5b (e.g. white pawn), 4b (e.g. white pawn), 6b (e.g. white pawn).
    //   Black plays N*4c or similar where the knight checks 5a.
    //   N at 4c attacks 5a (knight L-shape).
    //   White king has no escape (all 8 surrounding squares blocked or
    //   off-board), and capturing knight at 4c isn't possible from 5a.
    //   Mate.
    //
    // SFEN: 3sk1g2/3pPp3/9/9/9/9/9/9/4K4 b N 1
    //   Wait let me be careful about file numbering.
    //   Files: 9 8 7 6 5 4 3 2 1 (in that order in SFEN)
    //   Rank a: "3sk1g2" → file9-7=empty(3), file6=s, file5=k, file4=empty(1), file3=g, file2-1=empty(2). Hmm, that's wrong.
    //
    //   Let me try: white king at 5a; white silver at 4a, white gold at 6a;
    //   white pawns at 4b, 5b, 6b.
    //   SFEN rank a: file9-7 empty=3; file6=g (white gold); file5=k (white king); file4=s (white silver); file3-1 empty=3
    //   → "3gks3"
    //   Wait, gold is "g", silver is "s", king is "k". Lower-case for white.
    //
    //   Actually I realize this is getting complicated. Let me just trust
    //   our helper's correctness via simpler checks and skip mate-in-1
    //   fixtures for now. Phase 3's testing will use proper mate puzzle
    //   files.
    {
        // Mate-in-1: black has gold in hand, white king at 5a totally
        // surrounded by friendlies on rank a and rank b, except 5b (open).
        // Black plays G*5b → mate.
        //
        // Position: white K at 5a; white pawns at 4a, 6a (block king
        // escape sideways on rank a); white pawns at 4b, 6b (block
        // diagonal escape). Black drops gold at 5b — gold attacks 5a
        // from 5b directly. White king's neighbors on rank a are
        // blocked, neighbors on rank b are 4b (blocked), 5b (gold here),
        // 6b (blocked). King can't escape, can't capture gold (no
        // attacker support, but king CAN capture at 5b if gold is
        // unsupported)... this isn't actually mate then.
        //
        // To make it real mate, the gold must be supported.
        // Let's add: black bishop somewhere supporting 5b along a
        // diagonal. e.g. bishop at 1e attacks 5a... no wait attacks
        // 5b? Bishop at 8h attacks 5b? 8h to 5b is 8→5 (3 files) and h→b (6 ranks). Not equal, so no diagonal.
        // Bishop at 2b attacks 5b? 2→5 is 3 files, b→b is 0 ranks. No.
        // Bishop at 9f attacks 5b? 9→5 is 4 files, f→b is 4 ranks. YES, diagonal.
        //
        // Position:
        //   rank a: file 9-7 empty(3), 6=p(white pawn), 5=k(white king), 4=p(white pawn), 3-1 empty(3) → "3pkp3"
        //   rank b: file 9-7 empty(3), 6=p, 5=empty, 4=p, 3-1 empty(3) → "3p1p3"
        //   ...
        //   rank f: file9-7 empty, 6 empty, 5 empty, 4 empty, 3 empty, 2 empty, 1 empty? wait need bishop somewhere
        //   Let me put black bishop at 9f: rank f = "B8" (B at 9f, then 8 empty)
        //   ranks c, d, e: all empty = "9"
        //   rank g, h: empty = "9", "9"
        //   rank i: black king at 5i = "4K4"
        //
        // SFEN: 3pkp3/3p1p3/9/9/9/B8/9/9/4K4 b G 1
        //
        // Wait — I need to verify that 9f is on the diagonal to 5b.
        //   File 9 to file 5: difference 4. Rank f to rank b: f=6, b=2,
        //   difference 4. Yes! Same diagonal (positive: file decreases,
        //   rank decreases). So bishop at 9f attacks: 8e, 7d, 6c, 5b, 4a.
        //   Bishop attacks 5b. ✓
        //
        // After black plays G*5b: gold on 5b checks king at 5a.
        // King's escape squares: 4a (blocked), 6a (blocked), 4b (blocked),
        // 5b (occupied by gold — would need to capture; gold is supported
        // by black bishop on 9f, so king can't take), 6b (blocked).
        // No moves available → mate.

        assert_mate("Mate-in-1 (G*5b smother)",
                    "3pkp3/3p1p3/9/9/9/B8/9/9/4K4 b G 1",
                    /*actual_mate_depth=*/1);
    }

    // Exact mate-in-7 fixture from dlshogi's shallow-mate test set.
    assert_mate(
        "Mate-in-7 (dlshogi fixture)",
        "l3S1kpl/3r1gs2/1p2p2P1/p1p2P1+Bp/3s2Ps1/2P2p+b1P/"
        "PP2K4/7R1/LN1g4L w GNPg2n3p 5",
        /*actual_mate_depth=*/7);

    {
        ShogiBoard b;
        b.SetFromSfen(
            "l3S1kpl/3r1gs2/1p2p2P1/p1p2P1+Bp/3s2Ps1/2P2p+b1P/"
            "PP2K4/7R1/LN1g4L w GNPg2n3p 5");
        jhbr2::MateDfpnSolver solver(100000);
        const Move result = solver.search(b, 100000);
        const auto pv = solver.get_pv();
        check("DFPN mate-in-7 returns a mate", !result.is_null() &&
              !jhbr2::MateDfpnSolver::IsNoMate(result));
        check("DFPN mate-in-7 returns complete 7-ply PV", pv.size() == 7,
              "got " + std::to_string(pv.size()));
        check("DFPN mate-in-7 PV is legal checkmate", IsValidMatePv(b, pv));
    }

    // Mate-in-3 fixture: classic 3-ply puzzle.
    // I'll use a known 3-ply position.
    // Position: black has rook giving check from far, defender's only
    // escape is blocked, mate forced in 3.
    //
    // Constructing one explicitly is tedious; let's start with this:
    // mate3 from the user's existing test corpus would be ideal.
    // For now, hand-construct a simple "skewer mate":
    //   Black rook on 5e, white king on 5a, white rook on 5c (blocks).
    //   Black plays R5e×5c+ (capture+promote). Rook on 5c gives check.
    //   White must move king (only square: 4a, 6a, 4b, 6b — let's say
    //   4b is empty, 6b empty, 4a empty, 6a empty; many escapes).
    //
    // Without a curated fixture this is hard to construct manually.
    // Skip mate-in-3 fixture for Phase 1; will use real tsume puzzles
    // from mate3/ directory in Phase 3.

    // Phase 1 sanity: starting position has no mate within 5 plies
    {
        ShogiBoard b;
        b.SetStartPos();
        check("Starting position: no mate within 5",
              !HasMateWithin(b, 5));
    }

    // -------------------------------------------------------------
    // Cross-validation: real mate puzzle corpora
    // -------------------------------------------------------------
    //
    // For each puzzle file, sample the first N positions and verify
    // that HasMateWithin returns true at the expected depth and false
    // at the depth-2 below.

    auto verify_corpus = [&](const std::string& path, int mate_depth,
                             int sample_size) {
        std::ifstream f(path);
        if (!f) {
            std::printf("  SKIP  Corpus %s (file not found)\n", path.c_str());
            return;
        }
        int n_total = 0, n_correct_at_mate = 0, n_correct_below = 0;
        std::string line;
        while (std::getline(f, line) && n_total < sample_size) {
            if (line.empty()) continue;
            ShogiBoard b;
            if (!b.SetFromSfen(line)) continue;
            ++n_total;
            if (HasMateWithin(b, mate_depth)) ++n_correct_at_mate;
            if (mate_depth > 1) {
                if (!HasMateWithin(b, mate_depth - 2)) ++n_correct_below;
            }
        }
        std::string label = "Corpus mate-in-" + std::to_string(mate_depth) +
                            " (n=" + std::to_string(n_total) + ")";
        // Expect ≥95% to pass — small slack for repetition / sennichite
        // edge cases that the shallow check may handle differently.
        check(label + ": found at depth " + std::to_string(mate_depth),
              n_correct_at_mate >= n_total * 95 / 100,
              std::to_string(n_correct_at_mate) + "/" + std::to_string(n_total));
        if (mate_depth > 1) {
            check(label + ": NOT found at depth " +
                      std::to_string(mate_depth - 2),
                  n_correct_below >= n_total * 95 / 100,
                  std::to_string(n_correct_below) + "/" + std::to_string(n_total));
        }
    };

    std::printf("\n=== Cross-validation on real mate puzzles ===\n\n");
    verify_corpus("/home/ei/Downloads/JHBR2/mate3_5_7_9_11/mate3.sfen", 3, 200);
    verify_corpus("/home/ei/Downloads/JHBR2/mate3_5_7_9_11/mate5.sfen", 5, 200);
    verify_corpus("/home/ei/Downloads/JHBR2/mate3_5_7_9_11/mate7.sfen", 7, 100);

    // -------------------------------------------------------------
    // Phase 3: edge-case fixtures
    // -------------------------------------------------------------

    std::printf("\n=== Phase 3: Edge cases ===\n\n");

    // Edge 1: Uchifuzume (打ち歩詰め). A pawn drop that would mate is
    // illegal in Shogi. Set up a position where the only mating move
    // is P*5b — which is uchifuzume — and verify HasMateWithin(b, 1)
    // returns false (GenerateLegalMoves should not include the
    // illegal pawn drop).
    //
    // Position:
    //   White K on 5a, white pawns blocking all king escape squares
    //   on rank a (4a, 6a) and rank b (4b, 6b). Black pawn in hand.
    //   Black bishop on 8e supports 5b along the diagonal (so king
    //   couldn't capture the dropped pawn).
    //
    //   No other black piece has a checking move into this trapped
    //   king area.
    //
    //   If pawn drops were legal, P*5b would be mate. Since uchifuzume
    //   makes it illegal, the engine should not list P*5b in legal
    //   moves and HasMateWithin(b, 1) should return false.
    //
    // SFEN: 3pkp3/3p1p3/9/9/1B7/9/9/9/4K4 b P 1
    {
        ShogiBoard b;
        bool ok = b.SetFromSfen("3pkp3/3p1p3/9/9/1B7/9/9/9/4K4 b P 1");
        if (!ok) {
            check("Uchifuzume setup", false, "SetFromSfen failed");
        } else {
            // First confirm that the engine correctly omits P*5b from
            // legal moves (otherwise our test premise is wrong — would
            // be an engine bug, not a shallow_mate bug).
            auto moves = b.GenerateLegalMoves();
            bool has_p5b = false;
            for (size_t i = 0; i < moves.size(); ++i) {
                if (moves[i].ToString() == "P*5b") { has_p5b = true; break; }
            }
            check("Uchifuzume: P*5b NOT in legal moves",
                  !has_p5b,
                  "engine listed P*5b as legal");

            // Shallow mate-in-1 must NOT find mate (no other move mates
            // here either).
            check("Uchifuzume: HasMateWithin(b, 1) is false",
                  !HasMateWithin(b, 1));
            check("Uchifuzume: HasMateWithin(b, 3) is false",
                  !HasMateWithin(b, 3));
        }
    }

    // Edge 2: Promotion-conditional check.
    //
    // A silver moving to 5b checks a king on 4c (silver attacks the
    // back-diagonal). Promoting to gold at 5b does NOT check (gold
    // doesn't attack back-diagonals). So 5c5b and 5c5b+ behave
    // differently for MoveGivesCheck.
    //
    // SFEN: 9/9/4Sk3/9/9/9/9/9/4K4 b - 1
    //   (rank c reads file9..file1: 4 empty, S at file 5, k at file 4,
    //    3 empty)
    {
        ShogiBoard b;
        bool ok = b.SetFromSfen("9/9/4Sk3/9/9/9/9/9/4K4 b - 1");
        if (!ok) {
            check("Promotion-conditional setup", false, "SetFromSfen failed");
        } else {
            auto moves = b.GenerateLegalMoves();
            Move m_no_promo, m_promo;
            for (size_t i = 0; i < moves.size(); ++i) {
                std::string s = moves[i].ToString();
                if (s == "5c5b")  m_no_promo = moves[i];
                if (s == "5c5b+") m_promo = moves[i];
            }
            check("Prom-cond: 5c5b is in legal moves",
                  !m_no_promo.is_null());
            check("Prom-cond: 5c5b+ is in legal moves",
                  !m_promo.is_null());
            if (!m_no_promo.is_null()) {
                check("Prom-cond: 5c5b (silver) gives check",
                      MoveGivesCheck(b, m_no_promo));
            }
            if (!m_promo.is_null()) {
                check("Prom-cond: 5c5b+ (gold) does NOT give check",
                      !MoveGivesCheck(b, m_promo));
            }
        }
    }

    // Edge 3: In-check at root.
    //
    // Side to move is in check — must respond with an evasion. The
    // mate search should still terminate and behave correctly. This
    // hits the INCHECK template parameter path (handled identically
    // to non-INCHECK in our impl, but we want to verify nothing
    // misbehaves).
    //
    // Position: black king at 5i, white rook at 5e, black has nothing
    // else. Black is in check, has only king-evasion moves.
    {
        ShogiBoard b;
        bool ok = b.SetFromSfen("4k4/9/9/9/4r4/9/9/9/4K4 b - 1");
        if (!ok) {
            check("In-check root setup", false, "SetFromSfen failed");
        } else {
            check("In-check root: side to move is in check", b.InCheck());
            // No mate possible from this position — black just escapes
            // with a king move. HasMateWithin should return false, not
            // crash or hang.
            check("In-check root: HasMateWithin(b, 1) terminates and returns false",
                  !HasMateWithin(b, 1));
            check("In-check root: HasMateWithin(b, 3) terminates and returns false",
                  !HasMateWithin(b, 3));
            check("In-check root: HasMateWithin(b, 5) terminates and returns false",
                  !HasMateWithin(b, 5));
        }
    }

    // Black starts in check by the white rook on 5h. Capturing it with
    // 5g5h both evades the check and checkmates White. This exercises the
    // DFPN OR-node countercheck path.
    {
        ShogiBoard b;
        bool ok = b.SetFromSfen(
            "3pkp3/3p1p3/9/9/9/9/4R4/4r4/4K4 b - 1");
        if (!ok) {
            check("DFPN countercheck setup", false, "SetFromSfen failed");
        } else {
            jhbr2::MateDfpnSolver solver(10000);
            const Move result = solver.search(b, 10000);
            const auto pv = solver.get_pv();
            check("DFPN finds mating countercheck", result.ToString() == "5g5h");
            check("DFPN countercheck PV is legal checkmate",
                  pv.size() == 1 && IsValidMatePv(b, pv));
        }
    }

    // Edge 4: Drop check by rook.
    //
    // Black has rook in hand, white king at 5a is hemmed in such that
    // R*5b is mate-in-1 (rook on 5b checks along file, king has no
    // legal escape and rook is supported).
    //
    // Reuse a position that puts the king in a smother:
    //   white K at 5a, white silver at 4a, white silver at 6a (block
    //   sideways), white pawn at 4b (block escape), white pawn at 6b
    //   (block escape).
    //   Black supports 5b via bishop at 9f (9-5=4, f-b=4 ✓).
    //   Black has rook in hand → R*5b mate.
    //
    // SFEN: 3sks3/3p1p3/9/9/9/B8/9/9/4K4 b R 1
    //   (similar structure to G*5b smother, but white silvers replace
    //    pawns on rank a so that king can't capture silver from 4a or
    //    6a; bishop at 9f supports 5b; black rook in hand)
    //
    // Wait — silvers attack diagonally, including 5b from 4a or 6a.
    // That means a rook drop at 5b would be captured by silver. Use
    // pawns instead (which only attack forward).
    //
    // SFEN: 3pkp3/3p1p3/9/9/9/B8/9/9/4K4 b R 1
    {
        ShogiBoard b;
        bool ok = b.SetFromSfen("3pkp3/3p1p3/9/9/9/B8/9/9/4K4 b R 1");
        if (!ok) {
            check("Rook-drop-mate setup", false, "SetFromSfen failed");
        } else {
            check("Rook drop: HasMateWithin(b, 1) true (R*5b mate)",
                  HasMateWithin(b, 1));
            check("Rook drop: HasMateWithin(b, 3) true",
                  HasMateWithin(b, 3));
            check("Rook drop: HasMateWithin(b, 5) true",
                  HasMateWithin(b, 5));
        }
    }

    // Edge 5: Cross-validate against df-pn at high budget on a small
    // sample. For a corpus position where df-pn (high budget) finds
    // mate, our shallow check at the same depth should also find it
    // (>=99% agreement allowed for edge cases).
    //
    // For each puzzle in mate3.sfen sample, run BOTH:
    //   - HasMateWithin(b, 3) — shallow
    //   - df-pn with 10,000 nodes — high-confidence reference
    // and assert they agree on the verdict.
    {
        std::ifstream f("/home/ei/Downloads/JHBR2/mate3_5_7_9_11/mate3.sfen");
        if (!f) {
            std::printf("  SKIP  Cross-check (corpus not found)\n");
        } else {
            int n_total = 0, n_agree = 0;
            std::string line;
            while (std::getline(f, line) && n_total < 100) {
                if (line.empty()) continue;
                ShogiBoard b;
                if (!b.SetFromSfen(line)) continue;
                ++n_total;
                bool shallow_says = HasMateWithin(b, 3);
                ShogiBoard b2;
                b2.SetFromSfen(line);
                jhbr2::MateDfpnSolver solver(10000);
                Move r = solver.search(b2, 10000);
                bool dfpn_says = !r.is_null() &&
                                 !jhbr2::MateDfpnSolver::IsNoMate(r);
                if (shallow_says == dfpn_says) ++n_agree;
            }
            std::string label =
                "Cross-check (n=" + std::to_string(n_total) + "): "
                "shallow agrees with df-pn n=10000";
            // Allow 1% disagreement for corner cases (uchifuzume edge
            // cases at the depth boundary, etc.)
            check(label,
                  n_agree >= n_total * 99 / 100,
                  std::to_string(n_agree) + "/" + std::to_string(n_total));
        }
    }

    std::printf("\n=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
