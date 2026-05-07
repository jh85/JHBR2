// Property test for ShogiBoard::GenerateCheckingMoves.
//
// The oracle is GenerateLegalMoves filtered by MoveGivesCheck (do/undo).
// For every position, the two methods must produce equal sets of moves.
//
// Test corpora:
//   1. Hand-curated edge cases
//   2. Real positions from the mate puzzle corpora (mate3/5/7)
//   3. Positions reached by playing N random moves from a starting position
//
// If the property holds across thousands of positions, GenerateCheckingMoves
// is correct and can replace the filter version in shallow_mate.h.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/types.h"
#include "mate/shallow_mate.h"

using lczero::ShogiBoard;
using lczero::Move;
using lczero::MoveList;
using jhbr2::shallow_mate::MoveGivesCheck;

namespace {

int n_pass = 0;
int n_fail = 0;

void check(const std::string& name, bool ok, const std::string& detail = "") {
    if (ok) {
        ++n_pass;
    } else {
        ++n_fail;
        std::printf("  FAIL  %s   %s\n", name.c_str(), detail.c_str());
    }
}

std::set<std::string> ToStringSet(const MoveList& moves) {
    std::set<std::string> out;
    for (size_t i = 0; i < moves.size(); ++i) {
        out.insert(moves[i].ToString());
    }
    return out;
}

// Property: GenerateCheckingMoves(b) == filter(GenerateLegalMoves, MoveGivesCheck)
// Returns true on success.
bool CheckPropertyOnSfen(const std::string& sfen,
                        const std::string& label) {
    ShogiBoard b;
    if (!b.SetFromSfen(sfen)) {
        std::printf("  FAIL  setup [%s]: SetFromSfen failed for %s\n",
                    label.c_str(), sfen.c_str());
        return false;
    }
    auto specialized = b.GenerateCheckingMoves();
    ShogiBoard b2;
    b2.SetFromSfen(sfen);
    auto legal = b2.GenerateLegalMoves();
    MoveList oracle;
    for (size_t i = 0; i < legal.size(); ++i) {
        if (MoveGivesCheck(b2, legal[i])) oracle.push_back(legal[i]);
    }
    auto s_set = ToStringSet(specialized);
    auto o_set = ToStringSet(oracle);
    if (s_set == o_set) return true;

    // Mismatch: print details for debugging.
    std::printf("  FAIL  [%s] mismatch on %s\n", label.c_str(), sfen.c_str());
    std::printf("        specialized (%zu): ", s_set.size());
    for (const auto& s : s_set) std::printf("%s ", s.c_str());
    std::printf("\n        oracle      (%zu): ", o_set.size());
    for (const auto& s : o_set) std::printf("%s ", s.c_str());
    std::printf("\n");
    return false;
}

}  // namespace


int main() {
    lczero::ShogiTables::Init();

    std::printf("=== GenerateCheckingMoves property test ===\n\n");

    // Tier 1: hand-curated positions
    std::printf("--- Tier 1: hand-curated ---\n");
    struct {
        const char* sfen;
        const char* label;
    } curated[] = {
        // Starting position — no checking moves
        {"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
         "starting position"},
        // Note: a "rook on 5g, white K on 5a" position is technically
        // illegal (opponent king in check at start of own-side turn),
        // so MoveGivesCheck and GenerateCheckingMoves diverge there
        // (the former returns true for unrelated king moves that
        // "leave" check intact). We don't test that case — it can't
        // arise in real play.
        // Promotion-conditional check (silver vs gold)
        {"9/9/4Sk3/9/9/9/9/9/4K4 b - 1",
         "silver moving to 5b"},
        // Uchifuzume position
        {"3pkp3/3p1p3/9/9/1B7/9/9/9/4K4 b P 1",
         "uchifuzume (P*5b illegal)"},
        // G*5b smother mate
        {"3pkp3/3p1p3/9/9/9/B8/9/9/4K4 b G 1",
         "G*5b smother mate (drop check)"},
        // R*5b smother
        {"3pkp3/3p1p3/9/9/9/B8/9/9/4K4 b R 1",
         "R*5b drop check"},
        // In-check (black king under attack)
        {"4k4/9/9/9/4r4/9/9/9/4K4 b - 1",
         "in-check evasions"},
        // Discovered check setup: black knight at 4f, black rook
        // at 5h (pinning), white king at 5b.
        // If knight moves, rook attacks king — discovered check.
        {"9/4k4/9/9/9/4N4/9/4R4/4K4 b - 1",
         "potential discovered check (knight blocks rook)"},
    };
    for (const auto& c : curated) {
        bool ok = CheckPropertyOnSfen(c.sfen, c.label);
        check(c.label, ok);
    }

    // Tier 2: real puzzle corpora
    std::printf("\n--- Tier 2: real puzzle corpora ---\n");
    auto run_corpus = [&](const std::string& path, int sample_size) {
        std::ifstream f(path);
        if (!f) {
            std::printf("  SKIP  %s (not found)\n", path.c_str());
            return;
        }
        int total = 0, ok = 0;
        std::string line;
        while (std::getline(f, line) && total < sample_size) {
            if (line.empty()) continue;
            ++total;
            if (CheckPropertyOnSfen(line, path)) ++ok;
        }
        std::string label = path + " (n=" + std::to_string(total) + ")";
        check(label, ok == total,
              std::to_string(ok) + "/" + std::to_string(total));
    };
    run_corpus("/home/ei/Downloads/JHBR2/mate3_5_7_9_11/mate3.sfen", 500);
    run_corpus("/home/ei/Downloads/JHBR2/mate3_5_7_9_11/mate5.sfen", 500);
    run_corpus("/home/ei/Downloads/JHBR2/mate3_5_7_9_11/mate7.sfen", 200);

    // Tier 3: random walks from starting position
    // For each game, play N random legal moves from start and check
    // the property at every step.
    std::printf("\n--- Tier 3: random-walk positions ---\n");
    {
        const int n_games = 50;
        const int max_plies = 80;
        int total = 0, ok = 0;
        std::vector<std::string> failed;
        for (int g = 0; g < n_games; ++g) {
            ShogiBoard b;
            b.SetStartPos();
            for (int ply = 0; ply < max_plies; ++ply) {
                ++total;
                std::string sfen = b.ToSfen();
                ShogiBoard tmp;
                tmp.SetFromSfen(sfen);
                auto specialized = tmp.GenerateCheckingMoves();
                ShogiBoard tmp2;
                tmp2.SetFromSfen(sfen);
                auto legal = tmp2.GenerateLegalMoves();
                MoveList oracle;
                for (size_t i = 0; i < legal.size(); ++i) {
                    if (MoveGivesCheck(tmp2, legal[i])) oracle.push_back(legal[i]);
                }
                auto s_set = ToStringSet(specialized);
                auto o_set = ToStringSet(oracle);
                if (s_set == o_set) {
                    ++ok;
                } else {
                    if (failed.size() < 5) failed.push_back(sfen);
                }
                // Pick a deterministic-pseudo-random legal move
                auto moves = b.GenerateLegalMoves();
                if (moves.empty()) break;
                size_t pick = (g * 11 + ply * 7) % moves.size();
                b.DoMove(moves[pick]);
            }
        }
        std::string label = "random walks (n=" + std::to_string(total) + ")";
        check(label, ok == total,
              std::to_string(ok) + "/" + std::to_string(total));
        if (!failed.empty()) {
            std::printf("  Failing positions (first 5):\n");
            for (const auto& s : failed) std::printf("    %s\n", s.c_str());
        }
    }

    std::printf("\n=== Summary: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
