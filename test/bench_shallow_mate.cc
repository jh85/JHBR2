// Microbenchmark: shallow mate vs df-pn on real positions.

#include <chrono>
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

using namespace lczero;
using namespace jhbr2;
using Clock = std::chrono::steady_clock;

static std::vector<std::string> LoadSfens(const std::string& path, int max_n) {
    std::vector<std::string> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line) && (int)out.size() < max_n) {
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

int main() {
    ShogiTables::Init();

    // Load both mate-in-N puzzles (where mate exists) AND non-mate
    // positions (where df-pn must exhaust budget).
    auto mate3 = LoadSfens("/home/ei/Downloads/JHBR2/mate3_5_7_9_11/mate3.sfen", 500);
    // For "non-mate" positions, use the starting position repeatedly —
    // not the most realistic but quick.

    std::printf("=== Microbenchmark: shallow mate (depth=5) vs df-pn ===\n");
    std::printf("Test set: %zu mate-in-3 puzzles\n\n", mate3.size());

    // Shallow benchmark at multiple depths
    for (int d : {3, 5}) {
        auto t0 = Clock::now();
        int found = 0;
        for (const auto& sfen : mate3) {
            ShogiBoard b;
            b.SetFromSfen(sfen);
            if (shallow_mate::HasMateWithin(b, d)) ++found;
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  shallow d=%d  : %5d/%zu mates found, "
                    "total %7.1f ms, per-call %6.1f us\n",
                    d, found, mate3.size(), ms, 1000.0 * ms / mate3.size());
    }

    // Bench on NON-mate positions (the common case at MCTS leaves):
    // use opening positions where no forced mate exists.
    std::printf("\n--- On 500 NON-mate positions (starting pos + variations) ---\n");
    std::vector<std::string> non_mate;
    {
        // Generate variations by playing 1-3 random-ish moves from start
        ShogiBoard b;
        b.SetStartPos();
        non_mate.push_back(b.ToSfen());
        for (int n = 1; n < 500; ++n) {
            ShogiBoard nb;
            nb.SetStartPos();
            // Play n%6 moves to diversify
            for (int i = 0; i < (n % 6); ++i) {
                auto moves = nb.GenerateLegalMoves();
                if (moves.empty()) break;
                nb.DoMove(moves[(n * 7 + i * 13) % moves.size()]);
            }
            non_mate.push_back(nb.ToSfen());
        }
    }

    for (int d : {3, 5}) {
        auto t0 = Clock::now();
        int found = 0;
        for (const auto& sfen : non_mate) {
            ShogiBoard b;
            b.SetFromSfen(sfen);
            if (shallow_mate::HasMateWithin(b, d)) ++found;
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  shallow d=%d  : %5d/%zu (non-mate) found, "
                    "total %7.1f ms, per-call %6.1f us\n",
                    d, found, non_mate.size(), ms,
                    1000.0 * ms / non_mate.size());
    }
    for (int budget : {10, 20, 100}) {
        auto t0 = Clock::now();
        int found = 0;
        for (const auto& sfen : non_mate) {
            ShogiBoard b;
            b.SetFromSfen(sfen);
            MateDfpnSolver solver(budget);
            Move res = solver.search(b, budget);
            if (!res.is_null() && !MateDfpnSolver::IsNoMate(res)) ++found;
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  df-pn n=%4d : %5d/%zu (non-mate) found, "
                    "total %7.1f ms, per-call %6.1f us\n",
                    budget, found, non_mate.size(), ms,
                    1000.0 * ms / non_mate.size());
    }

    // df-pn benchmarks at various budgets
    for (int budget : {10, 20, 100, 1000}) {
        auto t0 = Clock::now();
        int found = 0;
        for (const auto& sfen : mate3) {
            ShogiBoard b;
            b.SetFromSfen(sfen);
            MateDfpnSolver solver(budget);
            Move res = solver.search(b, budget);
            if (!res.is_null() && !MateDfpnSolver::IsNoMate(res)) ++found;
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  df-pn n=%4d : %5d/%zu mates found, "
                    "total %7.1f ms, per-call %6.1f us\n",
                    budget, found, mate3.size(), ms,
                    1000.0 * ms / mate3.size());
    }

    return 0;
}
