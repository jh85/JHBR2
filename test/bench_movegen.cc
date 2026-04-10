/*
  JHBR2 — Move generation benchmark

  Measures:
  1. Raw movegen speed: GenerateLegalMoves() calls per second
  2. Perft: recursive move count to verify correctness and measure DoMove speed

  Usage:
    ./bench_movegen
*/

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/types.h"

using namespace lczero;
using Clock = std::chrono::steady_clock;

// Perft: count leaf nodes at given depth
static uint64_t Perft(ShogiBoard& board, int depth) {
  if (depth == 0) return 1;

  MoveList moves = board.GenerateLegalMoves();
  if (depth == 1) return moves.size();

  uint64_t nodes = 0;
  for (const Move& m : moves) {
    ShogiBoard child = board;
    child.DoMove(m);
    nodes += Perft(child, depth - 1);
  }
  return nodes;
}

// Benchmark raw movegen: generate legal moves N times from various positions
static void BenchRawMovegen(const std::vector<std::string>& sfens, int repeats) {
  std::vector<ShogiBoard> boards;
  for (auto& s : sfens) {
    ShogiBoard b;
    b.SetFromSfen(s);
    boards.push_back(b);
  }

  // Warmup
  for (auto& b : boards) {
    volatile auto moves = b.GenerateLegalMoves();
    (void)moves;
  }

  uint64_t total_calls = 0;
  uint64_t total_moves = 0;

  auto t0 = Clock::now();
  for (int r = 0; r < repeats; r++) {
    for (auto& b : boards) {
      auto moves = b.GenerateLegalMoves();
      total_moves += moves.size();
      total_calls++;
    }
  }
  auto t1 = Clock::now();

  double secs = std::chrono::duration<double>(t1 - t0).count();
  printf("Raw movegen:\n");
  printf("  %lu calls in %.3f sec\n", total_calls, secs);
  printf("  %.0f calls/sec\n", total_calls / secs);
  printf("  %.0f moves/sec\n", total_moves / secs);
  printf("  Avg %.1f legal moves/position\n",
         (double)total_moves / total_calls);
}

// Benchmark perft
static void BenchPerft(const std::string& sfen, const std::string& label,
                       int depth) {
  ShogiBoard board;
  board.SetFromSfen(sfen);

  printf("Perft %s (depth %d):\n", label.c_str(), depth);
  printf("  SFEN: %s\n", sfen.c_str());

  auto t0 = Clock::now();
  uint64_t nodes = Perft(board, depth);
  auto t1 = Clock::now();

  double secs = std::chrono::duration<double>(t1 - t0).count();
  printf("  Nodes: %lu\n", nodes);
  printf("  Time:  %.3f sec\n", secs);
  printf("  Speed: %.0f nodes/sec\n", nodes / secs);
}

int main() {
  ShogiTables::Init();

  // === Test positions ===
  const std::string startpos =
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

  // Midgame with pieces in hand
  const std::string midgame1 =
      "ln1gk2nl/1r2g2b1/p1sppsppp/2p3p2/1p7/2P1P4/PP1PSP1PP/1BG4R1/LN2KG1NL b - 1";

  // Complex position with drops
  const std::string midgame2 =
      "l3k2nl/4g2b1/p1sppsppp/2p3p2/1p7/2P1P4/PP1PSP1PP/1BG4R1/LN2KG1NL b RNPrnp 1";

  // Late game
  const std::string endgame =
      "3g1k3/5+P3/4p1+Spp/p4N3/6p2/1P1P5/P3+b1P1P/2+r6/K1S3GNL w RBG2SN4Pl2p 1";

  std::vector<std::string> all_positions = {
      startpos, midgame1, midgame2, endgame};

  // --- Raw movegen benchmark ---
  printf("============================\n");
  printf("JHBR2 Move Generation Bench\n");
  printf("============================\n\n");

  BenchRawMovegen(all_positions, 100000);

  // --- Perft benchmarks ---
  printf("\n");
  BenchPerft(startpos, "startpos", 4);
  printf("\n");
  BenchPerft(startpos, "startpos", 5);
  printf("\n");
  BenchPerft(midgame2, "midgame+drops", 4);

  return 0;
}
