/*
  JHBR2 - Legal move generation benchmark

  Reads SFEN positions from a file and repeatedly calls GenerateLegalMoves().

  Usage:
    ./bench_legal100 test/legal100.sfens [repeats]
*/

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "shogi/bitboard.h"
#include "shogi/board.h"

using namespace lczero;
using Clock = std::chrono::steady_clock;

static std::vector<ShogiBoard> LoadBoards(const char* path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    std::fprintf(stderr, "ERROR: cannot open %s\n", path);
    std::exit(1);
  }

  std::vector<ShogiBoard> boards;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    ShogiBoard b;
    if (!b.SetFromSfen(line)) {
      std::fprintf(stderr, "ERROR: invalid SFEN: %s\n", line.c_str());
      std::exit(1);
    }
    boards.push_back(b);
  }

  if (boards.empty()) {
    std::fprintf(stderr, "ERROR: no positions in %s\n", path);
    std::exit(1);
  }
  return boards;
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "Usage: %s <sfens.txt> [repeats]\n", argv[0]);
    return 1;
  }

  const int repeats = (argc == 3) ? std::atoi(argv[2]) : 10000;
  if (repeats <= 0) {
    std::fprintf(stderr, "ERROR: repeats must be positive\n");
    return 1;
  }

  ShogiTables::Init();
  std::vector<ShogiBoard> boards = LoadBoards(argv[1]);

  volatile uint64_t sink = 0;
  for (auto& b : boards) sink += b.GenerateLegalMoves().size();

  uint64_t calls = 0;
  uint64_t moves = 0;

  auto t0 = Clock::now();
  for (int r = 0; r < repeats; ++r) {
    for (auto& b : boards) {
      MoveList legal = b.GenerateLegalMoves();
      moves += legal.size();
      ++calls;
    }
  }
  auto t1 = Clock::now();

  double secs = std::chrono::duration<double>(t1 - t0).count();
  std::printf("Legal movegen:\n");
  std::printf("  positions: %zu\n", boards.size());
  std::printf("  repeats:   %d\n", repeats);
  std::printf("  calls:     %lu\n", calls);
  std::printf("  moves:     %lu\n", moves);
  std::printf("  time:      %.6f sec\n", secs);
  std::printf("  calls/sec: %.0f\n", calls / secs);
  std::printf("  moves/sec: %.0f\n", moves / secs);
  std::printf("  avg moves: %.2f\n", static_cast<double>(moves) / calls);
  std::printf("  sink:      %lu\n", static_cast<uint64_t>(sink));
  return 0;
}
