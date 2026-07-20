#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/types.h"
#include "mate/shallow_mate.h"

using lczero::Move;
using lczero::ShogiBoard;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<std::string> LoadSfens(const std::string& path) {
  std::ifstream input(path);
  std::vector<std::string> sfens;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    const size_t tab = line.find('\t');
    sfens.push_back(line.substr(0, tab));
  }
  return sfens;
}

Move MateInOneOracle(ShogiBoard& board) {
  auto checks = board.GenerateCheckingMoves();
  for (const Move& move : checks) {
    auto undo = board.DoMove(move, true);
    const bool mate = !board.HasLegalEvasion();
    board.UndoMove(move, undo);
    if (mate) return move;
  }
  return Move();
}

template <typename HasMate>
void RunBenchmark(const char* label, std::vector<ShogiBoard>* boards,
                  int repeats, HasMate has_mate) {
  int found = 0;
  const auto start = Clock::now();
  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (ShogiBoard& board : *boards) {
      if (has_mate(board)) ++found;
    }
  }
  const auto end = Clock::now();
  const double elapsed_us =
      std::chrono::duration<double, std::micro>(end - start).count();
  const double calls = static_cast<double>(boards->size()) * repeats;
  std::printf("%s found=%d avg_us=%.3f\n",
              label, found, elapsed_us / calls);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "Usage: %s <sfens-or-positions.txt> [repeats]\n",
                 argv[0]);
    return 1;
  }

  const int repeats = argc == 3 ? std::atoi(argv[2]) : 1;
  if (repeats <= 0) return 1;

  lczero::ShogiTables::Init();
  const auto sfens = LoadSfens(argv[1]);
  if (sfens.empty()) {
    std::fprintf(stderr, "No positions loaded from %s\n", argv[1]);
    return 1;
  }

  std::vector<ShogiBoard> boards;
  boards.reserve(sfens.size());
  for (const auto& sfen : sfens) {
    ShogiBoard board;
    if (board.SetFromSfen(sfen)) boards.push_back(std::move(board));
  }

  std::printf("positions=%zu repeats=%d\n", boards.size(), repeats);
  RunBenchmark("oracle", &boards, repeats, [](ShogiBoard& board) {
    return !MateInOneOracle(board).is_null();
  });
  RunBenchmark("specialized", &boards, repeats, [](ShogiBoard& board) {
    return !board.FindMateInOne().is_null();
  });
  for (const int depth : {3, 5, 7}) {
    const std::string label = "shallow-d" + std::to_string(depth);
    RunBenchmark(label.c_str(), &boards, repeats,
                 [depth](ShogiBoard& board) {
                   return jhbr2::shallow_mate::HasMateWithin(
                       board, depth);
                 });
  }
  return 0;
}
