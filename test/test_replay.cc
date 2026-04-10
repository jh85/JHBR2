/*
  JHBR2 — Game replay test

  Replays a sequence of USI moves via DoMove and checks:
  1. SFEN matches expected at each step
  2. Legal moves match expected at each step

  Input format (one test per file):
    Line 1: USI move sequence (space-separated)
    Remaining lines: expected SFEN<TAB>expected_moves (one per move, from generate_positions.py)

  Usage:
    ./test_replay test/game1_replay.txt
*/

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/types.h"

using namespace lczero;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <replay_test.txt>" << std::endl;
    return 1;
  }

  ShogiTables::Init();

  std::ifstream fin(argv[1]);
  if (!fin.is_open()) {
    std::cerr << "ERROR: Cannot open " << argv[1] << std::endl;
    return 1;
  }

  // Line 1: USI moves
  std::string moves_line;
  std::getline(fin, moves_line);
  std::vector<std::string> usi_moves;
  {
    std::istringstream iss(moves_line);
    std::string tok;
    while (iss >> tok) usi_moves.push_back(tok);
  }

  // Remaining lines: expected SFEN + moves
  struct Expected {
    std::string sfen;
    std::set<std::string> moves;
  };
  std::vector<Expected> expected;
  {
    std::string line;
    while (std::getline(fin, line)) {
      if (line.empty() || line[0] == '#') continue;
      auto tab = line.find('\t');
      if (tab == std::string::npos) continue;
      Expected e;
      e.sfen = line.substr(0, tab);
      std::istringstream mss(line.substr(tab + 1));
      std::string m;
      while (mss >> m) e.moves.insert(m);
      expected.push_back(std::move(e));
    }
  }

  if (usi_moves.size() != expected.size()) {
    std::cerr << "ERROR: " << usi_moves.size() << " moves but "
              << expected.size() << " expected positions" << std::endl;
    return 1;
  }

  std::cout << "Replaying " << usi_moves.size() << " moves" << std::endl;

  ShogiBoard board;
  board.SetStartPos();

  int passed = 0;
  int failed = 0;

  for (size_t i = 0; i < usi_moves.size(); i++) {
    Move m = Move::Parse(usi_moves[i]);
    board.DoMove(m);

    // Check SFEN
    std::string got_sfen = board.ToSfen();
    if (got_sfen != expected[i].sfen) {
      std::cerr << "SFEN MISMATCH at move " << i + 1 << " (" << usi_moves[i] << ")" << std::endl;
      std::cerr << "  Expected: " << expected[i].sfen << std::endl;
      std::cerr << "  Got:      " << got_sfen << std::endl;
      failed++;
      // Continue to see if subsequent moves also fail
      continue;
    }

    // Check legal moves
    MoveList legal = board.GenerateLegalMoves();
    std::set<std::string> got_moves;
    for (const Move& lm : legal) {
      got_moves.insert(lm.ToString());
    }

    if (got_moves != expected[i].moves) {
      std::cerr << "MOVEGEN MISMATCH at move " << i + 1 << " (" << usi_moves[i] << ")" << std::endl;
      std::cerr << "  SFEN: " << got_sfen << std::endl;

      std::vector<std::string> missing, extra;
      std::set_difference(expected[i].moves.begin(), expected[i].moves.end(),
                          got_moves.begin(), got_moves.end(),
                          std::back_inserter(missing));
      std::set_difference(got_moves.begin(), got_moves.end(),
                          expected[i].moves.begin(), expected[i].moves.end(),
                          std::back_inserter(extra));

      if (!missing.empty()) {
        std::cerr << "  MISSING:";
        for (auto& s : missing) std::cerr << " " << s;
        std::cerr << std::endl;
      }
      if (!extra.empty()) {
        std::cerr << "  EXTRA:";
        for (auto& s : extra) std::cerr << " " << s;
        std::cerr << std::endl;
      }
      failed++;
      continue;
    }

    passed++;
  }

  std::cout << "\n=== RESULTS ===" << std::endl;
  std::cout << "Passed: " << passed << " / " << usi_moves.size() << std::endl;
  if (failed > 0) {
    std::cout << "FAILED: " << failed << std::endl;
    return 1;
  }
  std::cout << "ALL PASSED" << std::endl;
  return 0;
}
