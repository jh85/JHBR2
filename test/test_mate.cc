/*
  JHBR2 — Mate problem solver benchmark

  Reads SFEN mate problems and tests the df-pn solver.
  Reports: solved count, solve rate, average nodes, average time.

  Usage:
    ./test_mate <problems.sfen> [max_problems] [nodes_budget]

  Example:
    ./test_mate mate3_5_7_9_11/mate3.sfen 1000 1000
*/

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "mate/dfpn.h"

using namespace lczero;
using namespace jhbr2;
using Clock = std::chrono::steady_clock;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <problems.sfen> [max_problems] [nodes_budget]\n", argv[0]);
    return 1;
  }

  const char* filepath = argv[1];
  int max_problems = argc >= 3 ? atoi(argv[2]) : 1000;
  int nodes_budget = argc >= 4 ? atoi(argv[3]) : 10000;

  ShogiTables::Init();

  // Load problems
  std::vector<std::string> problems;
  {
    std::ifstream fin(filepath);
    if (!fin.is_open()) {
      fprintf(stderr, "Cannot open %s\n", filepath);
      return 1;
    }
    std::string line;
    while (std::getline(fin, line) && (int)problems.size() < max_problems) {
      if (!line.empty() && line[0] != '#') {
        problems.push_back(line);
      }
    }
  }

  printf("File: %s\n", filepath);
  printf("Problems: %d (budget: %d nodes each)\n", (int)problems.size(), nodes_budget);
  printf("\n");

  int solved = 0;
  int unsolved = 0;
  int nomate = 0;
  int parse_errors = 0;
  uint64_t total_nodes = 0;
  double total_time_ms = 0;
  int max_nodes_used = 0;

  auto t0_all = Clock::now();

  for (int i = 0; i < (int)problems.size(); i++) {
    ShogiBoard board;
    if (!board.SetFromSfen(problems[i])) {
      parse_errors++;
      continue;
    }

    MateDfpnSolver solver(nodes_budget);

    auto t0 = Clock::now();
    Move result = solver.search(board, nodes_budget);
    auto t1 = Clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int nodes = (int)solver.get_nodes_searched();
    total_time_ms += ms;
    total_nodes += nodes;

    if (!result.is_null() && !MateDfpnSolver::IsNoMate(result)) {
      solved++;
      if (nodes > max_nodes_used) max_nodes_used = nodes;
    } else if (MateDfpnSolver::IsNoMate(result)) {
      nomate++;
      if (i < 5) {
        fprintf(stderr, "  [%d] NOMATE: %s\n", i, problems[i].c_str());
      }
    } else {
      unsolved++;
    }

    // Progress report every 10%
    if ((i + 1) % std::max(1, (int)problems.size() / 10) == 0) {
      double elapsed = std::chrono::duration<double>(Clock::now() - t0_all).count();
      printf("  [%d/%d] solved=%d unsolved=%d nomate=%d (%.1fs)\n",
             i + 1, (int)problems.size(), solved, unsolved, nomate, elapsed);
    }
  }

  double elapsed_all = std::chrono::duration<double>(Clock::now() - t0_all).count();

  printf("\n=== RESULTS ===\n");
  printf("Problems:      %d\n", (int)problems.size());
  printf("Solved:        %d (%.1f%%)\n", solved, 100.0 * solved / problems.size());
  printf("Unsolved:      %d (%.1f%%)\n", unsolved, 100.0 * unsolved / problems.size());
  printf("No mate:       %d (%.1f%%)\n", nomate, 100.0 * nomate / problems.size());
  if (parse_errors > 0)
    printf("Parse errors:  %d\n", parse_errors);
  printf("Total time:    %.1f sec\n", elapsed_all);
  printf("Avg nodes:     %.0f (max: %d)\n",
         (double)total_nodes / problems.size(), max_nodes_used);
  printf("Avg time:      %.2f ms/problem\n", total_time_ms / problems.size());
  printf("Speed:         %.0f problems/sec\n", problems.size() / elapsed_all);

  return 0;
}
