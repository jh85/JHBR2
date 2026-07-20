#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/types.h"

using lczero::Move;
using lczero::MoveList;
using lczero::ShogiBoard;

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

std::set<std::string> MoveSet(const MoveList& moves) {
  std::set<std::string> result;
  for (const Move& move : moves) result.insert(move.ToString());
  return result;
}

Move MateInOneOracle(ShogiBoard& board) {
  const MoveList checks = board.GenerateCheckingMoves();
  for (const Move& move : checks) {
    const auto undo = board.DoMove(move, true);
    const bool mate = !board.HasLegalEvasion();
    board.UndoMove(move, undo);
    if (mate) return move;
  }
  return Move();
}

bool ValidateFastMove(ShogiBoard& board, Move move,
                      const std::set<std::string>& legal_before,
                      std::string* reason) {
  if (move.is_null()) return true;
  if (!legal_before.contains(move.ToString())) {
    *reason = "returned move is not legal: " + move.ToString();
    return false;
  }
  if (move.is_drop() && move.drop_piece() == lczero::kPawn) {
    *reason = "returned illegal pawn-drop mate: " + move.ToString();
    return false;
  }

  const auto undo = board.DoMove(move);
  const bool gives_check = board.InCheck();
  const bool mate = gives_check && !board.HasLegalEvasion();
  board.UndoMove(move, undo);
  if (!mate) {
    *reason = "returned move does not mate: " + move.ToString();
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const bool verdict_mode =
      argc >= 2 && std::string(argv[1]) == "--verdicts";
  const std::string path =
      verdict_mode ? (argc >= 3 ? argv[2] : "test/positions.txt")
                   : (argc >= 2 ? argv[1] : "test/positions.txt");
  const auto sfens = LoadSfens(path);
  if (sfens.empty()) {
    std::cerr << "No positions loaded from " << path << '\n';
    return 1;
  }

  lczero::ShogiTables::Init();
  if (verdict_mode) {
    for (const auto& sfen : sfens) {
      ShogiBoard board;
      if (!board.SetFromSfen(sfen)) {
        std::cout << "INVALID\n";
        continue;
      }
      std::cout << board.FindMateInOne().ToString() << '\n';
    }
    return 0;
  }

  int passed = 0;
  int failed = 0;
  int fast_mates = 0;
  int oracle_mates = 0;
  int incomplete_positions = 0;
  constexpr int kMaxDetails = 30;

  for (size_t i = 0; i < sfens.size(); ++i) {
    ShogiBoard fast_board;
    ShogiBoard oracle_board;
    if (!fast_board.SetFromSfen(sfens[i]) ||
        !oracle_board.SetFromSfen(sfens[i])) {
      ++failed;
      continue;
    }

    const std::string sfen_before = fast_board.ToSfen();
    const uint64_t hash_before = fast_board.Hash();
    const auto side_before = fast_board.side_to_move();

    if (!fast_board.king_square(lczero::BLACK).IsValid() ||
        !fast_board.king_square(lczero::WHITE).IsValid()) {
      ++incomplete_positions;
      const Move fast = fast_board.FindMateInOne();
      if (fast.is_null() && fast_board.ToSfen() == sfen_before &&
          fast_board.Hash() == hash_before &&
          fast_board.side_to_move() == side_before) {
        ++passed;
      } else {
        ++failed;
      }
      continue;
    }

    const std::set<std::string> legal_before =
        MoveSet(fast_board.GenerateLegalMoves());

    const Move fast = fast_board.InCheck()
                          ? fast_board.FindMateInOne()
                          : fast_board.FindMateInOneNonCheck();
    const Move oracle = MateInOneOracle(oracle_board);
    if (!fast.is_null()) ++fast_mates;
    if (!oracle.is_null()) ++oracle_mates;

    std::string reason;
    bool ok = fast.is_null() == oracle.is_null();
    if (!ok) {
      reason = "verdict mismatch: fast=" + fast.ToString() +
               " oracle=" + oracle.ToString();
    }
    if (ok) {
      ok = ValidateFastMove(fast_board, fast, legal_before, &reason);
    }
    if (ok && (fast_board.ToSfen() != sfen_before ||
               fast_board.Hash() != hash_before ||
               fast_board.side_to_move() != side_before ||
               MoveSet(fast_board.GenerateLegalMoves()) != legal_before)) {
      ok = false;
      reason = "board state changed";
    }

    if (ok) {
      ++passed;
    } else {
      ++failed;
      if (failed <= kMaxDetails) {
        std::cerr << "FAIL [" << i + 1 << "] " << reason << '\n'
                  << "  " << sfens[i] << '\n';
      }
    }
  }

  std::cout << "Positions: " << sfens.size() << '\n'
            << "Fast mates: " << fast_mates << '\n'
            << "Oracle mates: " << oracle_mates << '\n'
            << "Incomplete positions: " << incomplete_positions << '\n'
            << "Passed: " << passed << '\n'
            << "Failed: " << failed << '\n';
  return failed == 0 ? 0 : 1;
}
