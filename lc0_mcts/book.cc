/*
  JHBR2 — Opening Book (YaneuraOu DB format) implementation
*/

#include "lc0_mcts/book.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace lc0_shogi {

std::string OpeningBook::NormalizeSfen(const std::string& sfen) {
  // Strip trailing ply number: "lnsgkgsnl/... b - 1" -> "lnsgkgsnl/... b -"
  auto pos = sfen.rfind(' ');
  if (pos == std::string::npos) return sfen;
  // Check that everything after the last space is a number
  bool all_digits = true;
  for (size_t i = pos + 1; i < sfen.size(); i++) {
    if (!std::isdigit(static_cast<unsigned char>(sfen[i]))) {
      all_digits = false;
      break;
    }
  }
  if (all_digits && pos > 0) return sfen.substr(0, pos);
  return sfen;
}

int OpeningBook::Load(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    std::cerr << "OpeningBook: cannot open " << path << std::endl;
    return 0;
  }

  std::string line;
  std::string current_sfen;
  bool first_move = false;
  int count = 0;

  while (std::getline(ifs, line)) {
    // Strip trailing \r
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
      line.pop_back();
    if (line.empty()) continue;

    // Skip header
    if (line[0] == '#') continue;

    if (line.substr(0, 5) == "sfen ") {
      // New position: "sfen <board> <side> <hand> <ply>"
      current_sfen = NormalizeSfen(line.substr(5));
      first_move = true;
    } else if (!current_sfen.empty() && first_move) {
      // First move line for this position: "<move> <ponder> <eval> <depth> [<count>]"
      first_move = false;
      std::istringstream iss(line);
      std::string move_usi, ponder_usi;
      int eval = 0, depth = 0;
      if (!(iss >> move_usi >> ponder_usi >> eval >> depth)) continue;

      BookEntry entry;
      entry.move_usi = move_usi;
      entry.ponder_usi = ponder_usi;
      entry.eval = eval;
      entry.depth = depth;

      // Only store first occurrence (best move per position)
      if (entries_.find(current_sfen) == entries_.end()) {
        entries_[current_sfen] = std::move(entry);
        count++;
      }
    }
    // Subsequent move lines for the same position are ignored (we only want the best)
  }

  return count;
}

const BookEntry* OpeningBook::Probe(const std::string& sfen) const {
  std::string key = NormalizeSfen(sfen);
  auto it = entries_.find(key);
  if (it == entries_.end()) return nullptr;
  return &it->second;
}

}  // namespace lc0_shogi
