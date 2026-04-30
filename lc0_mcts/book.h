/*
  JHBR2 — Opening Book (YaneuraOu DB format)

  Reads #YANEURAOU-DB2016 format:
    sfen <SFEN>
    <move> <ponder> <eval> <depth> <count>
    ...

  First move per position is the best (sorted by eval descending).
  Lookup by SFEN string (exact match).
*/

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "shogi/types.h"

namespace lc0_shogi {

using lczero::Move;

struct BookEntry {
  std::string move_usi;    // best move USI string
  std::string ponder_usi;  // ponder move (or "none")
  int eval = 0;            // evaluation from side-to-move
  int depth = 0;           // search depth when analyzed
};

class OpeningBook {
 public:
  // Load book from file. Returns number of positions loaded.
  int Load(const std::string& path);

  // Probe: lookup position by SFEN. Returns nullptr if not found.
  const BookEntry* Probe(const std::string& sfen) const;

  int size() const { return static_cast<int>(entries_.size()); }

 private:
  // Key: SFEN without move number (e.g., "lnsgkgsnl/... b -")
  // Strips the trailing ply number for matching.
  static std::string NormalizeSfen(const std::string& sfen);

  std::unordered_map<std::string, BookEntry> entries_;
};

}  // namespace lc0_shogi
