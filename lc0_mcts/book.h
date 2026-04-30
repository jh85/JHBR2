/*
  JHBR2 — Opening Book (YaneuraOu DB format)

  Reads #YANEURAOU-DB2016 format:
    sfen <SFEN>
    <move> <ponder> <eval> <depth> <count>
    ...

  First move per position is the best (sorted by eval descending).
  Lookup by SFEN string (exact match, ply number ignored).

  Two modes:
    - Full load: entire book loaded into memory (fast probe, slow startup)
    - On-the-fly: binary search on sorted file (zero startup, zero memory)
*/

#pragma once

#include <fstream>
#include <string>
#include <unordered_map>

namespace lc0_shogi {

struct BookEntry {
  std::string move_usi;    // best move USI string
  std::string ponder_usi;  // ponder move (or "none")
  int eval = 0;            // evaluation from side-to-move
  int depth = 0;           // search depth when analyzed
};

class OpeningBook {
 public:
  ~OpeningBook();

  // Load book from file.
  // on_the_fly=false: loads entire file into memory, returns position count.
  // on_the_fly=true: opens file for binary search, returns 0 (no preload).
  int Load(const std::string& path, bool on_the_fly = false);

  // Probe: lookup position by SFEN. Returns nullptr if not found.
  const BookEntry* Probe(const std::string& sfen);

  bool is_loaded() const { return !entries_.empty() || on_the_fly_; }

 private:
  static std::string NormalizeSfen(const std::string& sfen);

  // On-the-fly binary search helpers
  std::string NextSfen(int64_t seek_from, int64_t& last_pos);
  const BookEntry* ProbeOnTheFly(const std::string& sfen);

  std::unordered_map<std::string, BookEntry> entries_;

  // On-the-fly state
  bool on_the_fly_ = false;
  std::fstream fs_;
  int64_t file_size_ = 0;
  BookEntry otf_result_;  // reusable buffer for on-the-fly results
};

}  // namespace lc0_shogi
