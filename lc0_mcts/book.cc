/*
  JHBR2 — Opening Book (YaneuraOu DB format) implementation

  On-the-fly mode: binary search on sorted file, following YaneuraOu's
  approach. The file must be sorted lexicographically by SFEN line.
*/

#include "lc0_mcts/book.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace lc0_shogi {

OpeningBook::~OpeningBook() {
  if (fs_.is_open()) fs_.close();
}

std::string OpeningBook::NormalizeSfen(const std::string& sfen) {
  // Strip trailing ply number: "lnsgkgsnl/... b - 1" -> "lnsgkgsnl/... b -"
  auto pos = sfen.rfind(' ');
  if (pos == std::string::npos) return sfen;
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

// ---------------------------------------------------------------------------
// Full-load mode
// ---------------------------------------------------------------------------

int OpeningBook::Load(const std::string& path, bool on_the_fly) {
  if (on_the_fly) {
    fs_.open(path, std::ios::in);
    if (!fs_.is_open()) {
      std::cerr << "OpeningBook: cannot open " << path << std::endl;
      return 0;
    }
    fs_.seekg(0, std::ios::end);
    file_size_ = static_cast<int64_t>(fs_.tellg());
    fs_.seekg(0, std::ios::beg);
    on_the_fly_ = true;
    return 0;
  }

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
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
      line.pop_back();
    if (line.empty()) continue;
    if (line[0] == '#') continue;

    if (line.substr(0, 5) == "sfen ") {
      current_sfen = NormalizeSfen(line.substr(5));
      first_move = true;
    } else if (!current_sfen.empty() && first_move) {
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

      if (entries_.find(current_sfen) == entries_.end()) {
        entries_[current_sfen] = std::move(entry);
        count++;
      }
    }
  }

  return count;
}

// ---------------------------------------------------------------------------
// On-the-fly binary search
// ---------------------------------------------------------------------------

std::string OpeningBook::NextSfen(int64_t seek_from, int64_t& last_pos) {
  // Seek back 2 bytes to avoid landing exactly on a newline boundary.
  seek_from = std::max(int64_t(0), seek_from - 2);

  fs_.clear();
  fs_.seekg(seek_from, std::ios::beg);

  std::string line;

  // Discard the first (potentially incomplete) line.
  std::getline(fs_, line);
  last_pos = seek_from + static_cast<int64_t>(line.size()) + 1;

  // Scan forward for the next "sfen " line.
  while (std::getline(fs_, line)) {
    last_pos += static_cast<int64_t>(line.size()) + 1;

    // Strip \r
    while (!line.empty() && line.back() == '\r') line.pop_back();

    if (line.size() >= 5 && line.compare(0, 5, "sfen ") == 0) {
      return NormalizeSfen(line.substr(5));
    }
  }

  return {};  // EOF
}

const BookEntry* OpeningBook::ProbeOnTheFly(const std::string& sfen) {
  std::string key = NormalizeSfen(sfen);

  int64_t lo = 0, hi = file_size_;
  int64_t last_pos;

  while (true) {
    int64_t mid = (lo + hi) / 2;

    auto found_sfen = NextSfen(mid, last_pos);
    if (found_sfen.empty() || key < found_sfen) {
      hi = mid;
    } else if (key > found_sfen) {
      lo = last_pos;
    } else {
      // Exact match — read the first move line.
      break;
    }

    // Minimum SFEN length is ~40 bytes. If the range is too small,
    // do one final check from lo.
    if (lo + 40 > hi) {
      auto final_sfen = NextSfen(lo, last_pos);
      if (final_sfen != key) return nullptr;
      break;
    }
  }

  // We're positioned right after the matched "sfen " line.
  // Read the first move line.
  std::string line;
  while (std::getline(fs_, line)) {
    while (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (line[0] == '#' || (line.size() >= 2 && line[0] == '/' && line[1] == '/'))
      continue;
    // If we hit the next sfen, there were no moves for this position.
    if (line.size() >= 5 && line.compare(0, 5, "sfen ") == 0)
      return nullptr;

    // Parse move line: "<move> <ponder> <eval> <depth> [<count>]"
    std::istringstream iss(line);
    std::string move_usi, ponder_usi;
    int eval = 0, depth = 0;
    if (!(iss >> move_usi >> ponder_usi >> eval >> depth)) return nullptr;

    otf_result_.move_usi = move_usi;
    otf_result_.ponder_usi = ponder_usi;
    otf_result_.eval = eval;
    otf_result_.depth = depth;
    return &otf_result_;
  }

  return nullptr;
}

// ---------------------------------------------------------------------------
// Unified probe
// ---------------------------------------------------------------------------

const BookEntry* OpeningBook::Probe(const std::string& sfen) {
  if (on_the_fly_) {
    return ProbeOnTheFly(sfen);
  }

  std::string key = NormalizeSfen(sfen);
  auto it = entries_.find(key);
  if (it == entries_.end()) return nullptr;
  return &it->second;
}

}  // namespace lc0_shogi
