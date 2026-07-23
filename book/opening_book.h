/*
  JHBR2 native YaneuraOu Binary Book (YBB) reader.

  The index and moves areas are accessed on demand, so even multi-million
  position books require negligible resident memory.
*/

#pragma once

#include <cstdint>
#include <fstream>
#include <string>

#include "book/ybb_format.h"
#include "shogi/board.h"

namespace jhbr2 {

struct BookEntry {
  lczero::Move move;
  int eval = 0;   // Evaluation from the side-to-move perspective.
  int depth = 0;
};

class OpeningBook {
 public:
  ~OpeningBook();

  // Open and validate a YBB V1 file. Returns its position count, or zero on
  // failure. Use is_loaded()/last_error() to distinguish an empty valid book
  // from an error.
  uint64_t Load(const std::string& path);
  void Close();

  // Probe by the board's PackedSfen key. Stored ply is intentionally ignored,
  // matching JHBR2's previous text-book behavior and IgnoreBookPly=true.
  const BookEntry* Probe(lczero::ShogiBoard& board);

  bool is_loaded() const { return loaded_; }
  uint64_t position_count() const { return record_count_; }
  const std::string& path() const { return path_; }
  const std::string& last_error() const { return last_error_; }

 private:
  bool ReadIndexEntry(uint64_t index, YbbIndexEntry* entry);
  bool ReadMove(const YbbIndexEntry& entry, uint16_t move_index,
                YbbMoveRecord* move);
  bool ValidateFile(uint64_t file_size);
  void Fail(std::string error);

  std::ifstream index_file_;
  std::ifstream moves_file_;
  std::string path_;
  std::string last_error_;
  uint64_t record_count_ = 0;
  uint64_t flags_ = 0;
  uint64_t moves_base_ = 0;
  uint64_t file_size_ = 0;
  bool loaded_ = false;
  BookEntry result_;
};

}  // namespace jhbr2
