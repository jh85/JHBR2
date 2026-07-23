#include "book/opening_book.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace jhbr2 {

using namespace lczero;

OpeningBook::~OpeningBook() { Close(); }

void OpeningBook::Close() {
  if (index_file_.is_open()) index_file_.close();
  if (moves_file_.is_open()) moves_file_.close();
  path_.clear();
  record_count_ = 0;
  flags_ = 0;
  moves_base_ = 0;
  file_size_ = 0;
  loaded_ = false;
}

void OpeningBook::Fail(std::string error) {
  Close();
  last_error_ = std::move(error);
}

uint64_t OpeningBook::Load(const std::string& path) {
  Close();
  last_error_.clear();

  index_file_.open(path, std::ios::in | std::ios::binary);
  moves_file_.open(path, std::ios::in | std::ios::binary);
  if (!index_file_ || !moves_file_) {
    Fail("cannot open YBB file: " + path);
    return 0;
  }

  index_file_.seekg(0, std::ios::end);
  const std::streamoff size = index_file_.tellg();
  if (size < 0) {
    Fail("cannot determine YBB file size: " + path);
    return 0;
  }
  file_size_ = static_cast<uint64_t>(size);
  index_file_.seekg(0, std::ios::beg);

  std::array<uint8_t, kYbbHeaderSize> header{};
  index_file_.read(reinterpret_cast<char*>(header.data()), header.size());
  if (!index_file_ ||
      std::memcmp(header.data(), kYbbMagic.data(), kYbbMagic.size()) != 0) {
    Fail("invalid YBB V1 header: " + path);
    return 0;
  }

  record_count_ = ReadLe64(header.data() + 16);
  flags_ = ReadLe64(header.data() + 24);
  if ((flags_ & ~kYbbKnownFlags) != 0) {
    Fail("unsupported YBB flags: " + std::to_string(flags_));
    return 0;
  }
  if (!YbbIndexSize(record_count_, &moves_base_) ||
      moves_base_ > file_size_) {
    Fail("invalid YBB index size");
    return 0;
  }
  if (!ValidateFile(file_size_)) return 0;

  path_ = path;
  loaded_ = true;
  return record_count_;
}

bool OpeningBook::ReadIndexEntry(uint64_t index, YbbIndexEntry* entry) {
  if (entry == nullptr || index >= record_count_) return false;
  const uint64_t offset =
      kYbbHeaderSize + index * kYbbIndexRecordSize;
  std::array<uint8_t, kYbbIndexRecordSize> bytes{};
  index_file_.clear();
  index_file_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  index_file_.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!index_file_) return false;
  DecodeYbbIndexEntry(bytes.data(), entry);
  return true;
}

bool OpeningBook::ReadMove(const YbbIndexEntry& entry, uint16_t move_index,
                           YbbMoveRecord* move) {
  if (move == nullptr || move_index >= entry.move_count) return false;
  const uint64_t record_size = YbbMoveRecordSize(flags_);
  const uint64_t relative =
      entry.moves_offset + uint64_t(move_index) * record_size;
  if (relative > file_size_ - moves_base_ ||
      record_size > file_size_ - moves_base_ - relative) {
    return false;
  }

  std::array<uint8_t, 6> bytes{};
  moves_file_.clear();
  moves_file_.seekg(
      static_cast<std::streamoff>(moves_base_ + relative), std::ios::beg);
  moves_file_.read(reinterpret_cast<char*>(bytes.data()), record_size);
  if (!moves_file_) return false;
  DecodeYbbMoveRecord(bytes.data(), flags_, move);
  return true;
}

bool OpeningBook::ValidateFile(uint64_t file_size) {
  const uint64_t move_record_size = YbbMoveRecordSize(flags_);
  if (record_count_ == 0) {
    if (moves_base_ != file_size) {
      Fail("empty YBB has trailing data");
      return false;
    }
    return true;
  }

  YbbIndexEntry first;
  YbbIndexEntry last;
  if (!ReadIndexEntry(0, &first) ||
      !ReadIndexEntry(record_count_ - 1, &last)) {
    Fail("cannot read YBB index");
    return false;
  }
  if (first.moves_offset != 0 ||
      last.moves_offset > file_size - moves_base_ ||
      uint64_t(last.move_count) >
          (file_size - moves_base_ - last.moves_offset) / move_record_size) {
    Fail("invalid YBB moves area");
    return false;
  }
  return true;
}

const BookEntry* OpeningBook::Probe(ShogiBoard& board) {
  if (!loaded_) return nullptr;

  PackedSfen target;
  if (!board.ToPackedSfen(&target)) return nullptr;

  uint64_t left = 0;
  uint64_t right = record_count_;
  YbbIndexEntry entry;
  bool found = false;
  while (left < right) {
    const uint64_t middle = left + (right - left) / 2;
    if (!ReadIndexEntry(middle, &entry)) return nullptr;
    const int compare = ComparePackedSfen(target, entry.packed_sfen);
    if (compare < 0) {
      right = middle;
    } else if (compare > 0) {
      left = middle + 1;
    } else {
      found = true;
      break;
    }
  }
  if (!found || entry.move_count == 0) return nullptr;

  // Reject malformed/illegal book moves and choose the highest stored value
  // among the remaining candidates. A valid PetaShock YBB is already sorted,
  // but selecting explicitly avoids depending on record order.
  const MoveList legal_moves = board.GenerateLegalMoves();
  bool have_best = false;
  YbbMoveRecord best;
  for (uint16_t i = 0; i < entry.move_count; ++i) {
    YbbMoveRecord candidate;
    if (!ReadMove(entry, i, &candidate)) return nullptr;
    const Move move = Move::FromRaw(candidate.move);
    const bool legal =
        std::find(legal_moves.begin(), legal_moves.end(), move) !=
        legal_moves.end();
    if (!legal) continue;
    if (!have_best || candidate.eval > best.eval) {
      best = candidate;
      have_best = true;
    }
  }
  if (!have_best) return nullptr;

  result_.move = Move::FromRaw(best.move);
  result_.eval = best.eval;
  result_.depth = best.depth;
  return &result_;
}

}  // namespace jhbr2
