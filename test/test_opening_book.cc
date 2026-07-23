#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "book/opening_book.h"
#include "book/ybb_format.h"
#include "shogi/board.h"

namespace {

using jhbr2::YbbIndexEntry;
using jhbr2::YbbMoveRecord;
using lczero::Move;
using lczero::PackedSfen;
using lczero::ShogiBoard;

struct PositionRecord {
  PackedSfen key;
  uint16_t ply;
  std::vector<YbbMoveRecord> moves;
};

int failures = 0;

void Check(const std::string& name, bool condition) {
  if (condition) {
    std::cout << "  OK    " << name << '\n';
  } else {
    std::cout << "  FAIL  " << name << '\n';
    ++failures;
  }
}

PositionRecord MakeRecord(const std::string& sfen,
                          std::vector<YbbMoveRecord> moves) {
  ShogiBoard board;
  board.SetFromSfen(sfen);
  PositionRecord record;
  board.ToPackedSfen(&record.key);
  record.ply = static_cast<uint16_t>(board.ply());
  record.moves = std::move(moves);
  return record;
}

bool WriteBook(const std::filesystem::path& path,
               std::vector<PositionRecord> records) {
  std::sort(records.begin(), records.end(),
            [](const PositionRecord& lhs, const PositionRecord& rhs) {
              return ComparePackedSfen(lhs.key, rhs.key) < 0;
            });

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output ||
      !jhbr2::WriteYbbHeader(output, records.size(),
                             jhbr2::kYbbFlagMoveDepth)) {
    return false;
  }

  uint64_t move_offset = 0;
  for (const auto& record : records) {
    YbbIndexEntry entry;
    entry.packed_sfen = record.key;
    entry.moves_offset = move_offset;
    entry.ply = record.ply;
    entry.move_count = static_cast<uint16_t>(record.moves.size());
    std::array<uint8_t, 44> bytes{};
    jhbr2::EncodeYbbIndexEntry(entry, &bytes);
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    move_offset += record.moves.size() * 6;
  }

  for (const auto& record : records) {
    for (const auto& move : record.moves) {
      std::array<uint8_t, 6> bytes{};
      jhbr2::EncodeYbbMoveRecord(move, &bytes);
      output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  return static_cast<bool>(output);
}

}  // namespace

int main() {
  lczero::ShogiTables::Init();

  const auto unique =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto path = std::filesystem::temp_directory_path() /
                    ("jhbr2_opening_book_" + std::to_string(unique) + ".ybb");

  const std::string start =
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/"
      "LNSGKGSNL b - 1";
  const std::string after_7g7f =
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/"
      "LNSGKGSNL w - 2";

  const Move move_7g7f = Move::Parse("7g7f");
  const Move move_2g2f = Move::Parse("2g2f");
  const Move move_3c3d = Move::Parse("3c3d");

  Check("write temporary YBB",
        WriteBook(path,
                  {
                      MakeRecord(start,
                                 {{move_7g7f.raw(), 100, 12},
                                  {move_2g2f.raw(), 120, 10},
                                  {0xffff, 30000, 99}}),
                      MakeRecord(after_7g7f,
                                 {{move_3c3d.raw(), -45, 7}}),
                  }));

  jhbr2::OpeningBook book;
  Check("opens YBB", book.Load(path.string()) == 2 && book.is_loaded());
  Check("reports position count", book.position_count() == 2);

  ShogiBoard black;
  black.SetFromSfen(start);
  const auto* black_entry = book.Probe(black);
  Check("selects highest legal evaluation",
        black_entry && black_entry->move == move_2g2f &&
            black_entry->eval == 120 && black_entry->depth == 10);

  ShogiBoard white;
  white.SetFromSfen(after_7g7f);
  const auto* white_entry = book.Probe(white);
  Check("finds Gote position",
        white_entry && white_entry->move == move_3c3d &&
            white_entry->eval == -45 && white_entry->depth == 7);

  ShogiBoard unknown;
  unknown.SetFromSfen(
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/P8/1PPPPPPPP/1B5R1/"
      "LNSGKGSNL w - 2");
  Check("misses unknown position", book.Probe(unknown) == nullptr);

  jhbr2::OpeningBook invalid;
  Check("rejects non-YBB input", invalid.Load(__FILE__) == 0 &&
                                      !invalid.is_loaded() &&
                                      !invalid.last_error().empty());

  std::error_code error;
  const bool removed = std::filesystem::remove(path, error);
  Check("temporary book removed", removed && !error);

  std::cout << "\n=== Summary: " << failures << " failed ===\n";
  return failures == 0 ? 0 : 1;
}
