#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "shogi/types.h"

namespace jhbr2 {

// One complete, GUI-facing USI search record.  JHBR2 currently reports a
// single PV, but keeping multipv explicit matches the common YaneuraOu form
// and avoids making zero mean "unspecified" in clients.
struct USISearchInfo {
  int depth = 0;
  int seldepth = 0;
  int multipv = 1;
  int score_cp = 0;
  std::uint64_t nodes = 0;
  std::uint64_t nps = 0;
  int hashfull = 0;
  std::uint64_t time_ms = 0;
  std::vector<lczero::Move> pv;
};

// Formats fields in the same canonical order used by YaneuraOu's InfoFull
// output.  USI parsers must accept any field order, but a single formatter
// keeps periodic and final records identical and easy for GUIs to consume.
std::string FormatUSISearchInfo(const USISearchInfo& info);

}  // namespace jhbr2
