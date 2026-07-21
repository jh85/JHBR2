#include <cstdio>
#include <string>

#include "shogi/types.h"
#include "usi/search_info.h"

namespace {

using jhbr2::FormatUSISearchInfo;
using jhbr2::USISearchInfo;
using lczero::Move;

int failures = 0;

void CheckEqual(const char* name, const std::string& actual,
                const std::string& expected) {
  if (actual == expected) return;
  std::printf("  FAIL  %s\n    expected: %s\n    actual:   %s\n", name,
              expected.c_str(), actual.c_str());
  ++failures;
}

void TestCompleteRecord() {
  USISearchInfo info;
  info.depth = 25;
  info.seldepth = 25;
  info.multipv = 1;
  info.score_cp = 4;
  info.nodes = 85109;
  info.nps = 16737;
  info.hashfull = 28;
  info.time_ms = 5085;
  info.pv = {Move::Parse("7g7f"), Move::Parse("3c3d"),
             Move::Parse("2g2f")};

  CheckEqual(
      "complete search record", FormatUSISearchInfo(info),
      "info depth 25 seldepth 25 multipv 1 score cp 4 nodes 85109 "
      "nps 16737 hashfull 28 time 5085 pv 7g7f 3c3d 2g2f");
}

void TestEmptyPVAndBounds() {
  USISearchInfo info;
  info.depth = -1;
  info.score_cp = -12;
  info.nodes = 42;
  info.nps = 21;
  info.hashfull = 1200;
  info.time_ms = 2000;

  CheckEqual("empty PV and bounded fields", FormatUSISearchInfo(info),
             "info depth 0 multipv 1 score cp -12 nodes 42 nps 21 "
             "hashfull 1000 time 2000");
}

}  // namespace

int main() {
  TestCompleteRecord();
  TestEmptyPVAndBounds();

  std::printf("\n=== USI search info: %d failed ===\n", failures);
  return failures == 0 ? 0 : 1;
}
