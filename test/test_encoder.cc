#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/encoder.h"

using namespace lczero;

namespace {

std::vector<int> ActiveNyugyokuPlanes(const ShogiInputPlanes& planes) {
  std::vector<int> active;
  for (int p = kShogiNyugyokuBasePlane; p < kShogiInputPlanes; ++p) {
    bool any = false;
    bool all_same = true;
    const float first = planes[p].data[0];
    for (float v : planes[p].data) {
      any = any || v != 0.0f;
      all_same = all_same && v == first;
    }
    if (!all_same) {
      std::cerr << "Plane " << p << " is not a broadcast plane\n";
      std::exit(1);
    }
    if (any) active.push_back(p);
  }
  return active;
}

bool PlaneIsAll(const ShogiInputPlanes& planes, int p, float expected) {
  for (float v : planes[p].data) {
    if (v != expected) return false;
  }
  return true;
}

bool CheckActive(const std::string& sfen, const std::vector<int>& expected) {
  ShogiBoard board;
  if (!board.SetFromSfen(sfen)) {
    std::cerr << "Failed to parse SFEN: " << sfen << "\n";
    return false;
  }
  const auto planes = EncodeShogiPosition(board);
  const auto active = ActiveNyugyokuPlanes(planes);
  if (active == expected) return true;

  std::cerr << "Nyugyoku planes mismatch for " << sfen << "\n";
  std::cerr << "  expected:";
  for (int p : expected) std::cerr << ' ' << p;
  std::cerr << "\n  got:";
  for (int p : active) std::cerr << ' ' << p;
  std::cerr << "\n";
  return false;
}

bool CheckHandPlanes() {
  ShogiBoard board;
  const std::string sfen =
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL "
      "b 10P5L4N4S4G3B3R 1";
  if (!board.SetFromSfen(sfen)) {
    std::cerr << "Failed to parse SFEN: " << sfen << "\n";
    return false;
  }

  const auto planes = EncodeShogiPosition(board);
  bool ok = true;
  for (int i = 0; i < 8; ++i) {
    ok &= PlaneIsAll(planes, kShogiHandBasePlane + i, 1.0f);
  }
  ok &= PlaneIsAll(planes, kShogiHandBasePlane + 8, 1.0f);
  ok &= PlaneIsAll(planes, kShogiHandBasePlane + 11, 1.0f);
  ok &= PlaneIsAll(planes, kShogiHandBasePlane + 12, 1.0f);
  ok &= PlaneIsAll(planes, kShogiHandBasePlane + 15, 1.0f);
  ok &= PlaneIsAll(planes, kShogiHandBasePlane + 16, 1.0f);
  ok &= PlaneIsAll(planes, kShogiHandBasePlane + 19, 1.0f);
  ok &= PlaneIsAll(planes, kShogiHandBasePlane + 20, 1.0f);
  ok &= PlaneIsAll(planes, kShogiHandBasePlane + 23, 1.0f);
  ok &= PlaneIsAll(planes, kShogiHandBasePlane + 24, 1.0f);
  ok &= PlaneIsAll(planes, kShogiHandBasePlane + 25, 1.0f);
  ok &= PlaneIsAll(planes, kShogiHandBasePlane + 26, 1.0f);
  ok &= PlaneIsAll(planes, kShogiHandBasePlane + 27, 1.0f);
  return ok;
}

}  // namespace

int main() {
  ShogiTables::Init();
  ShogiEncoderTables::Init();

  bool ok = true;
  ok &= (kShogiInputPlanes == 148);
  ok &= CheckHandPlanes();

  ok &= CheckActive(
      "4K4/RPPPP4/9/9/9/9/9/9/4k4 b - 1",
      {86, 92, 116, 117});

  // restPoint == 20 is outside dlshogi's explicit 0..19 buckets.
  ok &= CheckActive(
      "4K4/RPPP5/9/9/9/9/9/9/4k4 b - 1",
      {86, 93, 117});

  // White to move: WHITE remains a 27-point side even though it is encoded as
  // side 0. BLACK's 28-point nyugyoku status is encoded as side 1.
  ok &= CheckActive(
      "4K4/RPPPP4/9/9/9/9/9/9/4k4 w - 1",
      {86, 117, 123, 147});

  if (!ok) return 1;
  std::cout << "test_encoder: all checks passed\n";
  return 0;
}
