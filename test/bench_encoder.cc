/*
  JHBR2 - Encoder benchmark

  Measures EncodeShogiPosition() calls per second on representative SFENs.
*/

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/encoder.h"

using namespace lczero;
using Clock = std::chrono::steady_clock;

int main() {
  ShogiTables::Init();
  ShogiEncoderTables::Init();

  const std::vector<std::string> sfens = {
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
      "ln1gk2nl/1r2g2b1/p1sppsppp/2p3p2/1p7/2P1P4/PP1PSP1PP/1BG4R1/LN2KG1NL b - 1",
      "l3k2nl/4g2b1/p1sppsppp/2p3p2/1p7/2P1P4/PP1PSP1PP/1BG4R1/LN2KG1NL b RNPrnp 1",
      "3g1k3/5+P3/4p1+Spp/p4N3/6p2/1P1P5/P3+b1P1P/2+r6/K1S3GNL w RBG2SN4Pl2p 1",
  };

  std::vector<ShogiBoard> boards;
  boards.reserve(sfens.size());
  for (const auto& sfen : sfens) {
    ShogiBoard b;
    b.SetFromSfen(sfen);
    boards.push_back(b);
  }

  volatile float sink = 0.0f;
  for (const auto& b : boards) {
    auto planes = EncodeShogiPosition(b);
    sink += planes[0].data[0];
  }

  constexpr int repeats = 250000;
  uint64_t total_calls = 0;

  auto t0 = Clock::now();
  for (int r = 0; r < repeats; ++r) {
    for (const auto& b : boards) {
      auto planes = EncodeShogiPosition(b);
      sink += planes[43].data[0];
      ++total_calls;
    }
  }
  auto t1 = Clock::now();

  double secs = std::chrono::duration<double>(t1 - t0).count();
  std::printf("Encoder:\n");
  std::printf("  %lu calls in %.3f sec\n", total_calls, secs);
  std::printf("  %.0f calls/sec\n", total_calls / secs);
  std::printf("  sink %.1f\n", static_cast<double>(sink));
  return 0;
}
