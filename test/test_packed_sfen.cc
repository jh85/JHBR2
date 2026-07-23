#include <array>
#include <cstdint>
#include <iostream>
#include <string>

#include "shogi/board.h"

namespace {

int failures = 0;

void Check(const std::string& name, bool condition) {
  if (condition) {
    std::cout << "  OK    " << name << '\n';
  } else {
    std::cout << "  FAIL  " << name << '\n';
    ++failures;
  }
}

int HexDigit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

lczero::PackedSfen FromHex(const std::string& hex) {
  lczero::PackedSfen packed;
  if (hex.size() != packed.data.size() * 2) return packed;
  for (size_t i = 0; i < packed.data.size(); ++i) {
    const int high = HexDigit(hex[i * 2]);
    const int low = HexDigit(hex[i * 2 + 1]);
    packed.data[i] = static_cast<uint8_t>((high << 4) | low);
  }
  return packed;
}

void CheckPosition(const std::string& name, const std::string& sfen,
                   const std::string& expected_hex) {
  lczero::ShogiBoard board;
  Check(name + " parse SFEN", board.SetFromSfen(sfen));

  lczero::PackedSfen packed;
  Check(name + " encode", board.ToPackedSfen(&packed));
  Check(name + " matches cshogi", packed == FromHex(expected_hex));

  lczero::ShogiBoard decoded;
  Check(name + " decode", decoded.SetFromPackedSfen(packed, board.ply()));
  Check(name + " round trip", decoded.ToSfen() == board.ToSfen());
}

}  // namespace

int main() {
  CheckPosition(
      "startpos",
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/"
      "LNSGKGSNL b - 1",
      "58a451220ceb67227e9653221caf447824c22b119e53221ceb6f223e9651220c");

  CheckPosition(
      "after 7g7f",
      "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/"
      "LNSGKGSNL w - 2",
      "59a451220ceb67227e9693241caf447824c22b119e53121ceb6f223e9651220c");

  CheckPosition(
      "hands and promotions",
      "4k4/9/2+p6/9/4+B4/9/6+r2/9/4K4 w Rb 77",
      "59240000807f00e00b0068000080cf5392244992244929a5d45a6b6badb5f7de");

  std::cout << "\n=== Summary: " << failures << " failed ===\n";
  return failures == 0 ? 0 : 1;
}
