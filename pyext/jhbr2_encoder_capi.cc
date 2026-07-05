/*
  C-ABI wrapper around the verified JHBR2 encoder (EncodeShogiPosition), for
  fast shard generation from Python via ctypes. No pybind11 needed.

  Layout of `out` for position i, plane c, cell (= rank*9+file):
      out[(i*kShogiInputPlanes + c) * 81 + cell]
  which reshapes to (N, 148, 9, 9) matching sfen_to_planes()'s [c, rank, file].

  Build: pyext/build.sh
*/

#include <cstring>
#include <string>

#include "shogi/board.h"
#include "shogi/encoder.h"

using namespace lczero;

extern "C" {

// Initialize the global attack / encoder tables. Idempotent; call once per
// process (the Python wrapper does this on load).
void jhbr2_init() {
  ShogiTables::Init();
  ShogiEncoderTables::Init();
}

int jhbr2_num_planes() { return kShogiInputPlanes; }

// Encode `n` SFEN strings into `out` (n * kShogiInputPlanes * 81 floats).
// A malformed SFEN yields an all-zero block. Returns the number encoded OK.
int jhbr2_encode_sfens(const char** sfens, int n, float* out) {
  const int planes_n = kShogiInputPlanes;
  int ok = 0;
  for (int i = 0; i < n; ++i) {
    float* dst = out + static_cast<size_t>(i) * planes_n * 81;
    ShogiBoard b;
    if (sfens[i] == nullptr || !b.SetFromSfen(sfens[i])) {
      std::memset(dst, 0, static_cast<size_t>(planes_n) * 81 * sizeof(float));
      continue;
    }
    ShogiInputPlanes p = EncodeShogiPosition(b);
    for (int c = 0; c < planes_n; ++c) {
      std::memcpy(dst + static_cast<size_t>(c) * 81, p[c].data,
                  81 * sizeof(float));
    }
    ++ok;
  }
  return ok;
}

}  // extern "C"
