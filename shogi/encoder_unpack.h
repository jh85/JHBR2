/*
  JHBR2 — GPU expansion of bit-packed NN input features (dlshogi-style).

  PackShogiPosition() (shogi/encoder.cc) produces two packed bitsets per
  position; these kernels expand them, on the GPU, into the dense float input
  tensor TensorRT consumes. Transferring packed bits instead of full float
  planes cuts the host->device payload from ~48 KB to ~300 B per position.

  See docs/nyugyoku_dlshogi_features.md for the layout contract.
*/

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace jhbr2 {

// Expand packed features into d_input (batch * channels * 81 floats), laid out
// as features1 planes [0, num_f1) followed by features2 planes [num_f1,
// num_f1+num_f2). The kernels write every cell of every active row, so d_input
// needs no pre-zeroing for rows in [0, batch).
//
//   d_packed_f1: batch * f1_bytes bytes — 1 bit per square, num_f1 planes.
//   d_packed_f2: batch * f2_bytes bytes — 1 bit per plane, num_f2 planes.
void LaunchUnpackFeatures(int batch, int channels,
                          int num_f1, int f1_bytes,
                          int num_f2, int f2_bytes,
                          const uint8_t* d_packed_f1,
                          const uint8_t* d_packed_f2,
                          float* d_input, cudaStream_t stream);

}  // namespace jhbr2
