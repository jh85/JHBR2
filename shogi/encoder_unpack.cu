/*
  JHBR2 — GPU unpack kernels for bit-packed NN input features.
  See shogi/encoder_unpack.h and docs/nyugyoku_dlshogi_features.md.
*/

#include "shogi/encoder_unpack.h"

namespace jhbr2 {

namespace {

// Branchless bit -> float: 1 -> 1.0f, 0 -> 0.0f.
//   -1 (int) == 0xFFFFFFFF;  & 0x3f800000 (bits of 1.0f) -> 1.0f
//    0       == 0x00000000;  & 0x3f800000               -> 0.0f
__device__ __forceinline__ float BitToFloat(int bit) {
  return __int_as_float((-bit) & 0x3f800000);
}

// features1: one thread per positional plane; expands 81 squares (1 bit each).
__global__ void UnpackF1Kernel(const uint8_t* __restrict__ packed,
                               float* __restrict__ out,
                               int channels, int f1_bytes, int num_f1) {
  const int b = blockIdx.x;
  const int p = threadIdx.x;
  if (p >= num_f1) return;

  const uint8_t* pk = packed + static_cast<size_t>(b) * f1_bytes;
  float* dst = out + (static_cast<size_t>(b) * channels + p) * 81;
  const int base = p * 81;
#pragma unroll
  for (int i = 0; i < 81; ++i) {
    const int j = base + i;
    const int bit = (pk[j >> 3] >> (j & 7)) & 1;
    dst[i] = BitToFloat(bit);
  }
}

// features2: one thread per uniform plane; broadcasts a single bit to 81 cells.
__global__ void UnpackF2Kernel(const uint8_t* __restrict__ packed,
                               float* __restrict__ out,
                               int channels, int f2_bytes,
                               int num_f1, int num_f2) {
  const int b = blockIdx.x;
  const int j = threadIdx.x;
  if (j >= num_f2) return;

  const uint8_t* pk = packed + static_cast<size_t>(b) * f2_bytes;
  const int bit = (pk[j >> 3] >> (j & 7)) & 1;
  const float v = BitToFloat(bit);

  float* dst = out + (static_cast<size_t>(b) * channels + num_f1 + j) * 81;
#pragma unroll
  for (int i = 0; i < 81; ++i) dst[i] = v;
}

}  // namespace

void LaunchUnpackFeatures(int batch, int channels,
                          int num_f1, int f1_bytes,
                          int num_f2, int f2_bytes,
                          const uint8_t* d_packed_f1,
                          const uint8_t* d_packed_f2,
                          float* d_input, cudaStream_t stream) {
  if (batch <= 0) return;
  UnpackF1Kernel<<<batch, num_f1, 0, stream>>>(
      d_packed_f1, d_input, channels, f1_bytes, num_f1);
  UnpackF2Kernel<<<batch, num_f2, 0, stream>>>(
      d_packed_f2, d_input, channels, f2_bytes, num_f1, num_f2);
}

}  // namespace jhbr2
