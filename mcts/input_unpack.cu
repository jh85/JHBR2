#include "mcts/input_unpack.h"

#include <cstddef>

#include <cuda_runtime.h>

namespace jhbr2 {
namespace {

__global__ void UnpackBitsToFloatKernel(const uint8_t* packed, float* dense,
                                        size_t total_bits) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_bits) return;
  const uint8_t byte = packed[idx >> 3];
  dense[idx] = ((byte >> (idx & 7)) & 1) ? 1.0f : 0.0f;
}

}  // namespace

cudaError_t UnpackBitsToFloatAsync(const uint8_t* packed, float* dense,
                                   int batch_size, int channels, int squares,
                                   cudaStream_t stream) {
  const size_t total_bits =
      static_cast<size_t>(batch_size) * channels * squares;
  if (total_bits == 0) return cudaSuccess;

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total_bits + kThreads - 1) / kThreads);
  UnpackBitsToFloatKernel<<<blocks, kThreads, 0, stream>>>(
      packed, dense, total_bits);
  return cudaGetLastError();
}

}  // namespace jhbr2
