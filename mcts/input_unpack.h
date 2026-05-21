#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace jhbr2 {

cudaError_t UnpackBitsToFloatAsync(const uint8_t* packed, float* dense,
                                   int batch_size, int channels, int squares,
                                   cudaStream_t stream);

}  // namespace jhbr2
