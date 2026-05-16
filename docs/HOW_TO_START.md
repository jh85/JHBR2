# HOW TO START

This document assumes you have:

- CMake and a C++20 compiler, such as `g++` or `clang++`
- CUDA installed at `$CUDA_PATH`
- TensorRT installed at `$TENSORRT_PATH`
- cuDNN installed at `$CUDNN_PATH`
- `shogi_bt4_epoch23_dynamic.onnx`
- a fresh checkout of the JHBR2 repo

The native TensorRT build does not load ONNX directly at runtime. Build the
`jhbr2` binary first, then convert the ONNX model to a TensorRT `.engine` file
with `trtexec`.

## 1. Set Paths

```bash
export CUDA_PATH=/path/to/cuda
export TENSORRT_PATH=/path/to/TensorRT
export CUDNN_PATH=/path/to/cudnn
export MODEL_ONNX=/path/to/shogi_bt4_epoch23_dynamic.onnx

export LD_LIBRARY_PATH=$TENSORRT_PATH/lib:$CUDNN_PATH/lib:$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

If your cuDNN package uses `lib64` instead of `lib`, replace
`$CUDNN_PATH/lib` with `$CUDNN_PATH/lib64`.

Quick checks:

```bash
$CUDA_PATH/bin/nvcc --version
$TENSORRT_PATH/bin/trtexec --version
ls "$MODEL_ONNX"
```

## 2. Configure

From the repo root:

```bash
cd /path/to/JHBR2

cmake -S . -B build-trt -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON \
  -DCUDAToolkit_ROOT="$CUDA_PATH" \
  -DTENSORRT_ROOT="$TENSORRT_PATH" \
  -DCUDNN_ROOT="$CUDNN_PATH"
```

If you want a specific compiler:

```bash
CC=gcc CXX=g++ cmake -S . -B build-trt -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON \
  -DCUDAToolkit_ROOT="$CUDA_PATH" \
  -DTENSORRT_ROOT="$TENSORRT_PATH" \
  -DCUDNN_ROOT="$CUDNN_PATH"
```

or:

```bash
CC=clang CXX=clang++ cmake -S . -B build-trt -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON \
  -DCUDAToolkit_ROOT="$CUDA_PATH" \
  -DTENSORRT_ROOT="$TENSORRT_PATH" \
  -DCUDNN_ROOT="$CUDNN_PATH"
```

## 3. Build

```bash
cmake --build build-trt -j"$(nproc)"
```

The binary should be:

```bash
build-trt/jhbr2
```

Check dynamic libraries:

```bash
ldd build-trt/jhbr2 | grep -E 'nvinfer|cudart|not found'
```

There should be no `not found` lines. `libnvinfer` should resolve from
`$TENSORRT_PATH/lib`.

## 4. Run Basic Tests

```bash
./build-trt/test_movegen test/positions.txt
./build-trt/test_check_movegen
./build-trt/test_shallow_mate
```

## 5. Build A TensorRT Engine

Create an engine for the JHBR2 model:

```bash
mkdir -p engines

$TENSORRT_PATH/bin/trtexec \
  --onnx="$MODEL_ONNX" \
  --saveEngine=engines/shogi_bt4_epoch23_trt_b256.engine \
  --fp16 \
  --minShapes=input_planes:1x48x9x9 \
  --optShapes=input_planes:128x48x9x9 \
  --maxShapes=input_planes:256x48x9x9 \
  --memPoolSize=workspace:4096M
```

If engine build fails because of memory, retry with a smaller profile:

```bash
$TENSORRT_PATH/bin/trtexec \
  --onnx="$MODEL_ONNX" \
  --saveEngine=engines/shogi_bt4_epoch23_trt_b128.engine \
  --fp16 \
  --minShapes=input_planes:1x48x9x9 \
  --optShapes=input_planes:128x48x9x9 \
  --maxShapes=input_planes:128x48x9x9 \
  --memPoolSize=workspace:2048M
```

TensorRT engines are machine-specific. Rebuild the `.engine` file when you move
to a different GPU architecture, TensorRT version, CUDA stack, or driver stack.

## 6. USI Smoke Test

Use the engine path from the previous step:

```bash
ENGINE=/path/to/JHBR2/engines/shogi_bt4_epoch23_trt_b256.engine

printf 'usi\nsetoption name OnnxModel value %s\nsetoption name ModelFormat value jhbr2\nsetoption name NumGPUs value 1\nsetoption name WorkersPerGpu value 2\nsetoption name MinibatchSize value 128\nsetoption name MaxGpuBatch value 256\nisready\nposition startpos\ngo nodes 256\nquit\n' "$ENGINE" \
| ./build-trt/jhbr2
```

Expected:

- `usiok`
- `readyok`
- TensorRT log line showing the engine loaded
- a legal `bestmove`

The option name `OnnxModel` is historical. In a native TensorRT build, pass the
serialized TensorRT `.engine` file there.

## 7. Multi-GPU Run

For a two-GPU machine:

```bash
ENGINE=/path/to/JHBR2/engines/shogi_bt4_epoch23_trt_b256.engine

printf 'usi\nsetoption name OnnxModel value %s\nsetoption name ModelFormat value jhbr2\nsetoption name NumGPUs value 2\nsetoption name WorkersPerGpu value 2\nsetoption name MinibatchSize value 128\nsetoption name MaxGpuBatch value 256\nisready\nposition startpos\ngo byoyomi 1000\nquit\n' "$ENGINE" \
| ./build-trt/jhbr2
```

If you built the `b128` engine, use:

```text
setoption name MaxGpuBatch value 128
```

## Troubleshooting

If CMake cannot find TensorRT:

```bash
ls "$TENSORRT_PATH/include/NvInfer.h"
ls "$TENSORRT_PATH/lib/libnvinfer.so"*
```

If runtime loading fails:

```bash
export LD_LIBRARY_PATH=$TENSORRT_PATH/lib:$CUDNN_PATH/lib:$CUDA_PATH/lib64:$LD_LIBRARY_PATH
ldd build-trt/jhbr2 | grep 'not found'
```

If TensorRT says the engine is incompatible, rebuild the engine on the same
machine where you will run JHBR2.
