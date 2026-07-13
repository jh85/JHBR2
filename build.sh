#!/bin/bash

export CUDA_PATH=/usr/local/cuda
export TENSORRT_PATH=/data/new_jhbr2/TensorRT-11.1.0.106
export CUDNN_PATH=/data/new_jhbr2/cudnn-linux-x86_64-9.24.0.43_cuda13-archive
export MODEL_ONNX=/data/new_jhbr2/JHBR2/shogi_bt4_epoch1_dynamic_fp16.onnx
export CC=clang
export CXX=clang++
export BUILD_DIR=build-trt

export LD_LIBRARY_PATH=$TENSORRT_PATH/lib:$CUDNN_PATH/lib:$CUDA_PATH/lib64:$LD_LIBRARY_PATH


rm -rf "$BUILD_DIR"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON \
  -DCUDAToolkit_ROOT="$CUDA_PATH" \
  -DTENSORRT_ROOT="$TENSORRT_PATH" \
  -DCUDNN_ROOT="$CUDNN_PATH"

cmake --build "$BUILD_DIR" -j"$(nproc)"

./"$BUILD_DIR"/test_movegen test/positions.txt
./"$BUILD_DIR"/test_check_movegen
./"$BUILD_DIR"/test_shallow_mate

mkdir -p engines

$TENSORRT_PATH/bin/trtexec \
  --onnx="$MODEL_ONNX" \
  --saveEngine=engines/shogi_bt4_epoch23_trt_o128_m128_ws8192.engine \
  --minShapes=input_planes:1x148x9x9 \
  --optShapes=input_planes:128x148x9x9 \
  --maxShapes=input_planes:128x148x9x9 \
  --memPoolSize=workspace:8192M
