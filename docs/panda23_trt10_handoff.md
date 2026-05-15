# Panda23 TensorRT 10 Handoff

This note is for continuing the JHBR2 dlshogi-style MCTS validation on
Panda23 with 2x RTX 3090.

## Current State

The clean dlshogi-style MCTS implementation is in:

- `dlshogi_mcts/types.h`
- `dlshogi_mcts/uct_node.{h,cc}`
- `dlshogi_mcts/uct_search.{h,cc}`

USI is currently wired to `dlshogi_mcts::Search` through
`usi/usi_engine.{h,cc}`. CMake also has explicit TensorRT/cuDNN roots:

- `TENSORRT_ROOT`
- `CUDNN_ROOT`

Local non-Panda23 validation passed:

- `cmake --build build -j$(nproc)`
- `./build/test_movegen test/positions.txt`
- `./build/test_check_movegen`
- `./build/test_shallow_mate`

Local TensorRT 10 validation also worked, but only on one visible RTX 2060,
so throughput numbers from this machine are not meaningful for Panda23.

## Important Caveat

`dlshogi_mcts::Search::Run()` currently resets the tree per `go`.

That means persistent subtree reuse across `go` commands is not yet active.
This is acceptable for NPS benchmarking, because each benchmark position is a
fresh search. It is not yet the full long-game behavior expected from a
production dlshogi-style engine. If strength or match testing is next, tree
reuse should be implemented by preserving `NodeTree` across `Run()` calls and
calling `NodeTree::ResetToPosition(...)` from USI position history rather than
unconditionally deallocating the tree.

## Expected Panda23 Layout

Copied `/home/ei/Downloads/dlsport2` to Panda23 as /data/dlsport2.
The following paths are assumed:

```bash
/data/dlsport2/JHBR2
/data/dlsport2/TensorRT-10.16.1.11
/data/dlsport2/cudnn-linux-x86_64-9.22.0.52_cuda13-archive
/data/dlsport2/shogi_bt4_epoch23_dynamic.onnx
/usr/local/cuda (this is cuda 13.2)
```

First verify Panda23 sees both GPUs:

```bash
nvidia-smi
```

Expected: two RTX 3090 GPUs.

## Build JHBR2 With CUDA 13 + TensorRT 10

From `JHBR2`:

```bash
cd /data/dlsport2/JHBR2

cmake -S . -B build-trt10 -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DTENSORRT_ROOT=/data/dlsport2/TensorRT-10.16.1.11 \
  -DCUDNN_ROOT=/data/dlsport2/cudnn-linux-x86_64-9.22.0.52_cuda13-archive

cmake --build build-trt10 -j$(nproc)
```

Check linkage:

```bash
ldd build-trt10/jhbr2 | grep -E 'nvinfer|cudart|not found'
```

The `nvinfer` path should resolve under:

```bash
/data/dlsport2/TensorRT-10.16.1.11/lib
```

## Build A TensorRT 10 Engine On Panda23

Do not reuse an engine built on another machine. TensorRT engines are tied to
GPU architecture, TensorRT version, and driver/runtime details.

Recommended first profile for 3090:

```bash
cd /data/dlsport2/JHBR2
mkdir -p engines

LD_LIBRARY_PATH=/data/dlsport2/TensorRT-10.16.1.11/lib:/data/dlsport2/cudnn-linux-x86_64-9.22.0.52_cuda13-archive/lib:/usr/local/cuda/lib64 \
/data/dlsport2/TensorRT-10.16.1.11/bin/trtexec \
  --onnx=/data/dlsport2/shogi_bt4_epoch23_dynamic.onnx \
  --saveEngine=engines/shogi_bt4_epoch23_trt10_b256.engine \
  --fp16 \
  --minShapes=input_planes:1x48x9x9 \
  --optShapes=input_planes:128x48x9x9 \
  --maxShapes=input_planes:256x48x9x9 \
  --memPoolSize=workspace:4096M
```

If memory is tight or build fails, retry with `maxShapes=128` and
`optShapes=64`. If GPU utilization is low in search, try `maxShapes=512` and
larger minibatches.

## USI Smoke Test

```bash
cd /data/dlsport2/JHBR2

printf 'usi\nsetoption name OnnxModel value /data/dlsport2/JHBR2/engines/shogi_bt4_epoch23_trt10_b256.engine\nsetoption name NumGPUs value 2\nsetoption name WorkersPerGpu value 2\nsetoption name MinibatchSize value 128\nsetoption name MaxGpuBatch value 256\nisready\nposition startpos\ngo nodes 256\nquit\n' \
| LD_LIBRARY_PATH=/data/dlsport2/TensorRT-10.16.1.11/lib:/data/dlsport2/cudnn-linux-x86_64-9.22.0.52_cuda13-archive/lib:/usr/local/cuda/lib64 \
  ./build-trt10/jhbr2
```

Expected:

- `readyok`
- TensorRT logs showing one engine loaded on GPU 0 and one on GPU 1
- a legal `bestmove`

## Benchmark Command

Start with:

```bash
cd /data/dlsport2/JHBR2

LD_LIBRARY_PATH=/data/dlsport2/TensorRT-10.16.1.11/lib:/data/dlsport2/cudnn-linux-x86_64-9.22.0.52_cuda13-archive/lib:/usr/local/cuda/lib64 \
python3 tools/benchmark.py ./build-trt10/jhbr2 \
  /data/dlsport2/JHBR2/engines/shogi_bt4_epoch23_trt10_b256.engine \
  --threads 2 --gpus 2 --minibatch 128 --max-gpu-batch 256 \
  --byoyomi 1000 --leaf-mate-mode off
```

Then sweep:

```bash
--threads 1 --gpus 2 --minibatch 128 --max-gpu-batch 256
--threads 2 --gpus 2 --minibatch 64  --max-gpu-batch 256
--threads 2 --gpus 2 --minibatch 128 --max-gpu-batch 256
--threads 2 --gpus 2 --minibatch 256 --max-gpu-batch 256
```

Use `nvidia-smi dmon` or `nvidia-smi pmon` during benchmark to check both
GPUs are active.

## Current Target

The implementation goal is approximately 60k+ median NPS on Panda23 with
2x 3090, matching the original motivation for moving away from the previous
mechanical port.

## Panda23 Validation Results - 2026-05-15

Panda23 has the expected two RTX 3090 GPUs and CUDA 13.2 visible via
`nvidia-smi`.

The copied `build-trt10` CMake cache was created under
`/home/ei/Downloads/dlsport2`, so CMake refused to reuse it after the tree was
copied to `/data/dlsport2`. The stale directory was moved aside to
`build-trt10.copied-home-cache`, and `build-trt10` was regenerated with the
Panda23-local roots:

```bash
cmake -S . -B build-trt10 -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DTENSORRT_ROOT=/data/dlsport2/TensorRT-10.16.1.11 \
  -DCUDNN_ROOT=/data/dlsport2/cudnn-linux-x86_64-9.22.0.52_cuda13-archive
cmake --build build-trt10 -j$(nproc)
```

`ldd build-trt10/jhbr2` now resolves `libnvinfer.so.10` from:

```bash
/data/dlsport2/TensorRT-10.16.1.11/lib/libnvinfer.so.10
```

Local correctness checks passed:

```bash
./build-trt10/test_movegen test/positions.txt
./build-trt10/test_check_movegen
./build-trt10/test_shallow_mate
```

A fresh Panda23 TensorRT engine was built:

```bash
engines/shogi_bt4_epoch23_trt10_b256.engine
```

The USI smoke test passed with `NumGPUs=2`, `WorkersPerGpu=2`,
`MinibatchSize=128`, and `MaxGpuBatch=256`: both GPU 0 and GPU 1 loaded the
engine, `readyok` was returned, and `go nodes 256` produced a legal
`bestmove`.

Benchmark with the handoff baseline:

```bash
LD_LIBRARY_PATH=/data/dlsport2/TensorRT-10.16.1.11/lib:/data/dlsport2/cudnn-linux-x86_64-9.22.0.52_cuda13-archive/lib:/usr/local/cuda/lib64 \
python3 tools/benchmark.py ./build-trt10/jhbr2 \
  /data/dlsport2/JHBR2/engines/shogi_bt4_epoch23_trt10_b256.engine \
  --threads 2 --gpus 2 --minibatch 128 --max-gpu-batch 256 \
  --byoyomi 1000 --leaf-mate-mode off
```

Result:

```text
Positions completed : 100/100 (failed: 0)
Total bench time    : 95.8s
NPS mean            : 8,201
NPS median          : 8,248
NPS p10/p90         : 7,112 / 8,838
Nodes total         : 768,427
```

`nvidia-smi dmon` showed both GPUs active during the benchmark, often at high
SM utilization. The low NPS is therefore not caused by only one GPU being used.

Raw TensorRT throughput explains the observed search NPS:

```text
batch 64  : 16.58 ms median, 60.30 batches/s, about 3.86k samples/s/GPU
batch 128 : 29.24 ms median, 33.97 batches/s, about 4.35k samples/s/GPU
batch 256 : 57.68 ms median, 17.10 batches/s, about 4.38k samples/s/GPU
```

With two GPUs, the practical NN-eval ceiling for this engine is about
8.7k samples/s, which matches the 8.2k median MCTS NPS. Two TensorRT inference
streams on one GPU gave only a small increase, about 35.5 batches/s at batch
128. A `--builderOptimizationLevel=5` rebuild took 446 seconds and did not
improve throughput.

TensorRT profiling shows the time is spent in the model itself: the engine has
transformer-style `encoders.0` through `encoders.19` MatMul/attention blocks and
totals about 29.5 ms per batch-128 inference on one RTX 3090. Host/device copy
time is only about 0.26 ms per batch.

Conclusion: the current Panda23 result validates the build, two-GPU loading,
USI integration, and dlshogi-style MCTS wiring, but the 60k NPS target is not
reachable with this ONNX/TensorRT engine on 2x RTX 3090. The bottleneck is raw
model inference throughput, not the search loop or missing GPU utilization.

If observed NPS is far below target:

1. Verify both GPUs are loaded.
2. Verify each GPU gets the intended number of worker slots.
3. Check TensorRT engine batch profile and actual `MinibatchSize`.
4. Try `leaf-mate-mode off` first to isolate pure MCTS/NN throughput.
5. Compare `trtexec` batch latency for `input_planes=128x48x9x9` and
   `256x48x9x9`.
6. Inspect `dlshogi_mcts/uct_search.cc` for search loop bottlenecks before
   changing the algorithm.

## Files Changed In This Session

- `.gitignore`
- `CMakeLists.txt`
- `usi/usi_engine.h`
- `usi/usi_engine.cc`
- `dlshogi_mcts/`

Build artifacts and engines are ignored by git:

- `build-trt10/`
- `engines/*.engine`
