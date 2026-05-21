/*
  JHBR2 Shogi Engine — Native TensorRT Backend

  Direct TensorRT C++ API for neural network inference.
  No ONNX Runtime dependency — just TensorRT + CUDA.

  Usage:
    1. Convert ONNX model to TensorRT engine:
       trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 \
         --minShapes=input_planes:1x148x9x9 \
         --optShapes=input_planes:16x148x9x9 \
         --maxShapes=input_planes:32x148x9x9

    2. Build jhbr2 with TensorRT:
       cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON

    3. Run:
       setoption name OnnxModel value /path/to/model.engine
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mcts/model_format.h"
#include "shogi/board.h"
#include "shogi/encoder.h"
#include "shogi/types.h"

namespace jhbr2 {

using lczero::ShogiBoard;
using lczero::Move;
using lczero::MoveList;

// Re-use the same NNOutput struct.
struct NNOutput {
  float value;
  float draw;
  float wdl[3];
  float mlh = 0.0f;
  bool has_mlh = false;
  std::vector<float> policy;
};

class NNEvaluator {
 public:
  // num_slots controls how many independent execution contexts (each
  // with its own CUDA stream and pre-allocated H/D buffers) are
  // created on this GPU. Multiple workers can call EvaluateBatchSlot
  // concurrently — one worker per slot. Slot 0 is always allocated
  // and used by the legacy single-threaded entry points.
  // max_batch_size is forwarded to the ONNX Runtime path only.
  explicit NNEvaluator(const std::string& engine_path, bool use_gpu = true,
                       int device_id = 0, int max_batch_size = 1024,
                       int num_slots = 1,
                       ModelFormat model_format = ModelFormat::kAuto);
  ~NNEvaluator();

  // Single-position / single-threaded helpers (always use slot 0).
  NNOutput Evaluate(const ShogiBoard& board, const MoveList& legal_moves);
  std::vector<NNOutput> EvaluateBatch(
      const std::vector<std::pair<ShogiBoard, MoveList>>& batch);

  // Multi-worker entry: each worker passes its own slot_id (0 ..
  // num_slots-1). Slots are independent — no synchronization needed
  // between concurrent EvaluateBatchSlot calls on different slots.
  std::vector<NNOutput> EvaluateBatchSlot(
      int slot_id,
      const std::vector<std::pair<ShogiBoard, MoveList>>& batch);

  int num_slots() const;
  bool using_gpu() const { return true; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jhbr2
