/*
  JHBR2 Shogi Engine — Native TensorRT Backend

  Direct TensorRT C++ API for neural network inference.
  No ONNX Runtime dependency — just TensorRT + CUDA.

  Usage:
    1. Convert ONNX model to TensorRT engine:
       trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 \
         --minShapes=input_planes:1x48x9x9 \
         --optShapes=input_planes:16x48x9x9 \
         --maxShapes=input_planes:32x48x9x9

    2. Build jhbr2 with TensorRT:
       cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON

    3. Run:
       setoption name OnnxModel value /path/to/model.engine
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

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
  std::vector<float> policy;
};

class NNEvaluator {
 public:
  explicit NNEvaluator(const std::string& engine_path, bool use_gpu = true,
                       int device_id = 0);
  ~NNEvaluator();

  NNOutput Evaluate(const ShogiBoard& board, const MoveList& legal_moves);

  std::vector<NNOutput> EvaluateBatch(
      const std::vector<std::pair<ShogiBoard, MoveList>>& batch);

  bool using_gpu() const { return true; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jhbr2
