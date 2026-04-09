/*
  JHBR2 Shogi Engine — Neural Network Evaluator

  ONNX Runtime C++ interface for batched NN inference.
  Uses the encoder from shogi_engine/encoder.h for input/output mapping.

  Reference: lc0 src/neural/onnx/builder.h
*/

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "shogi/board.h"
#include "shogi/encoder.h"
#include "shogi/types.h"

namespace jhbr2 {

using lczero::ShogiBoard;
using lczero::Move;
using lczero::MoveList;

// =====================================================================
// NNOutput — result of evaluating one position
// =====================================================================

struct NNOutput {
  float value;    // W - L from side-to-move perspective
  float draw;     // Draw probability
  float wdl[3];   // [Win, Draw, Loss] probabilities

  // Policy: probability for each legal move.
  // Indexed by position in the legal_moves vector passed to Evaluate().
  std::vector<float> policy;
};

// =====================================================================
// NNEvaluator — wraps ONNX Runtime for model inference
// =====================================================================

class NNEvaluator {
 public:
  // Load model from ONNX file.
  // use_gpu: try CUDA provider first, fall back to CPU.
  explicit NNEvaluator(const std::string& onnx_path, bool use_gpu = true);
  ~NNEvaluator();

  // Evaluate a single position.
  // legal_moves: list of legal moves in the position.
  // Returns NNOutput with value, draw, and per-move policy.
  NNOutput Evaluate(const ShogiBoard& board, const MoveList& legal_moves);

  // Evaluate a batch of positions (more efficient on GPU).
  // Each element: (board, legal_moves).
  std::vector<NNOutput> EvaluateBatch(
      const std::vector<std::pair<ShogiBoard, MoveList>>& batch);

  // Is the evaluator using GPU?
  bool using_gpu() const { return using_gpu_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool using_gpu_ = false;
};

}  // namespace jhbr2
