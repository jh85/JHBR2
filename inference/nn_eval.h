/*
  JHBR2 Shogi Engine — Neural Network Evaluator

  ONNX Runtime C++ interface for batched NN inference.
  Uses the encoder from shogi/encoder.h for input/output mapping.

  Reference: lc0 src/neural/onnx/builder.h
*/

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "inference/model_format.h"
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
  float value = 0.0f;    // W - L from side-to-move perspective
  float draw = 0.0f;     // Draw probability
  float wdl[3] = {0.0f, 0.0f, 0.0f};  // [Win, Draw, Loss] probabilities
  float moves_left = 0.0f;  // MLH head: model's plies-to-end estimate (0 if none)
  bool valid = true;  // false results must not be cached or backed up

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
  // use_gpu: create a CUDA session first, then fall back to CPU.
  // The model must have one dynamic-batch input shaped (batch, C, 9, 9).
  // num_slots is accepted for signature parity and ignored; one ORT session
  // is shared by all workers.
  explicit NNEvaluator(const std::string& onnx_path, bool use_gpu = true,
                       int device_id = 0, int num_slots = 1,
                       ModelFormat model_format = ModelFormat::kAuto);
  ~NNEvaluator();

  // Evaluate a single position.
  // legal_moves: list of legal moves in the position.
  // Returns NNOutput with value, draw, and per-move policy.
  NNOutput Evaluate(const ShogiBoard& board, const MoveList& legal_moves);

  // Evaluate a batch of positions (more efficient on GPU).
  // Each element: (board, legal_moves).
  std::vector<NNOutput> EvaluateBatch(
      const std::vector<std::pair<ShogiBoard, MoveList>>& batch);

  // Slot-aware variant provided for signature compatibility. ORT sessions
  // support concurrent Run calls, so slot_id is not needed here.
  std::vector<NNOutput> EvaluateBatchSlot(
      int slot_id,
      const std::vector<std::pair<ShogiBoard, MoveList>>& batch) {
    (void)slot_id;
    return EvaluateBatch(batch);
  }

  int num_slots() const { return 1; }

  // Is the evaluator using GPU?
  bool using_gpu() const { return using_gpu_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool using_gpu_ = false;
};

}  // namespace jhbr2
