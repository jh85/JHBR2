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

#include "mcts/model_format.h"
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
  // num_slots is accepted for signature parity with the direct-TRT
  // path but ignored here — ORT sessions are themselves single-threaded.
  explicit NNEvaluator(const std::string& onnx_path, bool use_gpu = true,
                       int device_id = 0, int max_batch_size = 1024,
                       int num_slots = 1,
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

  // Slot-aware variant; for the ORT path slot_id is ignored (and a
  // mutex serializes calls). Provided only for signature compatibility
  // with the direct-TRT path.
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
  bool supports_batch_ = false;  // True if ONNX model handles batch > 1
};

}  // namespace jhbr2
