/*
  JHBR2 Shogi Engine — TensorRT Backend (lc0-style)

  Each worker creates its own Computation with separate buffers.
  Workers accumulate inputs independently, then call ComputeBlocking()
  which serializes GPU access internally via a shared mutex.

  Key benefit over raw NNEvaluator:
  - Workers can prepare inputs concurrently (no lock during encoding)
  - GPU serialization is internal (no external mutex in search code)
  - Future: can combine batches from multiple workers into one GPU call
*/

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#ifdef USE_TENSORRT
#include "mcts/nn_tensorrt.h"
#else
#include "mcts/nn_eval.h"
#endif
#include "shogi/board.h"
#include "shogi/encoder.h"

namespace lc0_shogi {

using lczero::ShogiBoard;
using lczero::MoveList;
using jhbr2::NNEvaluator;
using jhbr2::NNOutput;

class Backend;

// =====================================================================
// Computation — per-worker, accumulates inputs then evaluates
// =====================================================================

class Computation {
 public:
  Computation(Backend* backend) : backend_(backend) {}

  // Add a position to evaluate. Can be called without any lock.
  void AddInput(const ShogiBoard& board, const MoveList& legal_moves) {
    inputs_.emplace_back(board, legal_moves);
  }

  int UsedBatchSize() const { return static_cast<int>(inputs_.size()); }

  // Run GPU inference. Internally serialized — safe to call from any thread.
  void ComputeBlocking();

  // Access results after ComputeBlocking().
  float GetQ(int idx) const { return results_[idx].value; }
  float GetD(int idx) const { return results_[idx].draw; }
  float GetM(int idx) const { return 0.0f; }  // No MLH head
  const std::vector<float>& GetPolicy(int idx) const {
    return results_[idx].policy;
  }

 private:
  Backend* backend_;
  std::vector<std::pair<ShogiBoard, MoveList>> inputs_;
  std::vector<NNOutput> results_;
};

// =====================================================================
// Backend — owns the TensorRT evaluator and serialization mutex
// =====================================================================

class Backend {
 public:
  explicit Backend(NNEvaluator& evaluator) : evaluator_(evaluator) {}

  // Create a fresh computation (per worker, per iteration).
  std::unique_ptr<Computation> CreateComputation() {
    return std::make_unique<Computation>(this);
  }

 private:
  friend class Computation;
  NNEvaluator& evaluator_;
  std::mutex gpu_mutex_;  // Serializes GPU access across workers
};

// =====================================================================
// Computation::ComputeBlocking implementation
// =====================================================================

inline void Computation::ComputeBlocking() {
  if (inputs_.empty()) return;
  std::lock_guard<std::mutex> lock(backend_->gpu_mutex_);
  results_ = backend_->evaluator_.EvaluateBatch(inputs_);
}

}  // namespace lc0_shogi
