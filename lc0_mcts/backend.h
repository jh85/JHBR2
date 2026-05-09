/*
  JHBR2 Shogi Engine — Per-Worker GPU Backend (dlshogi-style)

  Each worker is pinned to one (evaluator, slot). Workers submit
  their own batch directly to the GPU via the slot's CUDA stream;
  no central dispatcher combines submissions across workers.

  Concurrency on the same GPU is achieved by allocating multiple
  slots (= execution contexts + streams) per evaluator. Workers on
  different slots can enqueue work concurrently on the same engine.

  NN cache lookup happens on the calling worker's thread.
*/

#pragma once

#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#ifdef USE_TENSORRT
#include "mcts/nn_tensorrt.h"
#else
#include "mcts/nn_eval.h"
#endif
#include "mcts/nn_cache.h"
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
  Computation(Backend* backend, int worker_id)
      : backend_(backend), worker_id_(worker_id) {}

  void AddInput(const ShogiBoard& board, const MoveList& legal_moves) {
    inputs_.emplace_back(board, legal_moves);
  }

  int UsedBatchSize() const { return static_cast<int>(inputs_.size()); }

  // Run NN evaluation on this worker's assigned (evaluator, slot).
  void ComputeBlocking();

  float GetQ(int idx) const { return results_[idx].value; }
  float GetD(int idx) const { return results_[idx].draw; }
  float GetM(int idx) const { return 0.0f; }
  const std::vector<float>& GetPolicy(int idx) const {
    return results_[idx].policy;
  }

 private:
  friend class Backend;
  Backend* backend_;
  int worker_id_;
  std::vector<std::pair<ShogiBoard, MoveList>> inputs_;
  std::vector<NNOutput> results_;
};

// =====================================================================
// Backend — owns evaluators, routes worker_id → (evaluator, slot)
// =====================================================================

class Backend {
 public:
  // num_workers must equal num_gpus * workers_per_gpu. Worker i is
  // pinned to evaluator (i / workers_per_gpu) and slot
  // (i % workers_per_gpu). Each evaluator must have been constructed
  // with at least workers_per_gpu slots.
  Backend(std::vector<NNEvaluator*> evaluators, int num_workers,
          int workers_per_gpu, size_t nn_cache_size = 0)
      : evaluators_(std::move(evaluators)),
        num_workers_(num_workers),
        workers_per_gpu_(workers_per_gpu > 0 ? workers_per_gpu : 1),
        nn_cache_(nn_cache_size) {}

  ~Backend() = default;

  jhbr2::NNCache& nn_cache() { return nn_cache_; }
  int num_workers() const { return num_workers_; }
  int workers_per_gpu() const { return workers_per_gpu_; }

  std::unique_ptr<Computation> CreateComputation(int worker_id = 0) {
    return std::make_unique<Computation>(this, worker_id);
  }

  // Cache-aware batch eval. Filters out cache hits, only sends misses
  // to the GPU. After GPU returns, inserts the new evaluations into
  // the cache. evaluator_idx selects the GPU; slot_id selects the
  // execution slot on that GPU.
  std::vector<NNOutput> EvalBatchWithCache(
      const std::vector<std::pair<ShogiBoard, MoveList>>& batch,
      int evaluator_idx, int slot_id) {
    const size_t N = batch.size();
    std::vector<NNOutput> results(N);
    if (N == 0) return results;

    std::vector<size_t> miss_indices;
    std::vector<std::pair<ShogiBoard, MoveList>> miss_batch;
    std::vector<uint64_t> miss_keys;
    miss_indices.reserve(N);
    miss_batch.reserve(N);
    miss_keys.reserve(N);

    for (size_t i = 0; i < N; ++i) {
      const ShogiBoard& board = batch[i].first;
      const MoveList& legal = batch[i].second;
      const uint64_t key = board.Hash();
      jhbr2::CachedNNValue cached;
      if (nn_cache_.Lookup(key, static_cast<uint16_t>(legal.size()), &cached)) {
        results[i].wdl[0] = cached.wdl[0];
        results[i].wdl[1] = cached.wdl[1];
        results[i].wdl[2] = cached.wdl[2];
        results[i].value  = cached.wdl[0] - cached.wdl[2];
        results[i].draw   = cached.wdl[1];
        results[i].policy = cached.policy;
      } else {
        miss_indices.push_back(i);
        miss_batch.emplace_back(board, legal);
        miss_keys.push_back(key);
      }
    }

    if (!miss_batch.empty()) {
      auto miss_results =
          evaluators_[evaluator_idx]->EvaluateBatchSlot(slot_id, miss_batch);
      for (size_t k = 0; k < miss_results.size(); ++k) {
        size_t i = miss_indices[k];
        results[i] = std::move(miss_results[k]);

        jhbr2::CachedNNValue to_cache;
        to_cache.wdl[0] = results[i].wdl[0];
        to_cache.wdl[1] = results[i].wdl[1];
        to_cache.wdl[2] = results[i].wdl[2];
        to_cache.policy = results[i].policy;
        to_cache.num_legal_moves =
            static_cast<uint16_t>(batch[i].second.size());
        nn_cache_.Insert(miss_keys[k], std::move(to_cache));
      }
    }

    return results;
  }

  // Map worker_id → (evaluator_idx, slot_id) per the pinning scheme.
  std::pair<int, int> RouteWorker(int worker_id) const {
    int eval_idx = worker_id / workers_per_gpu_;
    int slot_id  = worker_id % workers_per_gpu_;
    if (eval_idx >= static_cast<int>(evaluators_.size())) eval_idx = 0;
    return {eval_idx, slot_id};
  }

 private:
  friend class Computation;

  std::vector<NNEvaluator*> evaluators_;  // One per GPU
  int num_workers_;
  int workers_per_gpu_;
  jhbr2::NNCache nn_cache_;
};

// =====================================================================
// Computation::ComputeBlocking — direct per-worker submission
// =====================================================================

inline void Computation::ComputeBlocking() {
  if (inputs_.empty()) return;
  auto [eval_idx, slot_id] = backend_->RouteWorker(worker_id_);
  results_ = backend_->EvalBatchWithCache(inputs_, eval_idx, slot_id);
}

}  // namespace lc0_shogi
