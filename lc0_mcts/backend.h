/*
  JHBR2 Shogi Engine — Backend with Shared GPU Batching

  Workers gather leaves independently, then call ComputeBlocking().

  Two modes:
  1. Solo mode (num_workers=1): Direct GPU call, no batching overhead.
  2. Shared mode (num_workers>1): Workers submit leaves to a shared
     queue. A dedicated GPU thread collects and evaluates combined
     batches. Workers wait for their results via condition variables.

  This lets N workers keep the GPU busy with large batches while
  each worker only gathers a small number of leaves.
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
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
  Computation(Backend* backend, int worker_id)
      : backend_(backend), worker_id_(worker_id) {}

  void AddInput(const ShogiBoard& board, const MoveList& legal_moves) {
    inputs_.emplace_back(board, legal_moves);
  }

  int UsedBatchSize() const { return static_cast<int>(inputs_.size()); }

  // Run GPU inference. In shared mode, submits to batch queue and waits.
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
// Backend — shared GPU batching
// =====================================================================

class Backend {
 public:
  Backend(NNEvaluator& evaluator, int num_workers = 1)
      : evaluator_(evaluator), num_workers_(num_workers) {
    if (num_workers_ > 1) {
      // Shared mode: allocate per-worker result storage and CVs.
      worker_results_.resize(num_workers_);
      worker_ready_.resize(num_workers_, false);
      worker_cvs_ = std::make_unique<std::condition_variable[]>(num_workers_);
    }
  }

  ~Backend() { StopGPUThread(); }

  int num_workers() const { return num_workers_; }

  std::unique_ptr<Computation> CreateComputation(int worker_id = 0) {
    return std::make_unique<Computation>(this, worker_id);
  }

  // Start/stop GPU thread for shared batching mode.
  void StartGPUThread() {
    if (num_workers_ <= 1) return;
    gpu_stop_.store(false);
    gpu_thread_ = std::thread([this]() { GPULoop(); });
  }

  void StopGPUThread() {
    if (!gpu_thread_.joinable()) return;
    gpu_stop_.store(true);
    queue_cv_.notify_one();
    gpu_thread_.join();
    // Wake any waiting workers.
    {
      std::lock_guard<std::mutex> lock(result_mutex_);
      for (int i = 0; i < num_workers_; i++) worker_ready_[i] = true;
    }
    for (int i = 0; i < num_workers_; i++) worker_cvs_[i].notify_all();
  }

  // Direct GPU eval (bypasses shared queue). Used for root eval.
  void EvalDirect(Computation* comp) {
    std::lock_guard<std::mutex> lock(gpu_mutex_);
    comp->results_ = evaluator_.EvaluateBatch(comp->inputs_);
  }

 private:
  friend class Computation;

  // --- Shared mode (N workers) ---

  struct WorkerSubmission {
    int worker_id;
    std::vector<std::pair<ShogiBoard, MoveList>>* inputs;  // borrowed
    int count;
  };

  void SubmitAndWait(Computation* comp) {
    int wid = comp->worker_id_;
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      pending_.push_back({wid, &comp->inputs_,
                          static_cast<int>(comp->inputs_.size())});
    }
    queue_cv_.notify_one();

    // Wait for results.
    {
      std::unique_lock<std::mutex> lock(result_mutex_);
      worker_cvs_[wid].wait(lock, [&] { return worker_ready_[wid]; });
      worker_ready_[wid] = false;
    }
    comp->results_ = std::move(worker_results_[wid]);
  }

  void GPULoop() {
    while (!gpu_stop_.load(std::memory_order_relaxed)) {
      std::vector<WorkerSubmission> submissions;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [&] {
          return !pending_.empty() || gpu_stop_.load(std::memory_order_relaxed);
        });
        if (pending_.empty()) {
          if (gpu_stop_.load()) break;
          continue;
        }

        // Brief wait for more workers to submit (~200μs).
        if (CountPendingLeaves() < 32 && num_workers_ > 1) {
          queue_cv_.wait_for(lock, std::chrono::microseconds(200), [&] {
            return CountPendingLeaves() >= 32 ||
                   gpu_stop_.load(std::memory_order_relaxed);
          });
        }

        submissions = std::move(pending_);
        pending_.clear();
      }

      if (submissions.empty()) continue;

      // Flatten all workers' inputs into one batch.
      std::vector<std::pair<ShogiBoard, MoveList>> combined;
      struct LeafInfo { int worker_id; int leaf_idx; };
      std::vector<LeafInfo> leaf_map;

      for (auto& sub : submissions) {
        for (int i = 0; i < sub.count; i++) {
          combined.push_back(std::move((*sub.inputs)[i]));
          leaf_map.push_back({sub.worker_id, i});
        }
      }

      // ONE GPU call for all workers' leaves.
      auto all_results = evaluator_.EvaluateBatch(combined);

      // Distribute results back to workers.
      std::vector<int> worker_count(num_workers_, 0);
      for (auto& sub : submissions) worker_count[sub.worker_id] = sub.count;

      {
        std::lock_guard<std::mutex> lock(result_mutex_);
        for (int w = 0; w < num_workers_; w++) {
          if (worker_count[w] > 0) worker_results_[w].resize(worker_count[w]);
        }
        for (int i = 0; i < static_cast<int>(leaf_map.size()); i++) {
          int wid = leaf_map[i].worker_id;
          int idx = leaf_map[i].leaf_idx;
          worker_results_[wid][idx] = std::move(all_results[i]);
        }
        for (auto& sub : submissions) worker_ready_[sub.worker_id] = true;
      }
      for (auto& sub : submissions) worker_cvs_[sub.worker_id].notify_one();
    }
  }

  int CountPendingLeaves() const {
    int n = 0;
    for (auto& s : pending_) n += s.count;
    return n;
  }

  NNEvaluator& evaluator_;
  int num_workers_;

  // Solo mode.
  std::mutex gpu_mutex_;

  // Shared mode.
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::vector<WorkerSubmission> pending_;

  std::mutex result_mutex_;
  std::vector<std::vector<NNOutput>> worker_results_;
  std::vector<bool> worker_ready_;
  std::unique_ptr<std::condition_variable[]> worker_cvs_;

  std::atomic<bool> gpu_stop_{false};
  std::thread gpu_thread_;
};

// =====================================================================
// Computation::ComputeBlocking
// =====================================================================

inline void Computation::ComputeBlocking() {
  if (inputs_.empty()) return;
  if (backend_->num_workers_ <= 1) {
    backend_->EvalDirect(this);
  } else {
    backend_->SubmitAndWait(this);
  }
}

}  // namespace lc0_shogi
