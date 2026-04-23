/*
  JHBR2 Shogi Engine — Async Multi-Gather MCTS (lc0-style)

  Each worker thread gathers multiple leaves (minibatch), then submits
  them all to a shared GPU queue. A dedicated GPU thread collects
  leaves from all workers, evaluates in large batches, and distributes
  results back. Workers wait for their batch via condition variables.

  With 2-4 workers each gathering 32-64 leaves, the GPU sees batches
  of 64-128 — full utilization with minimal virtual loss.
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "mcts/node.h"
#ifdef USE_TENSORRT
#include "mcts/nn_tensorrt.h"
#else
#include "mcts/nn_eval.h"
#endif
#include "shogi/board.h"

namespace jhbr2 {

using lczero::ShogiBoard;
using lczero::MoveList;

// =====================================================================
// LeafRequest — a single leaf submitted for GPU evaluation
// =====================================================================

struct LeafRequest {
  ShogiBoard board;
  MoveList legal_moves;
  Node* leaf;
  std::vector<Node*> path;
};

// =====================================================================
// WorkerBatch — a batch of leaves from one worker
// =====================================================================

struct WorkerBatch {
  int worker_id;
  std::vector<LeafRequest> leaves;
};

// =====================================================================
// AsyncBatchQueue — multi-gather batch queue
// =====================================================================

class AsyncBatchQueue {
 public:
  AsyncBatchQueue(NNEvaluator& evaluator, int max_batch_size, int num_workers)
      : evaluator_(evaluator), max_batch_size_(max_batch_size),
        num_workers_(num_workers) {
    worker_results_.resize(num_workers);
    worker_ready_.resize(num_workers, false);
    worker_cvs_ = std::make_unique<std::condition_variable[]>(num_workers);
  }

  // Signal the GPU thread to stop (call after all workers have joined).
  void NotifyStop() {
    gpu_stop_.store(true, std::memory_order_relaxed);
    queue_cv_.notify_one();
  }

  // Called by worker: submit a batch of leaves and wait for NN results.
  // Returns one NNOutput per leaf, in the same order.
  std::vector<NNOutput> SubmitBatchAndWait(WorkerBatch& batch) {
    int wid = batch.worker_id;

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      pending_.push_back(std::move(batch));
    }
    queue_cv_.notify_one();

    // Wait for results.
    {
      std::unique_lock<std::mutex> lock(result_mutex_);
      worker_cvs_[wid].wait(lock, [&] { return worker_ready_[wid]; });
      worker_ready_[wid] = false;
    }
    return std::move(worker_results_[wid]);
  }

  // Called by GPU thread: process batches until NotifyStop() is called.
  void GPULoop() {
    while (!gpu_stop_.load(std::memory_order_relaxed)) {
      std::vector<WorkerBatch> batches;

      // Wait for enough leaves to fill a batch, or a short timeout.
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // First wait: block until at least one worker submits or stop.
        queue_cv_.wait(lock, [&] {
          return !pending_.empty() || gpu_stop_.load(std::memory_order_relaxed);
        });

        if (pending_.empty()) {
          if (gpu_stop_.load(std::memory_order_relaxed)) break;
          continue;
        }

        // Second wait: give other workers a chance to submit (up to 500μs)
        // so we can build a full batch and avoid padding waste.
        if (CountPendingLeaves() < max_batch_size_) {
          queue_cv_.wait_for(lock, std::chrono::microseconds(500), [&] {
            return CountPendingLeaves() >= max_batch_size_ ||
                   gpu_stop_.load(std::memory_order_relaxed);
          });
        }

        // Take all pending worker batches.
        batches = std::move(pending_);
        pending_.clear();
      }

      if (batches.empty()) continue;

      // Flatten all leaves into one big NN batch.
      // Track which worker each leaf belongs to and its index within that worker's batch.
      struct LeafInfo {
        int worker_id;
        int leaf_idx;  // index within the worker's batch
      };
      std::vector<std::pair<ShogiBoard, MoveList>> nn_batch;
      std::vector<LeafInfo> leaf_map;

      for (auto& wb : batches) {
        for (int i = 0; i < (int)wb.leaves.size(); i++) {
          nn_batch.emplace_back(std::move(wb.leaves[i].board),
                                std::move(wb.leaves[i].legal_moves));
          leaf_map.push_back({wb.worker_id, i});
        }
      }

      // Evaluate in sub-batches if needed.
      std::vector<NNOutput> all_results;
      int total = (int)nn_batch.size();
      for (int start = 0; start < total; start += max_batch_size_) {
        int end = std::min(start + max_batch_size_, total);
        std::vector<std::pair<ShogiBoard, MoveList>> sub(
            std::make_move_iterator(nn_batch.begin() + start),
            std::make_move_iterator(nn_batch.begin() + end));
        auto sub_results = evaluator_.EvaluateBatch(sub);
        all_results.insert(all_results.end(),
                           std::make_move_iterator(sub_results.begin()),
                           std::make_move_iterator(sub_results.end()));
      }

      // Prepare per-worker result vectors.
      // First, count leaves per worker to pre-allocate.
      std::vector<int> worker_leaf_count(num_workers_, 0);
      for (auto& wb : batches) {
        worker_leaf_count[wb.worker_id] = (int)wb.leaves.size();
      }

      {
        std::lock_guard<std::mutex> lock(result_mutex_);
        for (int w = 0; w < num_workers_; w++) {
          if (worker_leaf_count[w] > 0) {
            worker_results_[w].resize(worker_leaf_count[w]);
          }
        }

        // Distribute results.
        for (int i = 0; i < (int)leaf_map.size(); i++) {
          int wid = leaf_map[i].worker_id;
          int idx = leaf_map[i].leaf_idx;
          worker_results_[wid][idx] = std::move(all_results[i]);
        }

        // Signal all workers that submitted batches.
        for (auto& wb : batches) {
          worker_ready_[wb.worker_id] = true;
        }
      }
      for (auto& wb : batches) {
        worker_cvs_[wb.worker_id].notify_one();
      }
    }

    // Wake all waiting workers on stop.
    {
      std::lock_guard<std::mutex> lock(result_mutex_);
      for (int i = 0; i < num_workers_; i++) {
        worker_ready_[i] = true;
      }
    }
    for (int i = 0; i < num_workers_; i++) {
      worker_cvs_[i].notify_all();
    }
  }

 private:
  NNEvaluator& evaluator_;
  int max_batch_size_;
  int num_workers_;

  // Count total leaves across all pending worker batches.
  // Must be called with queue_mutex_ held.
  int CountPendingLeaves() const {
    int count = 0;
    for (auto& wb : pending_) count += (int)wb.leaves.size();
    return count;
  }

  // Pending worker batches.
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::vector<WorkerBatch> pending_;

  // GPU thread stop flag (set by NotifyStop after workers have joined).
  std::atomic<bool> gpu_stop_{false};

  // Per-worker results.
  std::mutex result_mutex_;
  std::vector<std::vector<NNOutput>> worker_results_;
  std::vector<bool> worker_ready_;
  std::unique_ptr<std::condition_variable[]> worker_cvs_;
};

}  // namespace jhbr2
