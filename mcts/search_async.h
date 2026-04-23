/*
  JHBR2 Shogi Engine — Async Queue MCTS (lc0-style)

  Workers submit leaves to a shared batch queue. A dedicated GPU thread
  collects batches and evaluates them. Workers wait for their specific
  result via condition variables.

  Supports multi-GPU: one GPU thread per GPU, workers assigned to a GPU.
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
// LeafRequest — a leaf submitted by a worker for GPU evaluation
// =====================================================================

struct LeafRequest {
  int worker_id;
  ShogiBoard board;
  MoveList legal_moves;
  Node* leaf;
  std::vector<Node*> path;  // for virtual loss removal + backprop
};

// =====================================================================
// AsyncBatchQueue — collects leaves and triggers GPU evaluation
// =====================================================================

class AsyncBatchQueue {
 public:
  AsyncBatchQueue(NNEvaluator& evaluator, int batch_size, int num_workers)
      : evaluator_(evaluator), batch_size_(batch_size), num_workers_(num_workers) {
    worker_results_.resize(num_workers);
    worker_ready_.resize(num_workers, false);
    worker_cvs_ = std::make_unique<std::condition_variable[]>(num_workers);
  }

  // Called by worker: submit a leaf and wait for the NN result.
  NNOutput SubmitAndWait(LeafRequest& req) {
    int wid = req.worker_id;

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      pending_.push_back(std::move(req));
    }
    // Notify GPU thread that a new leaf is available.
    queue_cv_.notify_one();

    // Wait for this worker's result.
    {
      std::unique_lock<std::mutex> lock(result_mutex_);
      worker_cvs_[wid].wait(lock, [&] { return worker_ready_[wid]; });
      worker_ready_[wid] = false;
    }
    return std::move(worker_results_[wid]);
  }

  // Called by GPU thread: process batches until stopped.
  void GPULoop(std::atomic<bool>& stop) {
    while (!stop.load(std::memory_order_relaxed)) {
      std::vector<LeafRequest> batch;

      // Wait for enough leaves or timeout.
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait_for(lock, std::chrono::milliseconds(1), [&] {
          return (int)pending_.size() >= batch_size_ ||
                 stop.load(std::memory_order_relaxed);
        });

        if (pending_.empty()) continue;

        // Take all pending leaves (up to batch_size).
        int take = std::min((int)pending_.size(), batch_size_);
        batch.assign(
            std::make_move_iterator(pending_.begin()),
            std::make_move_iterator(pending_.begin() + take));
        pending_.erase(pending_.begin(), pending_.begin() + take);
      }

      if (batch.empty()) continue;

      // Build NN batch.
      std::vector<std::pair<ShogiBoard, MoveList>> nn_batch;
      nn_batch.reserve(batch.size());
      for (auto& req : batch) {
        nn_batch.emplace_back(std::move(req.board), std::move(req.legal_moves));
      }

      // Evaluate.
      auto results = evaluator_.EvaluateBatch(nn_batch);

      // Distribute results to waiting workers.
      {
        std::lock_guard<std::mutex> lock(result_mutex_);
        for (int i = 0; i < (int)batch.size(); i++) {
          int wid = batch[i].worker_id;
          worker_results_[wid] = std::move(results[i]);
          worker_ready_[wid] = true;
        }
      }
      for (int i = 0; i < (int)batch.size(); i++) {
        worker_cvs_[batch[i].worker_id].notify_one();
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
  int batch_size_;
  int num_workers_;

  // Pending leaves queue.
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::vector<LeafRequest> pending_;

  // Per-worker results.
  std::mutex result_mutex_;
  std::vector<NNOutput> worker_results_;
  std::vector<bool> worker_ready_;
  std::unique_ptr<std::condition_variable[]> worker_cvs_;
};

}  // namespace jhbr2
