/*
  JHBR2 Shogi Engine — Multi-Threaded MCTS Support

  Barrier synchronization, thread context, and batch queue for
  lc0-style parallel MCTS search.
*/

#pragma once

#include <condition_variable>
#include <mutex>
#include <vector>

#include "mcts/node.h"
#include "mcts/nn_eval.h"
#include "shogi/board.h"

namespace jhbr2 {

using lczero::ShogiBoard;
using lczero::MoveList;

// =====================================================================
// SearchBarrier — barrier synchronization for N threads
// =====================================================================

class SearchBarrier {
 public:
  explicit SearchBarrier(int num_threads)
      : num_threads_(num_threads), count_(0), generation_(0) {}

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    int gen = generation_;
    if (++count_ == num_threads_) {
      count_ = 0;
      generation_++;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [&] { return generation_ != gen; });
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int num_threads_;
  int count_;
  int generation_;
};

// =====================================================================
// ThreadContext — per-thread data for one search iteration
// =====================================================================

struct ThreadContext {
  int thread_id = 0;

  // Selection result
  Node* leaf = nullptr;
  ShogiBoard board;
  MoveList legal_moves;

  // Path from root to leaf (for virtual loss tracking)
  std::vector<Node*> path;

  // Classification of the leaf
  enum LeafType {
    kNeedsNN,          // Normal leaf, needs NN evaluation
    kTerminal,         // Terminal node (checkmate, repetition, etc.)
    kCollision,        // Another thread is expanding this node
  };
  LeafType leaf_type = kNeedsNN;

  // Terminal/collision value
  float value = 0.0f;
  float draw = 0.0f;

  // Index into the NN batch results (-1 if not evaluated)
  int batch_index = -1;

  // NN output (set in Phase 2)
  NNOutput nn_output;

  void Reset() {
    leaf = nullptr;
    path.clear();
    leaf_type = kNeedsNN;
    value = 0.0f;
    draw = 0.0f;
    batch_index = -1;
  }
};

// =====================================================================
// BatchQueue — collects leaves for batched NN evaluation
// =====================================================================
// Each thread writes at its own index. Thread 0 builds the batch
// after the barrier. No locking needed — indices are disjoint.

class BatchQueue {
 public:
  void Init(int num_threads) {
    contexts_.resize(num_threads);
  }

  void SetContext(int thread_id, ThreadContext* ctx) {
    contexts_[thread_id] = ctx;
  }

  // Build NN batch from all threads that need evaluation.
  // Returns the batch and assigns batch_index to each context.
  std::vector<std::pair<ShogiBoard, MoveList>> BuildBatch() {
    std::vector<std::pair<ShogiBoard, MoveList>> batch;
    int idx = 0;
    for (auto* ctx : contexts_) {
      ctx->batch_index = -1;
      if (ctx->leaf_type == ThreadContext::kNeedsNN) {
        ctx->batch_index = idx++;
        batch.emplace_back(ctx->board, ctx->legal_moves);
      }
    }
    return batch;
  }

  // Distribute NN results back to each thread's context.
  void DistributeResults(const std::vector<NNOutput>& results) {
    for (auto* ctx : contexts_) {
      if (ctx->batch_index >= 0 && ctx->batch_index < (int)results.size()) {
        ctx->nn_output = results[ctx->batch_index];
      }
    }
  }

 private:
  std::vector<ThreadContext*> contexts_;
};

}  // namespace jhbr2
