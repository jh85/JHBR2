/*
  JHBR2 Shogi Engine — Multi-Threaded MCTS Support

  Barrier synchronization, thread context, and batch queue for
  lc0-style parallel MCTS search with multi-expand support.
*/

#pragma once

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
// Supports multi-expand: each thread can hold multiple leaves
// from successive depths within one simulation.

struct ThreadContext {
  int thread_id = 0;

  // Current leaf (set during each select phase)
  Node* leaf = nullptr;
  ShogiBoard board;
  MoveList legal_moves;

  // Classification of the current leaf
  enum LeafType {
    kNeedsNN,
    kTerminal,
    kCollision,
  };
  LeafType leaf_type = kNeedsNN;

  // Terminal/collision value
  float value = 0.0f;
  float draw = 0.0f;

  // NN batch index (-1 if not evaluated)
  int batch_index = -1;

  // NN output (set in Phase 2)
  NNOutput nn_output;

  // --- Multi-expand tracking ---
  // Full path from root through all expansion depths (virtual loss applied here).
  std::vector<Node*> path;

  // Per-depth expansion records (for multi-expand backpropagation).
  struct ExpandRecord {
    int path_index;      // Index into path[] where this expansion's leaf is
    float value;         // NN value (or terminal value)
    float draw;          // Draw probability
    LeafType leaf_type;  // What happened at this depth
  };
  std::vector<ExpandRecord> expand_records;

  // Whether this thread should continue deeper (set after each expand phase).
  bool can_continue = true;

  void Reset() {
    leaf = nullptr;
    path.clear();
    expand_records.clear();
    leaf_type = kNeedsNN;
    value = 0.0f;
    draw = 0.0f;
    batch_index = -1;
    can_continue = true;
  }

  void ResetForNextDepth() {
    leaf = nullptr;
    leaf_type = kNeedsNN;
    value = 0.0f;
    draw = 0.0f;
    batch_index = -1;
  }
};

// =====================================================================
// BatchQueue — collects leaves for batched NN evaluation
// =====================================================================

class BatchQueue {
 public:
  void Init(int num_threads) {
    contexts_.resize(num_threads);
  }

  void SetContext(int thread_id, ThreadContext* ctx) {
    contexts_[thread_id] = ctx;
  }

  std::vector<std::pair<ShogiBoard, MoveList>> BuildBatch() {
    std::vector<std::pair<ShogiBoard, MoveList>> batch;
    int idx = 0;
    for (auto* ctx : contexts_) {
      ctx->batch_index = -1;
      if (ctx->leaf_type == ThreadContext::kNeedsNN && ctx->can_continue) {
        ctx->batch_index = idx++;
        batch.emplace_back(ctx->board, ctx->legal_moves);
      }
    }
    return batch;
  }

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
