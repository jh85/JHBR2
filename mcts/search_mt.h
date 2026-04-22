/*
  JHBR2 Shogi Engine — Multi-Threaded MCTS Support

  Barrier synchronization, thread context, and batch queue.
  Supports multi-leaf (K sims per thread) and multi-expand (D depths).
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
// SearchBarrier
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
// SimContext — one independent simulation (leaf selection + path)
// =====================================================================

struct SimContext {
  enum LeafType { kNeedsNN, kTerminal, kCollision };

  Node* leaf = nullptr;
  ShogiBoard board;
  MoveList legal_moves;
  LeafType leaf_type = kNeedsNN;
  float value = 0.0f;
  float draw = 0.0f;
  int batch_index = -1;
  NNOutput nn_output;

  // Full path from root through all expansion depths.
  std::vector<Node*> path;

  // Per-depth expansion records.
  struct ExpandRecord {
    int path_length;    // path.size() at this expansion point
    float value;
    float draw;
    LeafType leaf_type;
  };
  std::vector<ExpandRecord> expand_records;

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
// ThreadContext — per-thread data holding K simulations
// =====================================================================

struct ThreadContext {
  int thread_id = 0;
  std::vector<SimContext> sims;  // K independent simulations

  void Init(int num_sims) {
    sims.resize(num_sims);
  }

  void ResetAll() {
    for (auto& s : sims) s.Reset();
  }
};

// =====================================================================
// BatchQueue — collects leaves from all threads × all sims
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
      for (auto& sim : ctx->sims) {
        sim.batch_index = -1;
        if (sim.leaf_type == SimContext::kNeedsNN && sim.can_continue) {
          sim.batch_index = idx++;
          batch.emplace_back(sim.board, sim.legal_moves);
        }
      }
    }
    return batch;
  }

  void DistributeResults(const std::vector<NNOutput>& results) {
    for (auto* ctx : contexts_) {
      for (auto& sim : ctx->sims) {
        if (sim.batch_index >= 0 && sim.batch_index < (int)results.size()) {
          sim.nn_output = results[sim.batch_index];
        }
      }
    }
  }

 private:
  std::vector<ThreadContext*> contexts_;
};

}  // namespace jhbr2
