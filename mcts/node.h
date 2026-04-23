/*
  JHBR2 Shogi Engine — MCTS Node

  Tree node for Monte Carlo Tree Search with PUCT selection.
  Thread-safe for barrier-based multi-threaded search.

  References:
    - lc0 src/mcts/node.h
    - dlshogi Node.h (YaneuraOu/source/engine/dlshogi-engine/Node.h)
*/

#pragma once

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "shogi/types.h"

namespace jhbr2 {

using lczero::Move;

// =====================================================================
// Edge — one move from a parent node, with NN policy prior
// =====================================================================

struct Edge {
  Move move;
  float policy = 0.0f;  // P(s,a) from NN

  // Mate flags (following dlshogi convention).
  enum MateFlag : uint8_t {
    kNone = 0,
    kWin  = 1,  // This move leads to a won position for the parent
    kLose = 2,  // This move leads to a lost position for the parent
    kDraw = 3,  // Repetition draw
  };
  MateFlag mate_flag = kNone;

  bool IsWin() const { return mate_flag == kWin; }
  bool IsLose() const { return mate_flag == kLose; }
  bool IsDraw() const { return mate_flag == kDraw; }

  void SetWin()  { mate_flag = kWin; }
  void SetLose() { mate_flag = kLose; }
  void SetDraw() { mate_flag = kDraw; }
};

// =====================================================================
// Node — a position in the MCTS search tree (thread-safe)
// =====================================================================

class Node {
 public:
  Node() = default;
  ~Node() = default;

  // Non-copyable (tree structure with parent pointers).
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  // --- Tree structure ---

  Node* parent() const { return parent_; }
  void set_parent(Node* p) { parent_ = p; }

  int num_edges() const { return num_edges_.load(std::memory_order_acquire); }
  Edge& edge(int i) { return edges_[i]; }
  const Edge& edge(int i) const { return edges_[i]; }

  // Does this node have children allocated?
  // Acquire ensures edge data written before the release store is visible.
  bool is_expanded() const {
    return num_edges_.load(std::memory_order_acquire) > 0;
  }

  // Get child node for edge i. May be nullptr if not yet created.
  Node* child(int i) const {
    return children_ ? children_[i].get() : nullptr;
  }

  // Create child node for edge i (thread-safe).
  Node* GetOrCreateChild(int i) {
    std::lock_guard<std::mutex> lock(children_mutex_);
    if (!children_) {
      children_ = std::make_unique<std::unique_ptr<Node>[]>(num_edges_);
    }
    if (!children_[i]) {
      children_[i] = std::make_unique<Node>();
      children_[i]->parent_ = this;
      children_[i]->parent_edge_idx_ = i;
    }
    return children_[i].get();
  }

  int parent_edge_idx() const { return parent_edge_idx_; }

  // --- Expand: set edges from legal moves + policy priors ---
  // Called only by the thread that won TryStartExpansion().

  void Expand(const std::vector<std::pair<Move, float>>& move_priors) {
    int n = static_cast<int>(move_priors.size());
    if (n == 0) return;
    edges_ = std::make_unique<Edge[]>(n);
    for (int i = 0; i < n; i++) {
      edges_[i].move = move_priors[i].first;
      edges_[i].policy = move_priors[i].second;
    }
    // Release store — ensures all edge data is visible to threads
    // that load num_edges_ with acquire (is_expanded(), num_edges()).
    num_edges_.store(n, std::memory_order_release);
  }

  // --- MCTS statistics (thread-safe with atomics) ---

  uint32_t n() const { return n_.load(std::memory_order_relaxed); }
  float w() const { return w_.load(std::memory_order_relaxed); }
  float d() const { return d_.load(std::memory_order_relaxed); }

  float q() const {
    uint32_t visits = n_.load(std::memory_order_relaxed);
    return visits > 0 ? w_.load(std::memory_order_relaxed) / visits : 0.0f;
  }

  float d_avg() const {
    uint32_t visits = n_.load(std::memory_order_relaxed);
    return visits > 0 ? d_.load(std::memory_order_relaxed) / visits : 0.0f;
  }

  void AddVisit(float value, float draw) {
    n_.fetch_add(1, std::memory_order_relaxed);
    AtomicFloatAdd(w_, value);
    AtomicFloatAdd(d_, draw);
  }

  // Set initial evaluation (for root node after first NN eval).
  void SetFirstEval(float value, float draw) {
    n_.store(1, std::memory_order_relaxed);
    w_.store(value, std::memory_order_relaxed);
    d_.store(draw, std::memory_order_relaxed);
  }

  // --- Virtual loss (for multi-threaded search, lc0-style) ---
  // VL uses a separate in-flight counter so Q = W/N is never corrupted.
  // Only the PUCT exploration term is affected (via n_started).

  void AddVirtualLoss(int count = 1) {
    n_in_flight_.fetch_add(count, std::memory_order_relaxed);
  }

  void RemoveVirtualLoss(int count = 1) {
    n_in_flight_.fetch_sub(count, std::memory_order_relaxed);
  }

  // N + in-flight — used in PUCT denominator for exploration.
  uint32_t n_started() const {
    return n_.load(std::memory_order_relaxed) +
           n_in_flight_.load(std::memory_order_relaxed);
  }

  // --- Expansion lock (prevents two threads expanding same node) ---

  // Try to claim expansion rights. Returns true if this thread should expand.
  bool TryStartExpansion() {
    bool expected = false;
    return expanding_.compare_exchange_strong(expected, true,
        std::memory_order_acq_rel);
  }

  void FinishExpansion() {
    expanding_.store(false, std::memory_order_release);
  }

  bool is_being_expanded() const {
    return expanding_.load(std::memory_order_acquire);
  }

  // --- Terminal status ---

  bool is_terminal() const { return is_terminal_.load(std::memory_order_relaxed); }
  float terminal_v() const { return terminal_v_; }
  float terminal_d() const { return terminal_d_; }

  void SetTerminal(float v, float d = 0.0f) {
    terminal_v_ = v;
    terminal_d_ = d;
    is_terminal_.store(true, std::memory_order_release);
  }

  // --- df-pn mate status ---
  // 0 = unknown, 1 = side-to-move can force mate (win), -1 = mated (loss)
  int8_t mate_status() const { return mate_status_; }
  void set_mate_status(int8_t s) { mate_status_ = s; }

  bool dfpn_checked() const { return dfpn_checked_; }
  void set_dfpn_checked(bool v) { dfpn_checked_ = v; }

  bool dfpn_proven_no_mate() const { return dfpn_proven_no_mate_; }
  void set_dfpn_proven_no_mate(bool v) { dfpn_proven_no_mate_ = v; }

  // --- NN evaluation status ---
  bool is_evaluated() const { return is_evaluated_; }
  void set_evaluated(bool v) { is_evaluated_ = v; }

 private:
  // Atomic float add via CAS loop
  static void AtomicFloatAdd(std::atomic<float>& target, float value) {
    float current = target.load(std::memory_order_relaxed);
    while (!target.compare_exchange_weak(current, current + value,
        std::memory_order_relaxed, std::memory_order_relaxed)) {}
  }

  // Tree
  Node* parent_ = nullptr;
  int parent_edge_idx_ = -1;
  std::unique_ptr<Edge[]> edges_;
  std::unique_ptr<std::unique_ptr<Node>[]> children_;
  std::mutex children_mutex_;
  std::atomic<int> num_edges_{0};

  // MCTS stats (atomic for thread safety)
  std::atomic<uint32_t> n_{0};
  std::atomic<float> w_{0.0f};
  std::atomic<float> d_{0.0f};
  std::atomic<uint32_t> n_in_flight_{0};  // Virtual loss counter (lc0-style)

  // Expansion lock
  std::atomic<bool> expanding_{false};

  // Terminal
  std::atomic<bool> is_terminal_{false};
  float terminal_v_ = 0.0f;
  float terminal_d_ = 0.0f;

  // Mate solver
  int8_t mate_status_ = 0;
  bool dfpn_checked_ = false;
  bool dfpn_proven_no_mate_ = false;

  // NN
  bool is_evaluated_ = false;
};

// =====================================================================
// MCTS Configuration
// =====================================================================

struct MCTSConfig {
  // PUCT constants (lc0 defaults)
  float cpuct_init = 1.745f;
  float cpuct_base = 38739.0f;
  float cpuct_factor = 3.894f;

  // First Play Urgency
  float fpu_value = 0.330f;
  float fpu_root = 1.0f;

  // Dirichlet noise (for self-play training)
  float noise_epsilon = 0.0f;
  float noise_alpha = 0.15f;

  // Search limits
  int max_nodes = 800;
  float max_time = 0.0f;           // Seconds, 0 = unlimited
  int max_depth = 200;

  // Game limits
  int max_game_moves = 320;

  // Temperature for move selection
  float temperature = 0.0f;
  int temp_moves = 30;

  // Draw score
  float draw_score = 0.0f;

  // df-pn mate search settings
  int leaf_dfpn_nodes = 100;
  int pv_dfpn_nodes = 100000;

  // Threading
  int num_search_threads = 1;       // 1 = single-threaded (backward compatible)
  int virtual_loss_count = 3;       // Virtual visits per selection
  int expand_depth = 1;             // Nodes to expand per simulation (1 = standard)
  int sims_per_thread = 1;          // Independent leaves per thread per barrier phase
  int minibatch_size = 32;           // Leaves per worker per gather round
  int warmup_nodes = 0;             // Nodes to expand in single-threaded warmup (0=disabled)
  int warmup_batch = 256;           // Batch size for warmup GPU evaluation
};

}  // namespace jhbr2
