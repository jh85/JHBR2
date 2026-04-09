/*
  JHBR2 Shogi Engine — MCTS Node

  Tree node for Monte Carlo Tree Search with PUCT selection.
  Designed with df-pn mate solver integration from day one.

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

  // Mate flags encoded in the move (following dlshogi convention).
  // These are separate from the MCTS node's mate_status because
  // an edge can be marked win/lose before the child node is expanded.
  enum MateFlag : uint8_t {
    kNone = 0,
    kWin  = 1,  // This move leads to a won position for the parent
                 // (child is mated / child's side loses)
    kLose = 2,  // This move leads to a lost position for the parent
                 // (child can force mate on us)
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
// Node — a position in the MCTS search tree
// =====================================================================

class Node {
 public:
  Node() = default;
  ~Node() = default;

  // Non-copyable (tree structure with parent pointers).
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  // Move is fine.
  Node(Node&&) = default;
  Node& operator=(Node&&) = default;

  // --- Tree structure ---

  Node* parent() const { return parent_; }
  void set_parent(Node* p) { parent_ = p; }

  int num_edges() const { return num_edges_; }
  Edge& edge(int i) { return edges_[i]; }
  const Edge& edge(int i) const { return edges_[i]; }

  // Does this node have children allocated?
  bool is_expanded() const { return num_edges_ > 0; }

  // Get child node for edge i. May be nullptr if not yet created.
  Node* child(int i) const { return children_ ? children_[i].get() : nullptr; }

  // Create child node for edge i (if not already created).
  Node* GetOrCreateChild(int i) {
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

  void Expand(const std::vector<std::pair<Move, float>>& move_priors) {
    num_edges_ = static_cast<int>(move_priors.size());
    if (num_edges_ == 0) return;
    edges_ = std::make_unique<Edge[]>(num_edges_);
    for (int i = 0; i < num_edges_; i++) {
      edges_[i].move = move_priors[i].first;
      edges_[i].policy = move_priors[i].second;
    }
  }

  // --- MCTS statistics ---

  uint32_t n() const { return n_; }
  float w() const { return w_; }
  float d() const { return d_; }

  float q() const { return n_ > 0 ? w_ / n_ : 0.0f; }
  float d_avg() const { return n_ > 0 ? d_ / n_ : 0.0f; }

  void AddVisit(float value, float draw) {
    n_ += 1;
    w_ += value;
    d_ += draw;
  }

  // Set initial evaluation (for root node after first NN eval).
  void SetFirstEval(float value, float draw) {
    n_ = 1;
    w_ = value;
    d_ = draw;
  }

  // --- Terminal status ---

  bool is_terminal() const { return is_terminal_; }
  float terminal_v() const { return terminal_v_; }
  float terminal_d() const { return terminal_d_; }

  void SetTerminal(float v, float d = 0.0f) {
    is_terminal_ = true;
    terminal_v_ = v;
    terminal_d_ = d;
  }

  // --- df-pn mate status ---
  // 0 = unknown, 1 = side-to-move can force mate (win), -1 = mated (loss)
  int8_t mate_status() const { return mate_status_; }
  void set_mate_status(int8_t s) { mate_status_ = s; }

  // Has df-pn already searched this position?
  bool dfpn_checked() const { return dfpn_checked_; }
  void set_dfpn_checked(bool v) { dfpn_checked_ = v; }

  // Proved by df-pn that no mate exists from this position.
  bool dfpn_proven_no_mate() const { return dfpn_proven_no_mate_; }
  void set_dfpn_proven_no_mate(bool v) { dfpn_proven_no_mate_ = v; }

  // --- NN evaluation status ---
  bool is_evaluated() const { return is_evaluated_; }
  void set_evaluated(bool v) { is_evaluated_ = v; }

 private:
  // Tree
  Node* parent_ = nullptr;
  int parent_edge_idx_ = -1;
  std::unique_ptr<Edge[]> edges_;
  std::unique_ptr<std::unique_ptr<Node>[]> children_;
  int num_edges_ = 0;

  // MCTS stats
  uint32_t n_ = 0;
  float w_ = 0.0f;
  float d_ = 0.0f;

  // Terminal
  bool is_terminal_ = false;
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
  float fpu_value = 0.330f;       // Non-root FPU reduction
  float fpu_root = 1.0f;          // Root FPU reduction

  // Dirichlet noise (for self-play training)
  float noise_epsilon = 0.0f;     // 0 = disabled
  float noise_alpha = 0.15f;      // Shogi: smaller than chess

  // Search limits
  int max_nodes = 800;
  float max_time = 0.0f;          // Seconds, 0 = unlimited
  int max_depth = 200;

  // Game limits
  int max_game_moves = 320;       // Draw after this

  // Temperature for move selection
  float temperature = 0.0f;       // 0 = argmax
  int temp_moves = 30;            // Apply temperature for first N moves

  // Draw score
  float draw_score = 0.0f;

  // df-pn mate search settings
  int leaf_dfpn_nodes = 40;       // Shallow df-pn budget per leaf (0=disable)
  int pv_dfpn_nodes = 100000;     // Deep df-pn for PV search (0=disable)
};

}  // namespace jhbr2
