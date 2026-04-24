/*
  JHBR2 Shogi Engine — lc0 MCTS Node (ported from lc0)

  Original: lc0/src/search/classic/node.h
  Copyright (C) 2018 The LCZero Authors (GPL v3)

  Ported to shogi by replacing chess types with shogi adapter types.
  The MCTS logic (VL, PUCT, backup) is unchanged from lc0.
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

#include "lc0_mcts/types.h"

namespace lc0_shogi {

class Node;

// =====================================================================
// Edge — one move from a parent node, with NN policy prior
// =====================================================================

class Edge {
 public:
  static std::unique_ptr<Edge[]> FromMovelist(const MoveList& moves);

  // Returns move. If as_opponent, flips perspective.
  Move GetMove(bool as_opponent = false) const;

  // Policy prior from NN, compressed to 16-bit. Must be in [0,1].
  float GetP() const;
  void SetP(float val);

  std::string DebugString() const;

 private:
  Move move_;
  uint16_t p_ = 0;  // Compressed policy (5 bits exp, 11 bits significand)
  friend class Node;
};

// =====================================================================
// EdgeAndNode — convenience pair for iteration
// =====================================================================

class EdgeAndNode;
template <bool is_const>
class Edge_Iterator;
template <bool is_const>
class VisitedNode_Iterator;

class Node {
 public:
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;

  enum class Terminal : uint8_t {
    NonTerminal,
    EndOfGame,
    Tablebase,  // Not used for shogi, kept for compatibility
    TwoFold
  };

  Node(Node* parent, uint16_t index)
      : parent_(parent),
        index_(index),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        solid_children_(false) {}

  Node(Node&& move_from) = default;
  Node& operator=(Node&& move_from) = default;

  // Allocates a new edge and a new node.
  Node* CreateSingleChildNode(Move m);

  // Creates edges from a movelist.
  void CreateEdges(const MoveList& moves);

  Node* GetParent() const { return parent_; }
  bool HasChildren() const { return static_cast<bool>(edges_); }

  float GetVisitedPolicy() const;
  uint32_t GetN() const { return n_; }
  uint32_t GetNInFlight() const { return n_in_flight_; }
  uint32_t GetChildrenVisits() const { return n_ > 0 ? n_ - 1 : 0; }
  int GetNStarted() const {
    uint64_t started = static_cast<uint64_t>(n_) + n_in_flight_;
    constexpr uint64_t kMaxSafeStarted =
        static_cast<uint64_t>(std::numeric_limits<int>::max() / 4);
    return static_cast<int>(std::min(started, kMaxSafeStarted));
  }
  float GetQ(float draw_score) const { return wl_ + draw_score * d_; }
  float GetWL() const { return wl_; }
  float GetD() const { return d_; }
  float GetM() const { return m_; }

  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  bool IsTwoFoldTerminal() const { return terminal_type_ == Terminal::TwoFold; }
  typedef std::pair<GameResult, GameResult> Bounds;
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  uint16_t GetNumEdges() const { return num_edges_; }

  void CopyPolicy(int max_needed, float* output) const {
    if (!edges_) return;
    int loops = std::min(static_cast<int>(num_edges_), max_needed);
    for (int i = 0; i < loops; i++) output[i] = edges_[i].GetP();
  }

  void MakeTerminal(GameResult result, float plies_left = 0.0f,
                    Terminal type = Terminal::EndOfGame);
  void MakeNotTerminal();
  void SetBounds(GameResult lower, GameResult upper);

  // Virtual loss / score update (lc0-style).
  bool TryStartScoreUpdate();
  void CancelScoreUpdate(int multivisit);
  void FinalizeScoreUpdate(float v, float d, float m, int multivisit);
  void AdjustForTerminal(float v, float d, float m, int multivisit);
  void RevertTerminalVisits(float v, float d, float m, int multivisit);
  void IncrementNInFlight(int multivisit) { n_in_flight_ += multivisit; }

  void UpdateMaxDepth(int depth);
  bool UpdateFullDepth(uint16_t* depth);

  ConstIterator Edges() const;
  Iterator Edges();

  VisitedNode_Iterator<true> VisitedNodes() const;
  VisitedNode_Iterator<false> VisitedNodes();

  void ReleaseChildren();
  void ReleaseChildrenExceptOne(Node* node);

  Edge* GetEdgeToNode(const Node* node) const;
  Edge* GetOwnEdge() const;

  std::string DebugString() const;

  bool MakeSolid();
  void SortEdges();

  uint16_t Index() const { return index_; }

  ~Node() {
    if (solid_children_ && child_) {
      for (int i = 0; i < num_edges_; i++) {
        child_.get()[i].~Node();
      }
      std::allocator<Node> alloc;
      alloc.deallocate(child_.release(), num_edges_);
    }
  }

 private:
  void UpdateChildrenParents();

  // Layout optimized for cache (same as lc0).
  double wl_ = 0.0f;
  std::unique_ptr<Edge[]> edges_;
  Node* parent_ = nullptr;
  std::unique_ptr<Node> child_;
  std::unique_ptr<Node> sibling_;
  float d_ = 0.0f;
  float m_ = 0.0f;
  uint32_t n_ = 0;
  uint32_t n_in_flight_ = 0;
  uint16_t index_;
  uint16_t num_edges_ = 0;  // Shogi can have 500+ legal moves (chess only ~218)
  Terminal terminal_type_ : 2;
  GameResult lower_bound_ : 2;
  GameResult upper_bound_ : 2;
  bool solid_children_ : 1;

  friend class NodeTree;
  friend class Edge_Iterator<true>;
  friend class Edge_Iterator<false>;
  friend class Edge;
  friend class VisitedNode_Iterator<true>;
  friend class VisitedNode_Iterator<false>;
};

static_assert(sizeof(Node) == 64, "Node layout must stay lc0-compatible");

// =====================================================================
// EdgeAndNode — convenience pair
// =====================================================================

class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : edge_(edge), node_(node) {}
  void Reset() { edge_ = nullptr; }
  explicit operator bool() const { return edge_ != nullptr; }
  bool operator==(const EdgeAndNode& other) const { return edge_ == other.edge_; }
  bool operator!=(const EdgeAndNode& other) const { return edge_ != other.edge_; }
  bool HasNode() const { return node_ != nullptr; }
  Edge* edge() const { return edge_; }
  Node* node() const { return node_; }

  float GetQ(float default_q, float draw_score) const {
    return (node_ && node_->GetN() > 0) ? node_->GetQ(draw_score) : default_q;
  }
  float GetWL(float default_wl) const {
    return (node_ && node_->GetN() > 0) ? node_->GetWL() : default_wl;
  }
  float GetD(float default_d) const {
    return (node_ && node_->GetN() > 0) ? node_->GetD() : default_d;
  }
  float GetM(float default_m) const {
    return (node_ && node_->GetN() > 0) ? node_->GetM() : default_m;
  }
  uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
  int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
  uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }
  bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }
  bool IsTbTerminal() const { return node_ ? node_->IsTbTerminal() : false; }
  Node::Bounds GetBounds() const {
    return node_ ? node_->GetBounds()
                 : Node::Bounds{GameResult::BLACK_WON, GameResult::WHITE_WON};
  }
  float GetP() const { return edge_->GetP(); }
  Move GetMove(bool flip = false) const {
    return edge_ ? edge_->GetMove(flip) : Move();
  }
  float GetU(float numerator) const {
    return numerator * GetP() / (1 + GetNStarted());
  }

  std::string DebugString() const;

 protected:
  Edge* edge_ = nullptr;
  Node* node_ = nullptr;
};

// =====================================================================
// Edge_Iterator — iterates over edges/children
// =====================================================================

template <bool is_const>
class Edge_Iterator : public EdgeAndNode {
 public:
  using Ptr = std::conditional_t<is_const, const std::unique_ptr<Node>*,
                                 std::unique_ptr<Node>*>;
  using value_type = Edge_Iterator;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = Edge_Iterator*;
  using reference = Edge_Iterator&;

  Edge_Iterator() {}

  Edge_Iterator(const Node& parent_node, Ptr child_ptr)
      : EdgeAndNode(parent_node.edges_.get(), nullptr),
        node_ptr_(child_ptr),
        total_count_(parent_node.num_edges_) {
    if (edge_ && child_ptr != nullptr) Actualize();
    if (edge_ && child_ptr == nullptr) {
      node_ = parent_node.child_.get();
    }
  }

  Edge_Iterator<is_const> begin() { return *this; }
  Edge_Iterator<is_const> end() { return {}; }

  void operator++() {
    if (++current_idx_ == total_count_) {
      edge_ = nullptr;
    } else {
      ++edge_;
      if (node_ptr_ != nullptr) {
        Actualize();
      } else {
        ++node_;
      }
    }
  }
  Edge_Iterator& operator*() { return *this; }

  Node* GetOrSpawnNode(Node* parent) {
    if (node_) return node_;
    assert(node_ptr_ != nullptr);
    Actualize();
    if (node_) return node_;
    std::unique_ptr<Node> tmp = std::move(*node_ptr_);
    *node_ptr_ = std::make_unique<Node>(parent, current_idx_);
    (*node_ptr_)->sibling_ = std::move(tmp);
    Actualize();
    return node_;
  }

 private:
  void Actualize() {
    assert(node_ptr_ != nullptr);
    while (*node_ptr_ && (*node_ptr_)->index_ < current_idx_) {
      node_ptr_ = &(*node_ptr_)->sibling_;
    }
    if (*node_ptr_ && (*node_ptr_)->index_ == current_idx_) {
      node_ = (*node_ptr_).get();
      node_ptr_ = &node_->sibling_;
    } else {
      node_ = nullptr;
    }
  }

  Ptr node_ptr_;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
};

// =====================================================================
// VisitedNode_Iterator — iterates over children with N > 0
// =====================================================================

template <bool is_const>
class VisitedNode_Iterator {
 public:
  VisitedNode_Iterator() {}

  VisitedNode_Iterator(const Node& parent_node, Node* child_ptr)
      : node_ptr_(child_ptr),
        total_count_(parent_node.num_edges_),
        solid_(parent_node.solid_children_) {
    if (node_ptr_ != nullptr && node_ptr_->GetN() == 0) {
      operator++();
    }
  }

  bool operator==(const VisitedNode_Iterator<is_const>& other) const {
    return node_ptr_ == other.node_ptr_;
  }
  bool operator!=(const VisitedNode_Iterator<is_const>& other) const {
    return node_ptr_ != other.node_ptr_;
  }

  VisitedNode_Iterator<is_const> begin() { return *this; }
  VisitedNode_Iterator<is_const> end() { return {}; }

  void operator++() {
    if (solid_) {
      while (++current_idx_ != total_count_ &&
             node_ptr_[current_idx_].GetN() == 0) {
        if (node_ptr_[current_idx_].GetNInFlight() == 0) {
          current_idx_ = total_count_;
          break;
        }
      }
      if (current_idx_ == total_count_) node_ptr_ = nullptr;
    } else {
      do {
        node_ptr_ = node_ptr_->sibling_.get();
        if (node_ptr_ != nullptr && node_ptr_->GetN() == 0 &&
            node_ptr_->GetNInFlight() == 0) {
          node_ptr_ = nullptr;
          break;
        }
      } while (node_ptr_ != nullptr && node_ptr_->GetN() == 0);
    }
  }

  Node* operator*() {
    if (solid_) return &(node_ptr_[current_idx_]);
    return node_ptr_;
  }

 private:
  Node* node_ptr_ = nullptr;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
  bool solid_ = false;
};

inline VisitedNode_Iterator<true> Node::VisitedNodes() const {
  return {*this, child_.get()};
}
inline VisitedNode_Iterator<false> Node::VisitedNodes() {
  return {*this, child_.get()};
}

// =====================================================================
// NodeTree — manages the search tree with tree reuse
// =====================================================================

class NodeTree {
 public:
  ~NodeTree() { DeallocateTree(); }

  void MakeMove(Move move);
  void TrimTreeAtHead();

  // Reset to a position. Returns true if tree was reused.
  bool ResetToPosition(const ShogiBoard& board,
                       const std::vector<Move>& moves);

  const ShogiBoard& GetHeadBoard() const { return head_board_; }
  int GetPlyCount() const { return ply_count_; }
  Node* GetCurrentHead() const { return current_head_; }
  Node* GetGameBeginNode() const { return gamebegin_node_.get(); }

 private:
  void DeallocateTree();

  Node* current_head_ = nullptr;
  std::unique_ptr<Node> gamebegin_node_;
  ShogiBoard head_board_;
  int ply_count_ = 0;
};

}  // namespace lc0_shogi
