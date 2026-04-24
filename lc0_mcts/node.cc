/*
  JHBR2 Shogi Engine — lc0 MCTS Node (ported from lc0)

  Original: lc0/src/search/classic/node.cc
  Copyright (C) 2018 The LCZero Authors (GPL v3)

  Changes from lc0:
    - Replaced chess PositionHistory/GameState with ShogiBoard
    - Replaced lc0 Mutex with std::mutex
    - Removed chess-specific NodeTree::ResetToPosition(fen, moves)
    - Kept all MCTS logic (GC, VL, edges, solid children) unchanged
*/

#include "lc0_mcts/node.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace lc0_shogi {

/////////////////////////////////////////////////////////////////////////
// Node garbage collector (identical to lc0)
/////////////////////////////////////////////////////////////////////////

namespace {
const int kGCIntervalMs = 100;

class NodeGarbageCollector {
 public:
  NodeGarbageCollector() : gc_thread_([this]() { Worker(); }) {}

  void AddToGcQueue(std::unique_ptr<Node> node, size_t solid_size = 0) {
    if (!node) return;
    std::lock_guard<std::mutex> lock(gc_mutex_);
    subtrees_to_gc_.emplace_back(std::move(node));
    subtrees_to_gc_solid_size_.push_back(solid_size);
  }

  ~NodeGarbageCollector() {
    stop_.store(true);
    gc_thread_.join();
  }

 private:
  void GarbageCollect() {
    while (!stop_.load()) {
      std::unique_ptr<Node> node_to_gc;
      size_t solid_size = 0;
      {
        std::lock_guard<std::mutex> lock(gc_mutex_);
        if (subtrees_to_gc_.empty()) return;
        node_to_gc = std::move(subtrees_to_gc_.back());
        subtrees_to_gc_.pop_back();
        solid_size = subtrees_to_gc_solid_size_.back();
        subtrees_to_gc_solid_size_.pop_back();
      }
      if (solid_size != 0) {
        for (size_t i = 0; i < solid_size; i++) {
          node_to_gc.get()[i].~Node();
        }
        std::allocator<Node> alloc;
        alloc.deallocate(node_to_gc.release(), solid_size);
      }
    }
  }

  void Worker() {
    while (!stop_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
      GarbageCollect();
    }
  }

  mutable std::mutex gc_mutex_;
  std::vector<std::unique_ptr<Node>> subtrees_to_gc_;
  std::vector<size_t> subtrees_to_gc_solid_size_;
  std::atomic<bool> stop_{false};
  std::thread gc_thread_;
};

NodeGarbageCollector gNodeGc;
}  // namespace

/////////////////////////////////////////////////////////////////////////
// Edge (identical to lc0)
/////////////////////////////////////////////////////////////////////////

Move Edge::GetMove(bool as_opponent) const {
  if (!as_opponent) return move_;
  Move m = move_;
  m.Flip();
  return m;
}

void Edge::SetP(float p) {
  assert(0.0f <= p && p <= 1.0f);
  constexpr int32_t roundings = (1 << 11) - (3 << 28);
  int32_t tmp;
  std::memcpy(&tmp, &p, sizeof(float));
  tmp += roundings;
  p_ = (tmp < 0) ? 0 : static_cast<uint16_t>(tmp >> 12);
}

float Edge::GetP() const {
  uint32_t tmp = (static_cast<uint32_t>(p_) << 12) | (3 << 28);
  float ret;
  std::memcpy(&ret, &tmp, sizeof(uint32_t));
  return ret;
}

std::string Edge::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move_.ToString() << " p_: " << p_
      << " GetP: " << GetP();
  return oss.str();
}

std::unique_ptr<Edge[]> Edge::FromMovelist(const MoveList& moves) {
  std::unique_ptr<Edge[]> edges = std::make_unique<Edge[]>(moves.size());
  auto* edge = edges.get();
  for (size_t i = 0; i < moves.size(); i++) edge[i].move_ = moves[i];
  return edges;
}

/////////////////////////////////////////////////////////////////////////
// Node (identical to lc0 except debug output)
/////////////////////////////////////////////////////////////////////////

Node* Node::CreateSingleChildNode(Move move) {
  assert(!edges_);
  assert(!child_);
  MoveList ml;
  ml.push_back(move);
  edges_ = Edge::FromMovelist(ml);
  num_edges_ = 1;
  child_ = std::make_unique<Node>(this, 0);
  return child_.get();
}

void Node::CreateEdges(const MoveList& moves) {
  assert(!edges_);
  assert(!child_);
  // Cap at 255 edges (uint8_t). Top policy moves cover >99.99% of probability.
  // Shogi can have 500+ legal moves but PUCT never selects low-policy moves.
  int n = std::min(static_cast<int>(moves.size()), 255);
  MoveList capped;
  for (int i = 0; i < n; i++) capped.push_back(moves[i]);
  edges_ = Edge::FromMovelist(capped);
  num_edges_ = static_cast<uint8_t>(n);
}

Node::ConstIterator Node::Edges() const {
  return {*this, !solid_children_ ? &child_ : nullptr};
}
Node::Iterator Node::Edges() {
  return {*this, !solid_children_ ? &child_ : nullptr};
}

float Node::GetVisitedPolicy() const {
  float sum = 0.0f;
  for (auto* node : VisitedNodes()) sum += GetEdgeToNode(node)->GetP();
  return sum;
}

Edge* Node::GetEdgeToNode(const Node* node) const {
  assert(node->parent_ == this);
  assert(node->index_ < num_edges_);
  return &edges_[node->index_];
}

Edge* Node::GetOwnEdge() const { return GetParent()->GetEdgeToNode(this); }

std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << " Term:" << static_cast<int>(terminal_type_) << " This:" << this
      << " Parent:" << parent_ << " Index:" << index_
      << " Child:" << child_.get() << " Sibling:" << sibling_.get()
      << " WL:" << wl_ << " N:" << n_ << " NIF:" << n_in_flight_
      << " Edges:" << static_cast<int>(num_edges_) << " Solid:" << solid_children_;
  return oss.str();
}

bool Node::MakeSolid() {
  if (solid_children_ || num_edges_ == 0 || IsTerminal()) return false;
  Node* old_child_to_check = child_.get();
  uint32_t total_in_flight = 0;
  while (old_child_to_check != nullptr) {
    if (old_child_to_check->GetN() <= 1 &&
        old_child_to_check->GetNInFlight() > 0) {
      return false;
    }
    if (old_child_to_check->IsTerminal() &&
        old_child_to_check->GetNInFlight() > 0) {
      return false;
    }
    total_in_flight += old_child_to_check->GetNInFlight();
    old_child_to_check = old_child_to_check->sibling_.get();
  }
  if (total_in_flight != GetNInFlight()) return false;

  std::allocator<Node> alloc;
  auto* new_children = alloc.allocate(num_edges_);
  for (int i = 0; i < num_edges_; i++) {
    new (&(new_children[i])) Node(this, i);
  }
  std::unique_ptr<Node> old_child = std::move(child_);
  while (old_child) {
    int index = old_child->index_;
    new_children[index] = std::move(*old_child.get());
    old_child->parent_ = nullptr;
    gNodeGc.AddToGcQueue(std::move(old_child));
    new_children[index].UpdateChildrenParents();
    old_child = std::move(new_children[index].sibling_);
  }
  child_ = std::unique_ptr<Node>(new_children);
  solid_children_ = true;
  return true;
}

void Node::SortEdges() {
  assert(edges_);
  assert(!child_);
  std::sort(edges_.get(), (edges_.get() + num_edges_),
            [](const Edge& a, const Edge& b) { return a.p_ > b.p_; });
}

void Node::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  if (type != Terminal::TwoFold) SetBounds(result, result);
  terminal_type_ = type;
  m_ = plies_left;
  if (result == GameResult::DRAW) {
    wl_ = 0.0f;
    d_ = 1.0f;
  } else if (result == GameResult::WHITE_WON) {
    wl_ = 1.0f;
    d_ = 0.0f;
  } else if (result == GameResult::BLACK_WON) {
    wl_ = -1.0f;
    d_ = 0.0f;
    if (GetParent() != nullptr) GetOwnEdge()->SetP(0.0f);
  }
}

void Node::MakeNotTerminal() {
  terminal_type_ = Terminal::NonTerminal;
  n_ = 0;
  if (edges_) {
    n_++;
    for (const auto& child : Edges()) {
      const auto n = child.GetN();
      if (n > 0) {
        n_ += n;
        wl_ += -child.GetWL(0.0f) * n;
        d_ += child.GetD(0.0f) * n;
      }
    }
    wl_ /= n_;
    d_ /= n_;
  }
}

void Node::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}

bool Node::TryStartScoreUpdate() {
  if (n_ == 0 && n_in_flight_ > 0) return false;
  ++n_in_flight_;
  return true;
}

void Node::CancelScoreUpdate(int multivisit) { n_in_flight_ -= multivisit; }

void Node::FinalizeScoreUpdate(float v, float d, float m, int multivisit) {
  wl_ += multivisit * (v - wl_) / (n_ + multivisit);
  d_ += multivisit * (d - d_) / (n_ + multivisit);
  m_ += multivisit * (m - m_) / (n_ + multivisit);
  n_ += multivisit;
  n_in_flight_ -= multivisit;
}

void Node::AdjustForTerminal(float v, float d, float m, int multivisit) {
  wl_ += multivisit * v / n_;
  d_ += multivisit * d / n_;
  m_ += multivisit * m / n_;
}

void Node::RevertTerminalVisits(float v, float d, float m, int multivisit) {
  const int n_new = n_ - multivisit;
  if (n_new <= 0) {
    wl_ = 0.0;
    d_ = 1.0;
    m_ = 0.0;
    n_ = 0;
  } else {
    wl_ -= multivisit * (v - wl_) / n_new;
    d_ -= multivisit * (d - d_) / n_new;
    m_ -= multivisit * (m - m_) / n_new;
    n_ -= multivisit;
  }
}

void Node::UpdateChildrenParents() {
  if (!solid_children_) {
    Node* cur_child = child_.get();
    while (cur_child != nullptr) {
      cur_child->parent_ = this;
      cur_child = cur_child->sibling_.get();
    }
  } else {
    Node* child_array = child_.get();
    for (int i = 0; i < num_edges_; i++) {
      child_array[i].parent_ = this;
    }
  }
}

void Node::ReleaseChildren() {
  gNodeGc.AddToGcQueue(std::move(child_), solid_children_ ? num_edges_ : 0);
}

void Node::ReleaseChildrenExceptOne(Node* node_to_save) {
  if (solid_children_) {
    std::unique_ptr<Node> saved_node;
    if (node_to_save != nullptr) {
      saved_node = std::make_unique<Node>(this, node_to_save->index_);
      *saved_node = std::move(*node_to_save);
    }
    gNodeGc.AddToGcQueue(std::move(child_), num_edges_);
    child_ = std::move(saved_node);
    if (child_) child_->UpdateChildrenParents();
    solid_children_ = false;
  } else {
    std::unique_ptr<Node> saved_node;
    for (std::unique_ptr<Node>* node = &child_; *node;
         node = &(*node)->sibling_) {
      if (node->get() == node_to_save) {
        gNodeGc.AddToGcQueue(std::move((*node)->sibling_));
        saved_node = std::move(*node);
        break;
      }
    }
    gNodeGc.AddToGcQueue(std::move(child_));
    child_ = std::move(saved_node);
  }
  if (!child_) {
    num_edges_ = 0;
    edges_.reset();
  }
}

/////////////////////////////////////////////////////////////////////////
// EdgeAndNode
/////////////////////////////////////////////////////////////////////////

std::string EdgeAndNode::DebugString() const {
  if (!edge_) return "(no edge)";
  return edge_->DebugString() + " " +
         (node_ ? node_->DebugString() : "(no node)");
}

/////////////////////////////////////////////////////////////////////////
// NodeTree — simplified for shogi (no PositionHistory/GameState)
/////////////////////////////////////////////////////////////////////////

void NodeTree::MakeMove(Move move) {
  Node* new_head = nullptr;
  for (auto& n : current_head_->Edges()) {
    if (n.GetMove() == move) {
      new_head = n.GetOrSpawnNode(current_head_);
      if (new_head->IsTerminal()) new_head->MakeNotTerminal();
      break;
    }
  }
  current_head_->ReleaseChildrenExceptOne(new_head);
  new_head = current_head_->child_.get();
  current_head_ =
      new_head ? new_head : current_head_->CreateSingleChildNode(move);
  // Apply move to the head board.
  head_board_.DoMove(move);
  ply_count_++;
}

void NodeTree::TrimTreeAtHead() {
  auto tmp = std::move(current_head_->sibling_);
  current_head_->ReleaseChildren();
  *current_head_ = Node(current_head_->GetParent(), current_head_->index_);
  current_head_->sibling_ = std::move(tmp);
}

bool NodeTree::ResetToPosition(const ShogiBoard& board,
                               const std::vector<Move>& moves) {
  // For simplicity, always rebuild the tree.
  // TODO: implement tree reuse by comparing positions.
  DeallocateTree();

  gamebegin_node_ = std::make_unique<Node>(nullptr, 0);
  head_board_ = board;
  ply_count_ = 0;

  current_head_ = gamebegin_node_.get();
  for (const Move& m : moves) {
    MakeMove(m);
  }

  return false;  // No tree reuse for now.
}

void NodeTree::DeallocateTree() {
  gNodeGc.AddToGcQueue(std::move(gamebegin_node_));
  gamebegin_node_ = nullptr;
  current_head_ = nullptr;
}

void Node::UpdateMaxDepth(int depth) {
  // Placeholder — used by lc0 for stats but not critical.
}

bool Node::UpdateFullDepth(uint16_t* depth) {
  // Placeholder.
  return false;
}

}  // namespace lc0_shogi
