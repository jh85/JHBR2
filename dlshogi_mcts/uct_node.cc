#include "dlshogi_mcts/uct_node.h"

#include <utility>

namespace dlshogi_mcts {

void uct_node_t::ExpandNode(const lczero::ShogiBoard* board) {
  auto moves = const_cast<lczero::ShogiBoard*>(board)->GenerateLegalMoves();
  child_num = static_cast<short>(moves.size());
  child = std::make_unique<child_node_t[]>(child_num);
  for (int i = 0; i < child_num; ++i) {
    child[i].move = moves[i];
  }
  InitChildNodes();
}

void uct_node_t::InitChildNodes() {
  if (!child_nodes && child_num > 0) {
    child_nodes = std::make_unique<std::unique_ptr<uct_node_t>[]>(child_num);
  }
}

uct_node_t* uct_node_t::CreateChildNode(int i) {
  InitChildNodes();
  child_nodes[i] = std::make_unique<uct_node_t>();
  return child_nodes[i].get();
}

uct_node_t* uct_node_t::ReleaseChildrenExceptOne(lczero::Move move) {
  InitChildNodes();
  for (int i = 0; i < child_num; ++i) {
    if (child[i].move == move) {
      if (!child_nodes[i]) child_nodes[i] = std::make_unique<uct_node_t>();
      auto kept = std::move(child_nodes[i]);
      child_nodes.reset();
      return (child_nodes = std::make_unique<std::unique_ptr<uct_node_t>[]>(1),
              child_nodes[0] = std::move(kept), child_nodes[0].get());
    }
  }
  child_nodes.reset();
  return nullptr;
}

NodeTree::NodeTree() { DeallocateTree(); }

bool NodeTree::ResetToPosition(uint64_t starting_pos_key,
                               const std::vector<lczero::Move>& moves) {
  if (!gamebegin_node_ || history_starting_pos_key_ != starting_pos_key) {
    DeallocateTree();
    history_starting_pos_key_ = starting_pos_key;
    return false;
  }

  uct_node_t* node = gamebegin_node_.get();
  for (lczero::Move move : moves) {
    if (!node->child || !node->child_nodes) {
      current_head_ = node;
      return false;
    }
    int found = -1;
    for (int i = 0; i < node->child_num; ++i) {
      if (node->child[i].move == move) {
        found = i;
        break;
      }
    }
    if (found < 0 || !node->child_nodes[found]) {
      current_head_ = node;
      return false;
    }
    node = node->child_nodes[found].get();
  }
  current_head_ = node;
  return true;
}

void NodeTree::DeallocateTree() {
  gamebegin_node_ = std::make_unique<uct_node_t>();
  current_head_ = gamebegin_node_.get();
}

}  // namespace dlshogi_mcts
