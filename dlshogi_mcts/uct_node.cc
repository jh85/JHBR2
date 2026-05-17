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

void uct_node_t::CreateSingleChildNode(lczero::Move move) {
  child_num = 1;
  child = std::make_unique<child_node_t[]>(1);
  child[0].move = move;
  child_nodes = std::make_unique<std::unique_ptr<uct_node_t>[]>(1);
}

uct_node_t* uct_node_t::ReleaseChildrenExceptOne(lczero::Move move) {
  if (child_num <= 0 || !child) {
    CreateSingleChildNode(move);
    child_nodes[0] = std::make_unique<uct_node_t>();
    return child_nodes[0].get();
  }

  InitChildNodes();
  for (int i = 0; i < child_num; ++i) {
    if (child[i].move == move) {
      if (!child_nodes[i]) child_nodes[i] = std::make_unique<uct_node_t>();
      auto kept_child = std::make_unique<child_node_t[]>(1);
      kept_child[0] = std::move(child[i]);
      auto kept_nodes = std::make_unique<std::unique_ptr<uct_node_t>[]>(1);
      kept_nodes[0] = std::move(child_nodes[i]);
      child = std::move(kept_child);
      child_nodes = std::move(kept_nodes);
      child_num = 1;
      return child_nodes[0].get();
    }
  }

  CreateSingleChildNode(move);
  child_nodes[0] = std::make_unique<uct_node_t>();
  return child_nodes[0].get();
}

NodeTree::NodeTree() { DeallocateTree(); }

bool NodeTree::ResetToPosition(uint64_t starting_pos_key,
                               const std::vector<lczero::Move>& moves) {
  const bool same_game =
      gamebegin_node_ && history_starting_pos_key_ == starting_pos_key;
  if (!same_game) {
    DeallocateTree();
  }
  if (!gamebegin_node_) DeallocateTree();
  history_starting_pos_key_ = starting_pos_key;

  uct_node_t* old_head = same_game ? current_head_ : nullptr;
  uct_node_t* prev_head = nullptr;
  current_head_ = gamebegin_node_.get();
  bool seen_old_head = old_head && current_head_ == old_head;
  for (lczero::Move move : moves) {
    // Check before pruning: ReleaseChildrenExceptOne may delete sibling
    // subtrees, so old_head must only be compared while it is still live.
    if (!seen_old_head && old_head && current_head_->child &&
        current_head_->child_nodes) {
      for (int i = 0; i < current_head_->child_num; ++i) {
        if (current_head_->child[i].move == move &&
            current_head_->child_nodes[i].get() == old_head) {
          seen_old_head = true;
          break;
        }
      }
    }
    prev_head = current_head_;
    current_head_ = current_head_->ReleaseChildrenExceptOne(move);
  }

  // Moving backward to a pruned ancestor would leave only the previously
  // played child available from the new root, so restart that head.
  if (old_head && !seen_old_head && current_head_ != old_head) {
    if (prev_head) {
      prev_head->child_nodes[0] = std::make_unique<uct_node_t>();
      current_head_ = prev_head->child_nodes[0].get();
    } else {
      DeallocateTree();
      history_starting_pos_key_ = starting_pos_key;
    }
  }
  return seen_old_head;
}

void NodeTree::DeallocateTree() {
  gamebegin_node_ = std::make_unique<uct_node_t>();
  current_head_ = gamebegin_node_.get();
}

}  // namespace dlshogi_mcts
