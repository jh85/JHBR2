#include "dlshogi_mcts/uct_node.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

namespace dlshogi_mcts {
namespace {

// Releasing a large discarded branch can otherwise consume the next move's
// time budget. Ownership is transferred here before the new search starts.
class NodeGarbageCollector {
 public:
  NodeGarbageCollector() : worker_([this] { Run(); }) {}

  ~NodeGarbageCollector() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_one();
    worker_.join();
  }

  void DeleteLater(std::unique_ptr<uct_node_t> node) {
    if (!node) return;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_.push_back(std::move(node));
    }
    ready_.notify_one();
  }

 private:
  void Run() {
    while (true) {
      std::unique_ptr<uct_node_t> node;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
        if (pending_.empty()) {
          if (stopping_) return;
          continue;
        }
        node = std::move(pending_.front());
        pending_.pop_front();
      }
      node.reset();
    }
  }

  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::unique_ptr<uct_node_t>> pending_;
  bool stopping_ = false;
  std::thread worker_;
};

NodeGarbageCollector& GarbageCollector() {
  static NodeGarbageCollector collector;
  return collector;
}

void DeleteSubtreeLater(std::unique_ptr<uct_node_t> node) {
  if (node) GarbageCollector().DeleteLater(std::move(node));
}

}  // namespace

child_node_slot_t::~child_node_slot_t() {
  delete node_.load(std::memory_order_relaxed);
}

uct_node_t* child_node_slot_t::get() const noexcept {
  return node_.load(std::memory_order_acquire);
}

uct_node_t* child_node_slot_t::GetOrCreate() {
  if (uct_node_t* existing = get()) return existing;

  auto candidate = std::make_unique<uct_node_t>();
  uct_node_t* expected = nullptr;
  if (node_.compare_exchange_strong(expected, candidate.get(),
                                    std::memory_order_acq_rel,
                                    std::memory_order_acquire)) {
    return candidate.release();
  }
  return expected;
}

std::unique_ptr<uct_node_t> child_node_slot_t::Take() noexcept {
  return std::unique_ptr<uct_node_t>(
      node_.exchange(nullptr, std::memory_order_acq_rel));
}

std::unique_ptr<uct_node_t> child_node_slot_t::Reset(
    std::unique_ptr<uct_node_t> replacement) noexcept {
  return std::unique_ptr<uct_node_t>(
      node_.exchange(replacement.release(), std::memory_order_acq_rel));
}

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
    child_nodes = std::make_unique<child_node_slot_t[]>(child_num);
  }
}

uct_node_t* uct_node_t::CreateChildNode(int i) {
  InitChildNodes();
  return child_nodes[i].GetOrCreate();
}

void uct_node_t::CreateSingleChildNode(lczero::Move move) {
  child_num = 1;
  child = std::make_unique<child_node_t[]>(1);
  child[0].move = move;
  child_nodes = std::make_unique<child_node_slot_t[]>(1);
}

uct_node_t* uct_node_t::ReleaseChildrenExceptOne(lczero::Move move) {
  if (child_num <= 0 || !child) {
    CreateSingleChildNode(move);
    return child_nodes[0].GetOrCreate();
  }

  InitChildNodes();
  if (child_num == 1 && child[0].move == move) {
    return child_nodes[0].GetOrCreate();
  }

  for (int i = 0; i < child_num; ++i) {
    if (child[i].move == move) {
      uct_node_t* selected = child_nodes[i].GetOrCreate();
      auto kept_child = std::make_unique<child_node_t[]>(1);
      kept_child[0] = std::move(child[i]);
      auto kept_nodes = std::make_unique<child_node_slot_t[]>(1);
      kept_nodes[0].Reset(child_nodes[i].Take());
      for (int sibling = 0; sibling < child_num; ++sibling) {
        if (sibling != i) {
          DeleteSubtreeLater(child_nodes[sibling].Take());
        }
      }
      child = std::move(kept_child);
      child_nodes = std::move(kept_nodes);
      child_num = 1;
      return selected;
    }
  }

  for (int i = 0; i < child_num; ++i) {
    DeleteSubtreeLater(child_nodes[i].Take());
  }
  CreateSingleChildNode(move);
  return child_nodes[0].GetOrCreate();
}

NodeTree::NodeTree() { DeallocateTree(); }

NodeTree::~NodeTree() { DeleteSubtreeLater(std::move(gamebegin_node_)); }

bool NodeTree::ResetToPosition(uint64_t starting_pos_key,
                               const std::vector<lczero::Move>& moves) {
  const bool same_game =
      has_position_ && history_starting_pos_key_ == starting_pos_key;
  const bool can_reuse =
      same_game && current_position_moves_.size() <= moves.size() &&
      std::equal(current_position_moves_.begin(),
                 current_position_moves_.end(), moves.begin());
  if (!same_game) {
    DeallocateTree();
  }

  uct_node_t* prev_head = nullptr;
  current_head_ = gamebegin_node_.get();
  for (lczero::Move move : moves) {
    prev_head = current_head_;
    current_head_ = current_head_->ReleaseChildrenExceptOne(move);
  }

  // Moving backward to a pruned ancestor would leave only the previously
  // played child available from the new root, so restart that head.
  if (same_game && !can_reuse) {
    if (prev_head) {
      DeleteSubtreeLater(prev_head->child_nodes[0].Take());
      auto fresh = std::make_unique<uct_node_t>();
      current_head_ = fresh.get();
      prev_head->child_nodes[0].Reset(std::move(fresh));
    } else {
      DeallocateTree();
    }
  }

  history_starting_pos_key_ = starting_pos_key;
  current_position_moves_ = moves;
  has_position_ = true;
  return can_reuse;
}

void NodeTree::DeallocateTree() {
  DeleteSubtreeLater(std::move(gamebegin_node_));
  gamebegin_node_ = std::make_unique<uct_node_t>();
  current_head_ = gamebegin_node_.get();
  history_starting_pos_key_ = 0;
  current_position_moves_.clear();
  has_position_ = false;
}

}  // namespace dlshogi_mcts
