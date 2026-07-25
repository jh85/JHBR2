#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "dlshogi_mcts/types.h"
#include "shogi/board.h"
#include "shogi/types.h"

namespace dlshogi_mcts {

struct uct_node_t;

// Owns one lazily-created child node while publishing its address atomically.
// Search workers only call get()/GetOrCreate(); tree reuse may Take()/Reset()
// slots after every search worker has joined.
class child_node_slot_t {
 public:
  child_node_slot_t() = default;
  ~child_node_slot_t();

  child_node_slot_t(const child_node_slot_t&) = delete;
  child_node_slot_t& operator=(const child_node_slot_t&) = delete;
  child_node_slot_t(child_node_slot_t&&) = delete;
  child_node_slot_t& operator=(child_node_slot_t&&) = delete;

  uct_node_t* get() const noexcept;
  explicit operator bool() const noexcept { return get() != nullptr; }

  // Concurrent callers either publish a new node or receive the node another
  // caller published. The returned pointer remains owned by this slot.
  uct_node_t* GetOrCreate();

  // Tree-lifecycle operations. These must only run after search workers join.
  std::unique_ptr<uct_node_t> Take() noexcept;
  std::unique_ptr<uct_node_t> Reset(
      std::unique_ptr<uct_node_t> replacement) noexcept;

 private:
  std::atomic<uct_node_t*> node_{nullptr};
};

struct child_node_t {
  child_node_t() = default;
  explicit child_node_t(lczero::Move m) : move(m) {}

  child_node_t(child_node_t&& o) noexcept
      : move(o.move),
        nnrate(o.nnrate),
        move_count(o.move_count.load(std::memory_order_relaxed)),
        win(o.win.load(std::memory_order_relaxed)),
        sum_m(o.sum_m.load(std::memory_order_relaxed)),
        flags(o.flags.load(std::memory_order_relaxed)) {}

  child_node_t& operator=(child_node_t&& o) noexcept {
    move = o.move;
    nnrate = o.nnrate;
    move_count.store(o.move_count.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    win.store(o.win.load(std::memory_order_relaxed), std::memory_order_relaxed);
    sum_m.store(o.sum_m.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
    flags.store(o.flags.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
    return *this;
  }

  child_node_t(const child_node_t&) = delete;
  child_node_t& operator=(const child_node_t&) = delete;

  bool IsWin() const { return flags.load(std::memory_order_acquire) & kWin; }
  bool IsLose() const { return flags.load(std::memory_order_acquire) & kLose; }
  bool IsDraw() const { return flags.load(std::memory_order_acquire) & kDraw; }
  void SetWin() { flags.fetch_or(kWin, std::memory_order_acq_rel); }
  void SetLose() { flags.fetch_or(kLose, std::memory_order_acq_rel); }
  void SetDraw() { flags.fetch_or(kDraw, std::memory_order_acq_rel); }

  lczero::Move move;
  float nnrate = 0.0f;
  std::atomic<int> move_count{0};
  std::atomic<float> win{0.0f};
  std::atomic<float> sum_m{0.0f};  // sum of subtree moves-left (for MLH M-effect)

 private:
  enum : uint8_t { kWin = 1, kLose = 2, kDraw = 4 };
  std::atomic<uint8_t> flags{0};
};

struct uct_node_t {
  uct_node_t() = default;

  bool IsEvaled() const {
    return move_count.load(std::memory_order_acquire) != kNotExpanded;
  }
  void SetEvaled() { move_count.store(0, std::memory_order_release); }

  void ExpandNode(const lczero::ShogiBoard* board);
  void InitChildNodes();
  uct_node_t* CreateChildNode(int i);
  void CreateSingleChildNode(lczero::Move move);
  uct_node_t* ReleaseChildrenExceptOne(lczero::Move move);

  std::atomic<int> move_count{kNotExpanded};
  std::atomic<float> win{0.0f};
  std::atomic<float> visited_nnrate{0.0f};
  float eval_m = 0.0f;  // this node's own NN moves-left (set once at eval)
  short child_num = 0;
  std::unique_ptr<child_node_t[]> child;
  std::unique_ptr<child_node_slot_t[]> child_nodes;
};

class NodeTree {
 public:
  NodeTree();
  ~NodeTree();

  // Returns true when the new move history extends the previous root.
  bool ResetToPosition(uint64_t starting_pos_key,
                       const std::vector<lczero::Move>& moves);
  uct_node_t* GetCurrentHead() const { return current_head_; }
  void DeallocateTree();

 private:
  uct_node_t* current_head_ = nullptr;
  std::unique_ptr<uct_node_t> gamebegin_node_;
  uint64_t history_starting_pos_key_ = 0;
  std::vector<lczero::Move> current_position_moves_;
  bool has_position_ = false;
};

}  // namespace dlshogi_mcts
