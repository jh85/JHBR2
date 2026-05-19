// Thread-safe NN inference cache for jhbr2.
//
// Caches (board.Hash()) → (wdl, mlh, policy, num_legal_moves) so that
// MCTS doesn't re-evaluate positions it has already evaluated.
//
// Common MCTS hit patterns:
//   1. Transpositions within a single search (same position reached
//      via different move orders) — depending on tree shape and book
//      depth, cache hit rate can be 10-40%.
//   2. Subtree revisits across moves when tree reuse is enabled.
//   3. Self-play training loops where many positions repeat.
//
// Design:
//   - FIFO eviction at fixed capacity (simpler than LRU; lc0 also uses
//     FIFO and reports it works fine in practice).
//   - Single std::mutex around all operations. SpinMutex would be
//     faster under heavy contention but adds dependency and most of
//     our use is during NN eval batching where contention is low.
//   - Lookup returns by VALUE (copy) instead of pointer-with-pin, so
//     callers don't need to manage entry lifetime.
//
// USI option: `NNCacheSize` (number of entries). Each entry costs
// ~32 bytes overhead + 4 bytes per legal move policy entry. At
// typical 30 legal moves that's ~152 bytes. So 2M entries = ~300 MB.
// Set to 0 to disable.

#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace jhbr2 {

struct CachedNNValue {
  float wdl[3] = {0, 0, 0};        // win, draw, loss probabilities
  float mlh = 0.0f;                // moves-left head output
  bool has_mlh = false;
  std::vector<float> policy;        // indexed by legal-move position
  uint16_t num_legal_moves = 0;     // collision guard
};

class NNCache {
 public:
  explicit NNCache(size_t capacity = 0)
      : capacity_(capacity) {
    if (capacity > 0) map_.reserve(static_cast<size_t>(capacity * 1.3));
  }

  // Look up by key. If found AND num_legal_moves matches, copy into
  // *out and return true. Otherwise return false.
  bool Lookup(uint64_t key, uint16_t expected_num_moves,
              CachedNNValue* out) {
    if (capacity_ == 0) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it == map_.end()) {
      misses_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (it->second->num_legal_moves != expected_num_moves) {
      // Hash collision (different position with same hash) or stale
      // entry (legal moves changed somehow). Treat as miss.
      misses_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    *out = *it->second;
    hits_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  // Insert. If key already present, leaves existing entry alone
  // (avoids invalidating any in-flight Lookup result references).
  void Insert(uint64_t key, CachedNNValue value) {
    if (capacity_ == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (map_.count(key)) return;
    map_.emplace(key, std::make_unique<CachedNNValue>(std::move(value)));
    insertion_order_.push_back(key);
    while (map_.size() > capacity_) {
      uint64_t evict_key = insertion_order_.front();
      insertion_order_.pop_front();
      map_.erase(evict_key);
    }
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    map_.clear();
    insertion_order_.clear();
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return map_.size();
  }
  size_t Capacity() const { return capacity_; }
  uint64_t Hits() const { return hits_.load(std::memory_order_relaxed); }
  uint64_t Misses() const { return misses_.load(std::memory_order_relaxed); }

  // Resets hit/miss counters but keeps the cached entries.
  void ResetStats() {
    hits_.store(0, std::memory_order_relaxed);
    misses_.store(0, std::memory_order_relaxed);
  }

 private:
  size_t capacity_;
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, std::unique_ptr<CachedNNValue>> map_;
  std::deque<uint64_t> insertion_order_;  // for FIFO eviction
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};
};

}  // namespace jhbr2
