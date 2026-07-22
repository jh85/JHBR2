// Thread-safe NN inference cache for jhbr2.
//
// The cache is shared by every GPU worker in a Search. Position hashes select
// one of up to 256 independent FIFO shards so unrelated lookups and inserts do
// not serialize. Values are immutable shared objects: Lookup only holds the
// shard lock long enough to acquire a handle, and eviction cannot invalidate a
// value that a worker is still applying to a node.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jhbr2 {

struct CachedNNValue {
  float wdl[3] = {0, 0, 0};
  float moves_left = 0.0f;
  std::vector<float> policy;
  uint16_t num_legal_moves = 0;
};

struct NNCacheStats {
  size_t size = 0;
  size_t capacity = 0;
  uint64_t lookups = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t inserts = 0;
  uint64_t duplicate_inserts = 0;
  uint64_t evictions = 0;
  uint64_t in_flight_owners = 0;
  uint64_t in_flight_waits = 0;
  uint64_t lock_contentions = 0;
  uint64_t lock_wait_ns = 0;
};

class NNCache {
 public:
  using Handle = std::shared_ptr<const CachedNNValue>;

 private:
  struct PendingState {
    std::mutex mutex;
    std::condition_variable ready_cv;
    bool ready = false;
    Handle value;
  };

 public:
  class Probe {
   public:
    enum class State { kHit, kOwner, kWaiter };

    bool IsHit() const { return state_ == State::kHit; }
    bool IsOwner() const { return state_ == State::kOwner; }
    bool IsWaiter() const { return state_ == State::kWaiter; }
    const Handle& Hit() const { return hit_; }

    Handle Wait() const {
      if (!pending_) return {};
      std::unique_lock<std::mutex> lock(pending_->mutex);
      pending_->ready_cv.wait(lock, [&] { return pending_->ready; });
      return pending_->value;
    }

   private:
    friend class NNCache;
    State state_ = State::kHit;
    uint64_t key_ = 0;
    uint16_t num_legal_moves_ = 0;
    Handle hit_;
    std::shared_ptr<PendingState> pending_;
  };

  explicit NNCache(size_t capacity = 0) : capacity_(capacity) {
    if (capacity_ == 0) return;

    shard_count_ = ChooseShardCount(capacity_);
    shard_mask_ = shard_count_ - 1;
    shards_.reserve(shard_count_);

    const size_t base_capacity = capacity_ / shard_count_;
    const size_t remainder = capacity_ % shard_count_;
    for (size_t i = 0; i < shard_count_; ++i) {
      const size_t shard_capacity = base_capacity + (i < remainder ? 1 : 0);
      shards_.push_back(std::make_unique<Shard>(shard_capacity));
    }
  }

  NNCache(const NNCache&) = delete;
  NNCache& operator=(const NNCache&) = delete;

  Handle Lookup(uint64_t key, uint16_t expected_num_moves) {
    if (!Enabled()) return {};

    Shard& shard = GetShard(key);
    auto lock = LockShard(shard);
    shard.lookups.fetch_add(1, std::memory_order_relaxed);

    const auto it = shard.map.find(key);
    if (it == shard.map.end() ||
        it->second->num_legal_moves != expected_num_moves) {
      shard.misses.fetch_add(1, std::memory_order_relaxed);
      return {};
    }

    shard.hits.fetch_add(1, std::memory_order_relaxed);
    return it->second;
  }

  Probe LookupOrReserve(uint64_t key, uint16_t expected_num_moves) {
    Probe probe;
    probe.key_ = key;
    probe.num_legal_moves_ = expected_num_moves;
    if (!Enabled()) {
      probe.state_ = Probe::State::kOwner;
      return probe;
    }

    Shard& shard = GetShard(key);
    auto lock = LockShard(shard);
    shard.lookups.fetch_add(1, std::memory_order_relaxed);

    const auto cached = shard.map.find(key);
    if (cached != shard.map.end() &&
        cached->second->num_legal_moves == expected_num_moves) {
      shard.hits.fetch_add(1, std::memory_order_relaxed);
      probe.state_ = Probe::State::kHit;
      probe.hit_ = cached->second;
      return probe;
    }

    shard.misses.fetch_add(1, std::memory_order_relaxed);
    const PendingKey pending_key{key, expected_num_moves};
    const auto pending = shard.in_flight.find(pending_key);
    if (pending != shard.in_flight.end()) {
      shard.in_flight_waits.fetch_add(1, std::memory_order_relaxed);
      probe.state_ = Probe::State::kWaiter;
      probe.pending_ = pending->second;
      return probe;
    }

    shard.in_flight_owners.fetch_add(1, std::memory_order_relaxed);
    probe.state_ = Probe::State::kOwner;
    probe.pending_ = std::make_shared<PendingState>();
    shard.in_flight.emplace(pending_key, probe.pending_);
    return probe;
  }

  Handle Insert(uint64_t key, CachedNNValue value) {
    if (!Enabled()) return {};

    // Allocate the value before acquiring the shard lock. If another worker
    // wins the same insertion race, destruction also happens after unlock.
    Handle new_value =
        std::make_shared<const CachedNNValue>(std::move(value));
    Handle evicted_value;

    Shard& shard = GetShard(key);
    auto lock = LockShard(shard);
    const auto existing = shard.map.find(key);
    if (existing != shard.map.end()) {
      if (existing->second->num_legal_moves ==
          new_value->num_legal_moves) {
        shard.duplicate_inserts.fetch_add(1, std::memory_order_relaxed);
        return existing->second;
      }
      evicted_value = std::move(existing->second);
      existing->second = std::move(new_value);
      return existing->second;
    }

    if (shard.map.size() >= shard.capacity) {
      const uint64_t evict_key = shard.insertion_order.front();
      shard.insertion_order.pop_front();
      auto evict_it = shard.map.find(evict_key);
      if (evict_it != shard.map.end()) {
        evicted_value = std::move(evict_it->second);
        shard.map.erase(evict_it);
        shard.evictions.fetch_add(1, std::memory_order_relaxed);
      }
    }

    shard.map.emplace(key, std::move(new_value));
    shard.insertion_order.push_back(key);
    shard.size.store(shard.map.size(), std::memory_order_relaxed);
    shard.inserts.fetch_add(1, std::memory_order_relaxed);
    return shard.map.find(key)->second;
  }

  Handle Publish(Probe probe, CachedNNValue value) {
    if (!probe.IsOwner() || !probe.pending_) return {};

    Handle published = Insert(probe.key_, std::move(value));
    Shard& shard = GetShard(probe.key_);
    {
      auto lock = LockShard(shard);
      const PendingKey pending_key{probe.key_, probe.num_legal_moves_};
      const auto pending = shard.in_flight.find(pending_key);
      if (pending != shard.in_flight.end() &&
          pending->second == probe.pending_) {
        shard.in_flight.erase(pending);
      }
    }
    {
      std::lock_guard<std::mutex> lock(probe.pending_->mutex);
      probe.pending_->value = published;
      probe.pending_->ready = true;
    }
    probe.pending_->ready_cv.notify_all();
    return published;
  }

  // Complete an in-flight reservation without publishing a value.  This is
  // used when inference returns an invalid result: waiters must be released,
  // but the bad value must never enter the cache.
  void Cancel(Probe probe) {
    if (!probe.IsOwner() || !probe.pending_) return;

    Shard& shard = GetShard(probe.key_);
    {
      auto lock = LockShard(shard);
      const PendingKey pending_key{probe.key_, probe.num_legal_moves_};
      const auto pending = shard.in_flight.find(pending_key);
      if (pending != shard.in_flight.end() &&
          pending->second == probe.pending_) {
        shard.in_flight.erase(pending);
      }
    }
    {
      std::lock_guard<std::mutex> lock(probe.pending_->mutex);
      probe.pending_->value.reset();
      probe.pending_->ready = true;
    }
    probe.pending_->ready_cv.notify_all();
  }

  void Clear() {
    for (auto& shard_ptr : shards_) {
      Shard& shard = *shard_ptr;
      std::lock_guard<std::mutex> lock(shard.mutex);
      shard.map.clear();
      shard.insertion_order.clear();
      shard.size.store(0, std::memory_order_relaxed);
    }
  }

  bool Enabled() const { return capacity_ != 0; }
  size_t Capacity() const { return capacity_; }

  size_t Size() const {
    size_t size = 0;
    for (const auto& shard : shards_) {
      size += shard->size.load(std::memory_order_relaxed);
    }
    return size;
  }

  NNCacheStats GetStats() const {
    NNCacheStats stats;
    stats.capacity = capacity_;
    for (const auto& shard : shards_) {
      stats.size += shard->size.load(std::memory_order_relaxed);
      stats.lookups += shard->lookups.load(std::memory_order_relaxed);
      stats.hits += shard->hits.load(std::memory_order_relaxed);
      stats.misses += shard->misses.load(std::memory_order_relaxed);
      stats.inserts += shard->inserts.load(std::memory_order_relaxed);
      stats.duplicate_inserts +=
          shard->duplicate_inserts.load(std::memory_order_relaxed);
      stats.evictions += shard->evictions.load(std::memory_order_relaxed);
      stats.in_flight_owners +=
          shard->in_flight_owners.load(std::memory_order_relaxed);
      stats.in_flight_waits +=
          shard->in_flight_waits.load(std::memory_order_relaxed);
      stats.lock_contentions +=
          shard->lock_contentions.load(std::memory_order_relaxed);
      stats.lock_wait_ns +=
          shard->lock_wait_ns.load(std::memory_order_relaxed);
    }
    return stats;
  }

  void ResetStats() {
    for (auto& shard : shards_) {
      shard->lookups.store(0, std::memory_order_relaxed);
      shard->hits.store(0, std::memory_order_relaxed);
      shard->misses.store(0, std::memory_order_relaxed);
      shard->inserts.store(0, std::memory_order_relaxed);
      shard->duplicate_inserts.store(0, std::memory_order_relaxed);
      shard->evictions.store(0, std::memory_order_relaxed);
      shard->in_flight_owners.store(0, std::memory_order_relaxed);
      shard->in_flight_waits.store(0, std::memory_order_relaxed);
      shard->lock_contentions.store(0, std::memory_order_relaxed);
      shard->lock_wait_ns.store(0, std::memory_order_relaxed);
    }
  }

 private:
  static constexpr size_t kMaxShards = 256;

  struct PendingKey {
    uint64_t key;
    uint16_t num_legal_moves;

    bool operator==(const PendingKey& other) const {
      return key == other.key && num_legal_moves == other.num_legal_moves;
    }
  };

  struct PendingKeyHash {
    size_t operator()(const PendingKey& pending) const {
      size_t hash = std::hash<uint64_t>{}(pending.key);
      hash ^= std::hash<uint16_t>{}(pending.num_legal_moves) +
              0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
      return hash;
    }
  };

  struct Shard {
    explicit Shard(size_t capacity_in) : capacity(capacity_in) {
      map.reserve(static_cast<size_t>(capacity * 1.3) + 1);
    }

    const size_t capacity;
    mutable std::mutex mutex;
    std::unordered_map<uint64_t, Handle> map;
    std::deque<uint64_t> insertion_order;
    std::unordered_map<PendingKey, std::shared_ptr<PendingState>,
                       PendingKeyHash>
        in_flight;

    std::atomic<size_t> size{0};
    std::atomic<uint64_t> lookups{0};
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> inserts{0};
    std::atomic<uint64_t> duplicate_inserts{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> in_flight_owners{0};
    std::atomic<uint64_t> in_flight_waits{0};
    std::atomic<uint64_t> lock_contentions{0};
    std::atomic<uint64_t> lock_wait_ns{0};
  };

  static size_t ChooseShardCount(size_t capacity) {
    const size_t limit = std::min(capacity, kMaxShards);
    size_t count = 1;
    while (count <= limit / 2) count *= 2;
    return count;
  }

  Shard& GetShard(uint64_t key) {
    return *shards_[static_cast<size_t>(key) & shard_mask_];
  }

  static std::unique_lock<std::mutex> LockShard(Shard& shard) {
    std::unique_lock<std::mutex> lock(shard.mutex, std::try_to_lock);
    if (lock.owns_lock()) return lock;

    shard.lock_contentions.fetch_add(1, std::memory_order_relaxed);
    const auto start = std::chrono::steady_clock::now();
    lock.lock();
    const auto waited = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start);
    shard.lock_wait_ns.fetch_add(static_cast<uint64_t>(waited.count()),
                                 std::memory_order_relaxed);
    return lock;
  }

  const size_t capacity_;
  size_t shard_count_ = 0;
  size_t shard_mask_ = 0;
  std::vector<std::unique_ptr<Shard>> shards_;
};

}  // namespace jhbr2
