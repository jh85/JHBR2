#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace dlshogi_mcts {

constexpr int kNotExpanded = -1;
constexpr int kVirtualLoss = 1;

template <typename T>
inline void AtomicFetchAdd(std::atomic<T>* obj, T arg) {
  T expected = obj->load(std::memory_order_relaxed);
  while (!obj->compare_exchange_weak(expected, expected + arg,
                                     std::memory_order_acq_rel,
                                     std::memory_order_relaxed)) {
  }
}

class Timer {
 public:
  Timer() { Restart(); }
  void Restart() { start_ = std::chrono::steady_clock::now(); }
  int ElapsedMs() const {
    return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_).count());
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace dlshogi_mcts
