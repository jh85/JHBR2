#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "inference/nn_cache.h"

namespace {

int failures = 0;

void Check(const char* name, bool condition) {
  if (!condition) {
    std::printf("  FAIL  %s\n", name);
    ++failures;
  }
}

jhbr2::CachedNNValue MakeValue(float value, int policy_size) {
  jhbr2::CachedNNValue cached;
  cached.wdl[0] = value;
  cached.wdl[1] = 0.0f;
  cached.wdl[2] = 1.0f - value;
  cached.num_legal_moves = static_cast<uint16_t>(policy_size);
  cached.policy.assign(policy_size, 1.0f / policy_size);
  return cached;
}

void TestDisabledCache() {
  jhbr2::NNCache cache;
  cache.Insert(1, MakeValue(0.75f, 4));
  Check("disabled lookup misses", !cache.Lookup(1, 4));
  Check("disabled cache remains empty", cache.Size() == 0);
  Check("disabled cache records no probes", cache.GetStats().lookups == 0);
}

void TestLookupAndCollisionGuard() {
  jhbr2::NNCache cache(4);
  cache.Insert(11, MakeValue(0.75f, 4));

  auto hit = cache.Lookup(11, 4);
  Check("lookup finds inserted value", hit && hit->wdl[0] == 0.75f);
  Check("lookup returns complete policy", hit && hit->policy.size() == 4);
  Check("legal move count guards lookup", !cache.Lookup(11, 3));

  cache.Insert(11, MakeValue(0.5f, 3));
  auto replacement = cache.Lookup(11, 3);
  Check("different legal count replaces colliding value",
        replacement && replacement->policy.size() == 3);

  const auto stats = cache.GetStats();
  Check("lookup count", stats.lookups == 3);
  Check("hit count", stats.hits == 2);
  Check("miss count", stats.misses == 1);
  Check("insert count", stats.inserts == 1);
}

void TestFifoAndHandleLifetime() {
  jhbr2::NNCache cache(1);
  cache.Insert(1, MakeValue(0.25f, 2));
  auto retained = cache.Lookup(1, 2);
  cache.Insert(2, MakeValue(0.75f, 3));

  Check("FIFO evicts oldest key", !cache.Lookup(1, 2));
  Check("FIFO retains newest key", static_cast<bool>(cache.Lookup(2, 3)));
  Check("evicted handle remains valid",
        retained && retained->wdl[0] == 0.25f && retained->policy.size() == 2);
  Check("capacity remains bounded", cache.Size() == 1);
  Check("eviction count", cache.GetStats().evictions == 1);
}

void TestDuplicateInsert() {
  jhbr2::NNCache cache(2);
  cache.Insert(7, MakeValue(0.25f, 2));
  cache.Insert(7, MakeValue(0.75f, 2));

  auto hit = cache.Lookup(7, 2);
  Check("first duplicate value wins", hit && hit->wdl[0] == 0.25f);
  Check("duplicate insert counted", cache.GetStats().duplicate_inserts == 1);
  Check("duplicate does not grow cache", cache.Size() == 1);
}

void TestInFlightReservation() {
  jhbr2::NNCache cache(4);
  auto owner = cache.LookupOrReserve(17, 4);
  auto waiter = cache.LookupOrReserve(17, 4);

  Check("first miss owns in-flight evaluation", owner.IsOwner());
  Check("second miss waits for in-flight evaluation", waiter.IsWaiter());

  jhbr2::NNCache::Handle waited_value;
  std::thread waiting_thread([&] { waited_value = waiter.Wait(); });
  auto published = cache.Publish(std::move(owner), MakeValue(0.625f, 4));
  waiting_thread.join();

  Check("publish returns cached value",
        published && published->wdl[0] == 0.625f);
  Check("waiter receives published value",
        waited_value && waited_value->wdl[0] == 0.625f);
  Check("published value becomes a normal hit",
        cache.LookupOrReserve(17, 4).IsHit());

  const auto stats = cache.GetStats();
  Check("one in-flight owner", stats.in_flight_owners == 1);
  Check("one in-flight waiter", stats.in_flight_waits == 1);
  Check("in-flight dedup inserts once", stats.inserts == 1);
  Check("in-flight dedup avoids duplicate insert",
        stats.duplicate_inserts == 0);
}

void TestCancelledReservation() {
  jhbr2::NNCache cache(4);
  auto owner = cache.LookupOrReserve(23, 5);
  auto waiter = cache.LookupOrReserve(23, 5);

  jhbr2::NNCache::Handle waited_value;
  std::thread waiting_thread([&] { waited_value = waiter.Wait(); });
  cache.Cancel(std::move(owner));
  waiting_thread.join();

  Check("cancel wakes waiter with no value", !waited_value);
  Check("cancel does not insert", cache.Size() == 0);
  Check("cancel permits a new owner", cache.LookupOrReserve(23, 5).IsOwner());
}

void TestConcurrentAccess() {
  constexpr int kThreads = 8;
  constexpr int kEntriesPerThread = 256;
  constexpr int kEntryCount = kThreads * kEntriesPerThread;
  jhbr2::NNCache cache(kEntryCount);
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;

  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&, thread] {
      const int first = thread * kEntriesPerThread;
      for (int i = 0; i < kEntriesPerThread; ++i) {
        const uint64_t key = static_cast<uint64_t>(first + i);
        cache.Insert(key, MakeValue(0.5f, 8));
        auto hit = cache.Lookup(key, 8);
        if (!hit || hit->policy.size() != 8) failed.store(true);
      }
    });
  }
  for (auto& thread : threads) thread.join();

  Check("concurrent insert and lookup", !failed.load());
  Check("concurrent cache reaches expected size", cache.Size() == kEntryCount);

  threads.clear();
  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&, thread] {
      for (int key = thread; key < kEntryCount; key += kThreads) {
        if (!cache.Lookup(static_cast<uint64_t>(key), 8)) failed.store(true);
      }
    });
  }
  for (auto& thread : threads) thread.join();
  Check("concurrent read pass", !failed.load());

  cache.ResetStats();
  const auto reset = cache.GetStats();
  Check("stats reset keeps entries", reset.size == kEntryCount);
  Check("stats reset clears counters",
        reset.lookups == 0 && reset.hits == 0 && reset.inserts == 0);
}

}  // namespace

int main() {
  TestDisabledCache();
  TestLookupAndCollisionGuard();
  TestFifoAndHandleLifetime();
  TestDuplicateInsert();
  TestInFlightReservation();
  TestCancelledReservation();
  TestConcurrentAccess();

  std::printf("\n=== NN cache: %d failed ===\n", failures);
  return failures == 0 ? 0 : 1;
}
