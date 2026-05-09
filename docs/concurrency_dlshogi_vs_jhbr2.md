# Concurrency model: dlshogi vs jhbr2

Goal: 60k NPS on a 2× RTX 3090 machine. dlshogi achieves ~31k NPS on
a single 3090 (2 workers/GPU), so 60k+ on 2× is the proven baseline
to match. This document compares the two engines' concurrency models
and lays out the shortest path for jhbr2 to close the gap.

---

## 1. dlshogi — what it does

**Worker topology.** N worker threads per GPU (`UCT_Threads`,
typically 2–4). One `UCTSearcherGroup` per GPU. Workers in a group
are pre-pinned to the GPU at construction (`UctSearch.cpp:544-558`,
`UctSearch.cpp:375` calls `cudaSetDevice(grp->gpu_id)`).

**Per-worker GPU resources.** Each worker owns:
- A pinned host buffer pair (features1, features2, y1, y2) sized for
  `policy_value_batch_maxsize` (`UctSearch.cpp:338-341`,
  `cudaHostAlloc`).
- A TRT execution slot (`nn_tensorrt.cpp:33-44`) consisting of:
  - One `IExecutionContext`.
  - One CUDA stream (`cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`).
  - Device buffers `p1_dev/p2_dev/x1_dev/x2_dev/y1_dev/y2_dev`.
- Slot index = thread index, fixed 1:1.

**Tree mutex model.** A **65 536-bucket position-hashed mutex array**:
```cpp
constexpr uint64_t MUTEX_NUM = 65536;       // 2^16
std::mutex mutexes[MUTEX_NUM];
std::mutex& GetPositionMutex(const Position* pos) {
  return mutexes[pos->getKey() & (MUTEX_NUM - 1)];
}
```
Held during the **entire PUCT walk + leaf expansion**
(`UctSearch.cpp:1410-1429`). Two workers landing on positions whose
zobrist hashes don't collide can walk concurrently with no
synchronization at all.

**Tree topology.** Children are **pre-allocated as a flat array** at
expansion time (`Node.h:78-84`):
```cpp
void ExpandNode(const Position* pos) {
  MoveList<Legal> ml(*pos);
  child = std::make_unique<child_node_t[]>(ml.size());
  child_num = ml.size();
}
```
No linked list, no lazy spawn. After expansion, the array is
read-only — concurrent readers are safe without further locking.
Concurrent expansion of the same node is prevented by the position
mutex; if a stale worker arrives at an already-evaluated node it
discards its playout (`UctSearch.cpp:1536-1537`).

**Backup.** `UpdateResult()` runs **outside any lock**
(`UctSearch.cpp:236-242`):
```cpp
atomic_fetch_add(&current->win, (WinType)result);
if constexpr (VIRTUAL_LOSS != 1) current->move_count += 1 - VIRTUAL_LOSS;
atomic_fetch_add(&child->win, (WinType)result);
```
- `win` is `atomic_t<float>` updated via a CAS loop (`UctSearch.cpp:212-216`).
- `move_count` is **non-atomic** under the assumption `VIRTUAL_LOSS=1`
  (the `if constexpr` is constant-folded away). The virtual loss
  was already added during PUCT under the position mutex.
- Backup is fully concurrent across workers.

**Batch submission.** Each worker accumulates its own batch in its
own pinned buffers, then calls `nn->forward(thread_id, batch_size,
…)` (`UctSearch.cpp:1675`) with no mutex. The TRT path is
`lock-free in `forward`; only the ONNX fallback path takes a mutex.

**No NN result cache.** Every batch is a fresh forward pass.

**Summary.** dlshogi's parallelism is bought with:
1. Coarse position-hashed mutex (held only briefly during PUCT
   descent, never for backup).
2. Pre-allocated children = no spawn-time race.
3. Atomic-where-needed (win) and non-atomic-where-safe (move_count
   under VL=1) for backup.
4. Zero coordination during NN inference (per-worker slots).

---

## 2. jhbr2 — what we do today (commit `156685e`)

**Worker topology.** Same shape as dlshogi (just landed in
`d2b55ba`): `num_gpus × workers_per_gpu` workers, each pinned to
one (evaluator, slot). USI option `WorkersPerGpu` (default 2),
`Threads` is an alias.

**Per-worker GPU resources.** Each evaluator owns N TRT slots
(`mcts/nn_tensorrt.cc`, `Slot` struct). Each slot has its own
`IExecutionContext`, CUDA stream, pinned host buffers, device
buffers. Allocated at evaluator construction. Worker `i` uses
slot `i % workers_per_gpu`. **Equivalent to dlshogi.**

**Tree mutex model.** **One global `std::shared_mutex nodes_mutex_`**
on `Search` (`lc0_mcts/search.h:239`). Plus a 256-bucket
`std::mutex` array hashed by parent address
(`lc0_mcts/node.h:spawn_lock`).

| Operation | Lock taken |
|---|---|
| `PickNodeToExtend` (PUCT walk) | shared_lock on global mutex |
| `ExtendNodeInPlace` (CreateEdges, MakeTerminal) | none |
| `Edge_Iterator::GetOrSpawnNode` | bucket-hashed spawn_lock |
| `DoBackupUpdate` | **unique_lock** on global mutex |
| `IsSearchActive` (info output) | shared_lock |

**Tree topology.** Children are stored as a **lazily-spawned linked
list** of unique_ptr (`Node::child_`, `Node::sibling_`,
`lc0_mcts/node.h:170-175`). `Edge_Iterator::GetOrSpawnNode` walks
the sibling chain looking for the wanted index, creating a new
Node and splicing it in if not present. The bucket spawn_lock
serializes inserts on the same parent; **walks (`Actualize`) are
NOT locked**, which is technically a race with concurrent inserts
but rarely crashes in practice.

**Backup.** `DoBackupUpdate` (`lc0_mcts/search.cc:970`) takes
`unique_lock` on the global mutex and walks ancestors, calling
`FinalizeScoreUpdate` per node. Float fields (`wl_`, `d_`, `m_`)
are non-atomic; safe only under unique_lock. **Backup blocks all
pickers.**

**`n_in_flight_` is atomic** (`std::atomic<uint32_t>`,
`lc0_mcts/node.h:189`). `TryStartScoreUpdate` is a CAS loop. This
was the only field made atomic in the recent rewrite.

**NN cache.** `mcts/nn_cache.h` — opt-in (default size 0 = off),
single global mutex. Disabled by default.

---

## 3. Side-by-side comparison

| Dimension | dlshogi | jhbr2 today |
|---|---|---|
| Per-GPU worker threads | tunable, typical 2–4 | tunable, default 2 |
| GPU exec contexts | 1 per worker | 1 per worker (= same) |
| CUDA streams | 1 per worker | 1 per worker (= same) |
| Global tree mutex | **none** | `shared_mutex` (block-the-world) |
| Per-position mutex | **65 536 bucket** | none |
| Per-parent mutex | none | 256 bucket (spawn only) |
| Children layout | **flat pre-allocated array** | linked-list, lazy-spawned |
| `Edge_Iterator::Actualize` race | impossible (array is read-only) | **present** (walks under-modification list) |
| Backup synchronization | **lock-free** (atomic win, non-atomic mc under VL=1) | **unique_lock on global mutex** |
| Visit-count atomicity | `move_count` non-atomic, `win` atomic | `n_in_flight_` atomic, rest non-atomic |
| NN cache | none | optional (default off) |
| Cross-worker batch combining | none | none (just removed) |

**Where the NPS gap comes from.** Two structural items:

1. **Backup serializes all pickers.** Any worker entering backup
   takes `unique_lock`, blocking every other worker's `shared_lock`
   pick. Under load this becomes the dominant bottleneck (classic
   reader/writer starvation). dlshogi's lock-free backup means
   workers never wait on each other for the score update path.

2. **Lazy-spawned linked list adds CPU work and a race.** Every
   PUCT iteration, every worker walks the sibling chain via
   `Actualize` to find each child. dlshogi reads a flat array
   indexed by edge position — O(1) instead of O(num_visited
   children). And the lazy spawn requires the bucket lock to
   synchronize topology mutations, where dlshogi pre-allocates
   and never mutates after expansion.

The per-worker multi-context GPU layer is **already equivalent**.
The remaining gap is entirely on the CPU/concurrency side.

---

## 4. Path forward — three options

### Option A: Copy dlshogi's MCTS+GPU layer wholesale into jhbr2

Replace `lc0_mcts/{node, search}` with a port of dlshogi's
`UctSearch.{cpp,h}` and `Node.h`. Keep `mcts/nn_tensorrt.cc` (we
already have multi-slot). Keep the encoder.

**Pros**
- Proven design, known to hit the NPS target.
- Eliminates entire categories of bugs we'd otherwise re-discover.
- Smaller engineering surface: we're translating, not designing.

**Cons**
- Lose lc0's algorithmic refinements: FPU at root, dynamic cpuct
  formula, sticky endgames, virtual-loss-weight tuning, our
  shallow mate integration, our tree-reuse via 0/1/2-ply nav.
- All of those would need re-porting on top of the dlshogi base
  (smaller but real follow-up work).
- The lc0 `Edge_Iterator` API is consumed by other places
  (`PickNodeToExtend` PUCT iteration, `VisitedNodes`, info output);
  call sites need rewriting.

**Effort estimate.** 2–3 days for the port + 1–2 days re-adding our
features. ~1 week.

### Option B: Surgically convert jhbr2 to dlshogi's concurrency model

Keep lc0's algorithm (PUCT, FPU, cpuct, etc.) but replace the
concurrency primitives:

1. **Replace child linked-list with pre-allocated `child_node_t[]`
   array** at `CreateEdges` time. Eliminates `GetOrSpawnNode`
   entirely; iterator becomes O(1) array index.
2. **Replace global `shared_mutex` with 65 536-bucket position
   mutex** keyed by `board.Hash()`. Held during PUCT walk +
   leaf expansion only.
3. **Make `wl_`, `n_` atomic** (or use dlshogi's `atomic_t<float>`
   CAS-loop helper) and run `DoBackupUpdate` lock-free.
4. **Drop the global `nodes_mutex_`** entirely — it's no longer
   needed.

**Pros**
- Keeps all our existing lc0 algorithm work.
- Smaller blast radius than option A (only `node.{h,cc}` and
  parts of `search.cc` change; PUCT math, search loop, info
  output all stay).

**Cons**
- Designing the migration carefully (keeping `Edge` policy priors,
  `Node` Q/W/D, etc., while flattening children) is non-trivial.
- Atomic-float backup with running-average Q (lc0's
  `FinalizeScoreUpdate`) is harder than dlshogi's
  `+= win` because we need a divide-by-N step. Would have to
  shift to a sum-then-divide model like dlshogi or accept a
  per-node spinlock just for backup.

**Effort estimate.** 4–6 days. Risky if the atomic-Q model fights
us.

### Option C: Hybrid — copy dlshogi's `Node` (tree layout + backup)
into our `Node` while keeping our `Search`/`SearchWorker`

This is a focused subset of A: replace `lc0_mcts/node.{h,cc}` with
something that mirrors dlshogi's `child_node_t[]` layout and
atomic backup, and adapt `search.cc` to the new API. The PUCT
math, leaf-mate dispatch, Dirichlet noise, info output, etc. all
stay in `search.cc`.

**Pros**
- Targets the exact two structural items causing the gap.
- Doesn't lose any of our algorithmic refinements (they all live
  in `search.cc`).
- Atomic-backup pattern can be copied directly from
  `UctSearch.cpp:236-242` once the field types match.

**Cons**
- `Edge_Iterator` API changes — every call site that iterates
  edges or reads child Q/N needs review. We have many.
- Still need to figure out lc0's running-average Q in the new
  field layout; probably switch to sum-then-divide.

**Effort estimate.** 3–5 days.

---

## 5. Recommendation

**Option C.**

Reasoning:
- Option A throws away the algorithmic work we've done (FPU, cpuct,
  shallow mate integration, virtual_loss_weight knob, tree reuse)
  for a clean architecture. Re-porting all of that on top of
  dlshogi's base is at least as much work as adapting our own
  search.cc to a new Node layout.
- Option B keeps the lazy-linked-list, which is our actual
  bottleneck. Atomic backup helps but the PUCT iteration is still
  walking sibling chains — the per-iteration CPU cost is real.
- Option C copies dlshogi's two winning ideas (flat children +
  lock-free backup) while keeping everything we've built on top
  of lc0's algorithm.

**Concrete migration plan for Option C:**

1. **Replace `Node` field layout:**
   - `wl_` → `std::atomic<double>` (or a wrapper with CAS-fetch-add).
   - `n_` → `std::atomic<uint32_t>`.
   - `child_/sibling_` (linked list) → `std::unique_ptr<Node[]>
     children_` (flat array, allocated at CreateEdges with
     `num_edges_` slots).
   - Keep `n_in_flight_` atomic (already done).
2. **Rewrite `Edge_Iterator` to be array-indexed.** No more
   `Actualize`, no more linked-list walk. `++iter` is `++current_idx_`,
   `iter.GetN()` is `children_[current_idx_].GetN()`.
3. **Drop `Node::child_`, `Node::sibling_`, `Edge::FromMovelist`'s
   chain construction.** Drop `spawn_lock` (no more lazy spawn).
4. **Rewrite `FinalizeScoreUpdate` to atomic CAS-update:**
   ```cpp
   void FinalizeScoreUpdate(float v, float d, float m, int multivisit) {
     uint32_t old_n = n_.fetch_add(multivisit);
     // Update wl_/d_/m_ via atomic CAS loop using running-average formula
     AtomicUpdateRunningMean(&wl_, v, old_n, multivisit);
     // ... d_, m_
     n_in_flight_.fetch_sub(multivisit);
   }
   ```
5. **Drop `Search::nodes_mutex_`.** Replace `IsSearchActive()`'s
   `shared_lock(nodes_mutex_)` with atomic reads of `total_playouts_`.
6. **Add a 65 536-bucket position mutex array** keyed by
   `board.Hash()`. PickNodeToExtend takes the bucket lock for the
   current position only.

**Validation gates after each step:**
- After step 2: single-worker NPS unchanged from baseline.
- After step 4: 2-worker NPS > single-worker NPS (no regression).
- After step 6: 2× workers ≈ 1.7×–1.9× NPS scaling.

**Acceptance:** measure on Panda23. Target: ≥ 60 KNPS at
`NumGPUs=2`, `WorkersPerGpu=2`.

---

## 6. Out of scope for this work

- Nyugyoku encoder change (independent, can run in parallel).
- Re-enabling `MakeSolid` (the dangling-pointer bug; orthogonal).
- One-hot hand-piece features (encoder change, requires retraining).

These should be tracked separately and not bundled into the
concurrency rewrite.
