# Postmortem: jhbr2 multi-worker MCTS scaling failure

This document is for a **fresh-eyes agent** on Panda23 (or similar
2× RTX 3090 hardware) to investigate why jhbr2's multi-worker MCTS
fails to scale, despite a series of changes that should — on paper —
have fixed it. The user has lost a few sessions to this and wants
an outside perspective. You may have access to real hardware that
the prior sessions didn't.

If you confirm the prior diagnosis, fine — write up your findings
and we'll proceed with the dlshogi MCTS port (see
`docs/dlshogi_port_plan.md`). If you find a different root cause,
that's also useful — please document it so we can act on it.

---

## 1. Goal

**Hit ≥ 60 KNPS on a 2× RTX 3090 machine (Panda23) running jhbr2.**

This number comes from dlshogi's published baseline:
~31 KNPS on a single 3090 with 2 workers per GPU. So 60 KNPS on
2× 3090 is the proven dlshogi number we want to match. Currently
jhbr2 maxes out at ~10 KNPS with **very poor minimum-NPS behavior**
in actual games (sometimes dropping below 1 KNPS).

The whole point of the recent rewrite was to match dlshogi's GPU
dispatch architecture so we could match their NPS. We got the
architecture right but the NPS didn't follow.

## 2. What we attempted

### Phase 0 (pre-rewrite): combining backend
- One central GPU dispatcher thread (`Backend::GPULoop`) collected
  per-worker submissions, fused them into one large batch, sent to
  GPU, distributed results back via condvars.
- Worked but bottlenecked at low NPS (~10 KNPS max) and produced
  very low minimum NPS in games.

### Phase 1 (commit `d2b55ba`): per-worker GPU dispatch
- Removed the central dispatcher.
- Each worker pinned to one (`evaluator`, `slot`) pair.
- Each evaluator on each GPU has N TRT execution contexts +
  CUDA streams + pinned host buffers + device buffers (one set
  per slot).
- Worker calls `evaluator->EvaluateBatchSlot(slot_id, batch)`
  directly, no central queue.
- This **matches dlshogi's per-worker GPU model** as far as we can
  tell from `DeepLearningShogi/usi/UctSearch.cpp` and
  `nn_tensorrt.cpp`. Multiple workers on the same GPU run on
  separate streams with separate execution contexts — TRT
  supports this.

### Phase 2 (commits `db0797a`, `156685e`): atomic n_in_flight + bucket spawn lock + shared_lock
- `n_in_flight_` field on `Node` made `std::atomic<uint32_t>`,
  with CAS-loop in `TryStartScoreUpdate` and atomic
  `fetch_add/sub` in `IncrementNInFlight` /
  `CancelScoreUpdate` / `FinalizeScoreUpdate`.
- Added 256-bucket `std::mutex` array hashed by parent address
  (`spawn_lock::Mutexes()` in `lc0_mcts/node.h`). Acquired by
  `Edge_Iterator::GetOrSpawnNode` before mutating the parent's
  child sibling-list.
- `PickNodeToExtend` switched from `unique_lock` →
  `shared_lock` on the global `Search::nodes_mutex_`.

### Phase 3 (commit `ae3429a`): lock-free backup
- Added per-Node `std::atomic_flag stats_spin_` (1 byte).
- Wrapped `FinalizeScoreUpdate` /
  `AdjustForTerminal` / `RevertTerminalVisits` with the spinlock
  for the running-mean update of `wl_/d_/m_/n_`.
- Dropped `unique_lock` from `DoBackupUpdate` entirely.
- Eliminated `shared_collisions_` shared list; each worker
  cancels its own collisions inline.

## 3. What we measured

All measurements done locally, **not on Panda23** — the prior
agent didn't have access. Test engine was
`shogi_bt4_epoch13_b128.engine` (small TRT engine with max
batch=128). Test position: startpos. `MinibatchSize=64`,
`byoyomi=4000ms`.

| WorkersPerGpu | NPS (Phase 1, post-`d2b55ba`) | NPS (Phase 3, post-`ae3429a`) |
|---|---|---|
| 1 | 432 | 463 |
| 2 | 308 | 325 |
| 3 | 221 | — |
| 4 | 130 | 155 |
| 8 | — | 51 |

**Two facts:**
- W=1 NPS unchanged after each phase (no regression).
- More workers always made things worse, not better. NPS scales
  *inversely* with worker count. None of phases 2–3 changed this.

## 4. Diagnosis (current best guess) and uncertainty

After lock-free backup landed and the multi-worker NPS still got
worse, the prior session concluded the bottleneck must be in tree
**iteration during PUCT walks**, specifically:

- `Edge_Iterator::Actualize` (in `lc0_mcts/node.h`) walks the
  parent's `child_/sibling_` linked list looking for the wanted
  child index, on every PUCT iteration step.
- With multiple workers walking under `shared_lock`, concurrent
  inserts via `GetOrSpawnNode` on the same parent (under bucket
  lock) can race against the readers. The bucket lock protects
  writers from each other but **not from concurrent readers**.
- Hypothesis: cache-line ping-pong between cores reading and
  writing the same `unique_ptr` chain causes severe slowdown
  even when the data races don't crash.

**Uncertainties / things we haven't actually verified:**
- We never profiled. No `perf` data on what the workers are
  actually waiting on.
- We didn't try locking the iteration too (would be over-broad
  but would isolate whether iteration races are the issue).
- We didn't measure on production hardware (Panda23 with
  production model). The b128 engine is tiny — its kernels
  finish in microseconds, so per-iteration CPU work dominates.
  On a real production engine where each TRT call is 5–20 ms,
  the CPU contention may matter much less.
- We didn't measure on the actual production model and engine
  size. NPS scaling might look different there.

**This last point is important.** If on Panda23 with a real
production engine you measure W=2 ≈ 1.7× × W=1 NPS, then the
"multi-worker doesn't scale" diagnosis from the local b128 tests
was wrong, and we don't need any of this rewrite. The combining
backend was the actual problem and the fix from `d2b55ba` is
sufficient.

**That's the first thing you should check.** Real hardware,
real model, just measure. If it scales, we're done — back out
the half-finished concurrency work and move on.

## 5. What you should do (suggested investigation order)

### Step 1: actually measure on Panda23

Use the production model (whatever the user is currently using
on floodgate) and `tools/benchmark.py`:

```bash
# Build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON
make -j jhbr2

# Sweep (run from repo root)
for W in 1 2 4; do
  for G in 1 2; do
    python3 tools/benchmark.py ./build/jhbr2 \
      /path/to/production.engine \
      --threads $W --gpus $G --minibatch 256 \
      --byoyomi 5000 --limit 20 \
      > "bench_w${W}_g${G}.log"
  done
done
```

Report median NPS for each cell. **If W=2 G=2 ≥ 60 KNPS we're
done.** Walk the user through backing out the unstable changes
and we ship.

### Step 2: if it doesn't scale, profile

Don't speculate further — get data. On Panda23:

```bash
# perf-profile a 30-second search at W=4 G=2
sudo sysctl kernel.perf_event_paranoid=1
perf record -g -F 999 -- ./build/jhbr2 < usi_input_w4g2.txt
perf report --stdio | head -100
```

Look for:
- Time in `std::__atomic_base::load` or similar — atomic contention.
- Time in `std::mutex::lock` or `__lll_lock_*` — mutex contention.
- Time in `Edge_Iterator::Actualize` — iteration overhead (should
  be small).
- Time in `PickNodeToExtend` vs `DoBackupUpdate` vs `RunNNComputation`
  — wall-clock distribution.
- Ratio of total CPU time to wall-clock time (parallel efficiency).

### Step 3: if profile points to Actualize / linked-list iteration

Confirms the linked-list-is-bottleneck hypothesis. Proceed with
the dlshogi port (`docs/dlshogi_port_plan.md` and
`docs/dlshogi_port_step1_briefing.md`).

### Step 4: if profile points elsewhere

This is the case where fresh eyes would be most valuable. Possible
alternatives:

- **CUDA stream contention.** TRT's claim that multi-context-on-same-engine
  runs concurrently may not hold on this specific engine + driver
  + GPU combo. Try `nsys profile` to see actual stream concurrency.
- **CPU/memory bandwidth saturation.** Maybe the encoder is
  bandwidth-bound and scales sub-linearly when 4 threads encode
  in parallel.
- **NN cache contention.** Default cache size is 0 (disabled),
  but worth confirming.
- **Spurious wakeups in `Search::IsSearchActive`.** Called per
  worker iteration; takes a `shared_lock`. Maybe contention there
  is more than expected.
- **Per-node spinlock contention at root.** All workers backup
  through root every iteration. The 1-byte spinlock could be a
  cache-line hotspot.
- **`std::shared_mutex` itself being slow on Linux.** pthread_rwlock
  has known performance pathologies. A real `std::mutex` is sometimes
  faster.

## 6. What's already known to be wrong / dead ends

So you don't waste time:

- **Pre-allocating `Node[num_edges]` at expansion time.** Memory
  blowup is ~5× because shogi has 50+ avg legal moves. Was
  considered then rejected.
- **CAS-based lock-free linked list for the child chain.** PhD-thesis
  territory; not justified for 2–8 workers.
- **Reverting to combining backend.** That was the original
  bottleneck (~10 KNPS max) — going back loses ground.

## 7. Setup pointers

Repo: `https://github.com/jh85/JHBR2`, branch `main`.

Current HEAD has commits:
- `d2b55ba` — per-worker GPU dispatch (the foundation; keep this).
- `db0797a` — atomic `n_in_flight_` (keep; harmless).
- `156685e` — bucket spawn lock + shared_lock pickers.
- `ae3429a` — lock-free backup with per-node spinlocks.
- `c8242af` — port plan doc.
- `10ec162` — port step 1 briefing.

Local CMake build: `mkdir -p build && cd build && cmake ..
-DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON && make -j jhbr2`.
Requires CUDA + TensorRT installed.

USI options that affect scaling:
- `Threads` (alias) / `WorkersPerGpu`: workers per GPU. Default 2.
- `NumGPUs`: number of GPUs. Default 1.
- `MinibatchSize`: per-worker batch size. Default 256.
- `MaxGpuBatch`: per-call batch cap (chunks if exceeded). Default 1024.
- `PerLeafGathering`: should stay `true`. Bulk path is dead code.

## 8. What "success" looks like for this investigation

In rough order of value:

1. **Measurement on real hardware.** "On Panda23 with W=2 G=2,
   I see X KNPS" is itself a data point worth multiple sessions of
   speculation.
2. **Profile data.** Where the workers actually spend their time
   when NPS doesn't scale.
3. **Either:** confirmation that the dlshogi port is the right next
   step, **or:** a different diagnosis with a smaller fix.
4. **A short writeup** — even a few paragraphs — so the next session
   doesn't re-discover the same things.

You won't lose anything by trying. Worst case you confirm what we
think; best case you find something cheaper.

## 9. Things the prior session might have been wrong about

A list of explicit "I'm not 100% sure about this" items, in case
you have evidence to the contrary:

- That the linked-list iteration race is the dominant bottleneck
  (never confirmed via profiling).
- That `shared_mutex`'s reader-writer cost on Linux is acceptable
  (we just used it without measuring).
- That TRT's multi-context-on-same-engine model gives true GPU
  concurrency (assumed from docs, never verified with `nsys`).
- That the b128 engine's behavior generalizes to production
  engines (it might not — production engines have much larger
  per-call costs).
- That `WorkersPerGpu=2` is the right default (dlshogi sometimes
  uses 4).

Be skeptical of all of these.

Good luck. Honest assessment beats matching the prior diagnosis.
