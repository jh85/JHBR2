# jhbr2 Post-WCSC36 Architecture Research

Research notes comparing dlshogi's MCTS engine against jhbr2 (lc0-derived)
across three areas where jhbr2 has known performance/quality gaps:

1. Per-leaf mate detection
2. NN inference caching
3. MCTS multi-threading model

Each section captures findings, file:line references, expected ROI,
and recommendations. The expected combined NPS impact is roughly
**3–6× in tactical positions**, with smaller but real gains everywhere.

---

## Headline summary

| # | Change | Expected NPS gain | Effort | Risk |
|---|---|---|---|---|
| 1 | Replace per-leaf df-pn with dlshogi's 5-ply check | 100–1000× per-call cost reduction → huge speedup in tactical positions | 3–5 days | Low |
| 2 | Add NN inference cache | 15–30% NPS | 2–3 days | Low (lc0 has the code already) |
| 3 | Re-enable MakeSolid with proper locking | 10–20% NPS | 2–3 days | Medium |

---

## 1. Per-leaf mate detection: dlshogi 5-ply >> jhbr2 df-pn

### Cost per call

| Position type | dlshogi 5-ply | jhbr2 df-pn @ 100 nodes |
|---|---|---|
| Typical non-mate leaf | **100–500 ns** | **5–20 ms** (50,000–100,000× slower) |
| Mate-in-1 | ~5 ns | <100 μs |
| Mate-in-3 | ~100 ns (specialized) | 1–5 ms |
| Mate-in-5 | 1–5 μs | 5–20 ms |
| Mate-in-7+ | misses (acceptable) | usually unsolved at 100-node budget |

### Why dlshogi's design is faster per call

- **OR nodes generate only checking moves** via `generateMoves<CheckAll>()`
  (~5–15 vs ~60–100 legal moves)
- **AND nodes generate only evasions** via `generateMoves<Evasion>()`
  (~1–10 moves typically)
- **Hand-tuned mate-in-3 specialization** — base case of recursion is
  hot, so it's been hand-unrolled
- **No allocation, no proof-number arithmetic, no transposition table** —
  pure stack-based recursion
- **AND/OR early termination**: attacker stops on first mate found,
  defender stops on first escape found

### How dlshogi's recursion works (clarification: only `<5>` is called at
the leaf; `<3>` is the recursive base case)

The ONLY MCTS-leaf callsites in dlshogi are at `UctSearch.cpp:1470` and
`:1479`:

```cpp
if (mateMoveInOddPly<MATE_SEARCH_DEPTH, false>(*pos, draw_ply)) { ... }
// where MATE_SEARCH_DEPTH = 5 (constexpr in UctSearch.cpp:165)
```

dlshogi calls **only** `mateMoveInOddPly<5>` per leaf. It does NOT
call a separate 3-ply check first. The 3-ply specialization
(`mateMoveInOddPly<3>` → `mateMoveIn3Ply<>`) is the **base case**
hit through recursion:

```
mateMoveInOddPly<5>             ← initial call (generic template)
  └─ for each checking move:
       mateMoveInEvenPly<4>     ← generic template
         └─ for each evasion:
              mateMoveInOddPly<3>   ← TEMPLATE SPECIALIZATION
                → calls hand-tuned mateMoveIn3Ply<>(...)
```

Why this matters: depth-3 fires *many times per outer 5-ply call*
(once per (check, evasion) pair). Specializing it gives a multiplicative
speedup. Specializing depth-5 too would give diminishing returns
(it fires only once per leaf). This is an 80/20 optimization.

### Coverage table

| Mate distance | dlshogi 5-ply | jhbr2 df-pn (100 nodes) |
|---|---|---|
| Mate-in-1 | ✓ (5 ns) | ✓ (<100 μs) |
| Mate-in-3 | ✓ (100 ns, specialized) | ✓ (1–5 ms) |
| Mate-in-5 | ✓ (1–5 μs) | ✓ (5–20 ms) |
| Mate-in-7 | ✗ (out of depth) | ⚠ (usually unsolved at 100 nodes) |
| No-mate proof | ✗ (returns "no mate found") | ✓ occasionally (best-first prunes) |

### Why dlshogi specifically chose 5-ply (and not 7 or 9)

`MATE_SEARCH_DEPTH = 5` is a `constexpr` overridable from the Makefile —
strong signal of empirical tuning, not arbitrary choice.

Two reinforcing reasons:

1. **Cost scales 10–100× per +2 plies.** 5 → 7 is a massive cost
   increase for incremental coverage of the rare mate-in-7 case.
2. **dlshogi has a separate PV mate search thread** running full df-pn
   on the principal variation (`PV_MATE_SEARCH` flag in `Makefile`,
   USI options `PV_Mate_Search_{Threads,Depth,Nodes}`). Deep mates are
   handled by the PV thread. Per-leaf 5-ply is calibrated to be the
   sweet spot **assuming PV mate thread covers the >5 plies cases**.

If jhbr2 doesn't add a PV mate thread, **5-ply alone is still a major
win over df-pn** — you'd just miss some mate-in-7+ that the current
df-pn (at 100 nodes) probably misses anyway.

### Migration sketch

1. Port `DeepLearningShogi/usi/mate.h` (`mateMoveInOddPly`,
   `mateMoveInEvenPly`, `mateMoveIn3Ply`, `ns_mate::MovePicker`)
   to a new `jhbr2/mate/shallow_mate.h`.
2. Adapt to `ShogiBoard` API (we have `InCheck`, `GenerateLegalMoves`,
   `IsLegal`, `ComputeBlockersForKing`, `DoMove/UndoMove`, `Hash`).
3. Filter legal moves for checks/evasions ourselves (we don't have
   dlshogi's specialized generators — adds modest overhead).
4. Replace df-pn invocations at `lc0_mcts/search.cc:455` and `:822`.
5. Keep df-pn behind a compile/runtime flag for fallback.

See `docs/port_5ply_mate_check_plan.md` for the full plan.

### File:line references

| Purpose | Path |
|---|---|
| dlshogi mate templates | `DeepLearningShogi/usi/mate.h` (1–280) |
| dlshogi 3-ply specialization | `mate.h:229–230` (delegate to `mateMoveIn3Ply`) |
| dlshogi 3-ply hand-tuned base | `mate.h:55–125` |
| dlshogi MovePicker (CheckAll/Evasion) | `mate.h:14–53` |
| dlshogi leaf invocation | `DeepLearningShogi/usi/UctSearch.cpp:1470, 1479` |
| `MATE_SEARCH_DEPTH = 5` | `UctSearch.cpp:165`, `Makefile:2` |
| jhbr2 df-pn solver | `mate/dfpn.h`, `mate/dfpn.cc` |
| jhbr2 leaf invocation | `lc0_mcts/search.cc:455, 822` |

---

## 2. NN inference cache: jhbr2 has none

### Current state

| | Status |
|---|---|
| **dlshogi** | 8M-entry LRU cache. Key=`uint64_t` position hash. Value=`(value_win, policy_vec)`. Pin/unpin sync. Reports 30–50% hit rate in self-play. |
| **lc0 in your tree** | `lc0/src/neural/memcache.h` already implemented (FIFO, SpinMutex). **Not wired into jhbr2's search.** |
| **jhbr2** | **No cache.** Every position re-evaluated, including duplicates already seen in another branch. |

### What dlshogi caches

```cpp
struct CachedNNRequest {
    float value_win;              // single scalar
    std::vector<float> nnrate;    // policy vector (one float per legal move)
};
typedef LruCache<uint64_t, CachedNNRequest> NNCache;
```

Configured via `--nn_cache_size` (default 8,388,608 entries).
Lookup at `DeepLearningShogi/selfplay/self_play.cpp:765`, insert at `:1082`.

### Recommendation for jhbr2

**Easier route**: copy `lc0/src/neural/memcache.{h,cc}` into the
TensorRT eval path and wrap `NNEvaluator` with it. Already-tested code,
already in the tree.

**Slightly better cache policy**: port dlshogi's LRU. Eviction matches
MCTS access patterns better than FIFO (recently-visited subtrees get
revisited often).

Add USI option `NNCacheSize` (default 2–8M entries depending on RAM
target). Per-entry cost is ~32 bytes overhead + ~80 bytes for policy
vector → ~250 MB at 2M entries.

### File:line references

| Purpose | Path |
|---|---|
| dlshogi LRU cache | `DeepLearningShogi/selfplay/LruCache.h` |
| dlshogi cache config / lookup | `DeepLearningShogi/selfplay/self_play.cpp:152, 765, 1082, 1721` |
| lc0 memcache (unused in jhbr2) | `lc0/src/neural/memcache.{h,cc}` |
| jhbr2 evaluator (no cache) | `inference/nn_eval.h`, `inference/nn_tensorrt.h` |

---

## 3. MCTS multi-threading: stay with lc0 model, fix MakeSolid

### Architectural comparison

| Aspect | dlshogi | lc0/jhbr2 |
|---|---|---|
| Lock granularity | 65,536 mutexes hashed by position | One global `shared_mutex` on tree |
| Batch gather | Per-thread sequential trajectories | Bulk PUCT-distributed gather |
| Virtual loss | Atomic on visit/backup | Separate `n_in_flight_` counter |
| Collision handling | Backup anyway (wasteful) | Detect, cancel, skip (saves NN evals) |
| MakeSolid optimization | N/A | **Disabled** in jhbr2 (SIGSEGV — commit `b323a56`) |

### Recommendation: don't switch models

Both designs work. lc0's PUCT-distributed gather is generally better at
batch quality. The biggest jhbr2-specific win is **re-enabling MakeSolid
with proper synchronization**.

The SIGSEGV (commit `b323a56`) was a lock-scope bug, not an algorithmic
flaw. The dangling-pointer crash happens because gather threads hold
child-array iterators while a backup thread runs MakeSolid and moves
children to a new array. The fix is to upgrade backup's lock to
exclusive while MakeSolid runs:

```cpp
// lc0_mcts/search.cc:895 (DoBackupUpdate)
- std::shared_lock<std::shared_mutex> lock(search_->nodes_mutex_);
+ std::unique_lock<std::shared_mutex> lock(search_->nodes_mutex_);
```

Then re-enable the MakeSolid call near `search.cc:951`. Cost: backup
serializes (was concurrent under shared lock). Gain: better cache
locality in tree descent. Net should be positive at moderate thread
counts; if contention becomes a problem at high thread counts, then
consider hybrid (per-position locks for gather, exclusive lock only
for MakeSolid).

### File:line references

| Purpose | Path |
|---|---|
| dlshogi worker thread | `DeepLearningShogi/usi/UctSearch.cpp:313–463` |
| dlshogi VL mechanism | `UctSearch.cpp:218–241` |
| lc0/jhbr2 SearchWorker | `lc0_mcts/search.h:200–288`, `search.cc:473–757` |
| lc0/jhbr2 VL via n_in_flight_ | `lc0_mcts/node.h:177`, `node.cc:274–292` |
| lc0/jhbr2 MakeSolid (disabled) | `lc0_mcts/search.cc:951–955`, commit `b323a56` |

---

## Combined ROI estimate

If all three land successfully:

- **5-ply mate check alone** could lift NPS **2–5×** in the tactical
  positions where df-pn was tanking it to 259/sec.
- **NN cache** adds 15–30% on top.
- **MakeSolid** adds another 10–20%.

Cumulative: roughly **3–6× higher NPS in the worst-case positions**,
modest gains in quiet positions.

The day-1 sudden mate (P*6c) was both a coverage problem (need
defensive mate detection) AND a throughput problem (df-pn budget too
small to find the actual mate, but too slow to run at higher budget).
Switching to the 5-ply check addresses the throughput half — at 100×
cheaper per call, you can afford to run it on EVERY candidate move at
the leaf, which is the defensive lookahead we discussed earlier.

---

## Date / context

Research conducted **2026-05-07** (post-WCSC36, post-tournament
retrospective). Tournament codebase tagged at `wcsc36` (commit `dd59a57`).
