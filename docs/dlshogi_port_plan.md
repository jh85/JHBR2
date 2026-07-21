# Plan: port dlshogi's UctSearch into jhbr2 (Option B)

After two sessions of incremental refactoring of jhbr2's lc0-derived
MCTS toward dlshogi's concurrency model, the linked-list child
layout was identified as the structural bottleneck for multi-worker
scaling. Lock-free backup, atomic `n_in_flight_`, and bucket spawn
mutex all landed without producing the expected NPS gains.

Decision (this commit): **port dlshogi's `usi/UctSearch.{cpp,h}`
into jhbr2 wholesale**, mating it with our existing components
(multi-slot TRT, encoder, USI engine, shallow mate, opening book).

---

## What we keep from jhbr2

These are independent of the MCTS internals and have been validated:

- `inference/nn_tensorrt.{h,cc}` — direct TRT, multi-slot per GPU, per-worker
  `IExecutionContext` + CUDA stream. Already matches dlshogi's slot model.
- `inference/nn_eval.{h,cc}` — ORT fallback (rarely used).
- `shogi/board.h`, `shogi/encoder.{h,cc}` — board + plane encoder. Includes
  recent fixes: OUTE_SENNICHITE detection (`continuous_check_` increment by 2,
  2nd-occurrence detection), MaxMovesToDraw, ply() accessor, CanDeclareWin.
- `mate/shallow_mate.h`, `mate/dfpn.{h,cc}` — leaf-mate detection
  (5-ply shallow port complete, 51/51 tests passing).
- `usi/usi_engine.{h,cc}` — USI handler, watchdog, persistent search
  config push, opening book.
- `tools/benchmark.py` — NPS benchmark harness.

## What we replace

- `lc0_mcts/node.{h,cc}` — Node, Edge, Edge_Iterator, NodeTree.
- `lc0_mcts/search.{h,cc}` — Search, SearchWorker, PUCT logic.
- `lc0_mcts/backend.h` — Backend, Computation. Most of the routing
  logic stays; the dispatcher is already gone.
- `lc0_mcts/types.h` — types, GameResult.

## What we re-port from lc0 onto the dlshogi base

Re-add iteratively after the base port works:

1. **Persistent Search across `go` commands** + **tree reuse via 0/1/2-ply nav**
   — lc0_mcts/node.cc:NodeTree::ResetToPosition. Tested in production.
2. **Watchdog enforcement** — already lives in usi_engine.cc, just
   needs to call the new Search's Stop().
3. **PerLeafGathering** — dlshogi already does per-worker batching, so
   this is essentially the default behavior.
4. **VirtualLossWeight** — replace dlshogi's hardcoded `VIRTUAL_LOSS=1`
   with a configurable knob inside the PUCT formula's `n_started`
   denominator.
5. **MaxMovesToDraw / draw-by-limit** — wrap dlshogi's `nyugyoku()` and
   leaf eval to also check `pos.gamePly() > limit`.
6. **VirtualLossWeight option** — lc0-style amplification at contended
   nodes; deferred (low priority).
7. **Sticky endgames / bound propagation** — lc0 feature; re-port if
   needed for tactical strength.
8. **FPU at root + value FPU** — lc0 PUCT refinements; the dlshogi base
   uses dlshogi's PUCT formula. Decide whether to override per-knob
   or keep dlshogi's defaults (likely keep at first, A/B later).

## Migration steps

### Step 1: clean port of dlshogi's UctSearch

1. Create `dlshogi_mcts/` directory (parallel to `lc0_mcts/`).
2. Copy `DeepLearningShogi/usi/UctSearch.{cpp,h}` and `Node.h` into
   `dlshogi_mcts/`. Adjust includes and namespaces.
3. Adapt to our types:
   - `Position*` → `lczero::ShogiBoard*` (different API; need a shim
     or selective rewrite).
   - `MoveList<Legal>` → `board.GenerateLegalMoves()`.
   - `nyugyoku(pos)` → `board.CanDeclareWin()`.
   - `pos->gamePly()` → `board.ply()`.
   - `pos->getKey()` → `board.Hash()`.
   - dlshogi's encoder → our `EncodeShogiPosition`.
   - dlshogi's `nn_tensorrt`/`nn_onnxruntime` → our `inference/nn_tensorrt.h`.
4. Build target `dlshogi_mcts` as a static lib alongside `mcts`.
5. Add a build flag `USE_DLSHOGI_MCTS` (CMake option) to switch between
   `lc0_mcts` and `dlshogi_mcts` at link time.
6. USI engine selects between the two backends at startup.

### Step 2: adapt USI to dlshogi's option names (or keep ours via aliasing)

Map our existing options to dlshogi's where they overlap:

| Ours | dlshogi | Action |
|---|---|---|
| `Threads` / `WorkersPerGpu` | `UCT_Threads` | Keep our names; pass through. |
| `MinibatchSize` | `DNN_Batch_Size` | Keep our names. |
| `MaxNodes` | `UCT_NodeLimit` | Keep our names. |
| `OnnxModel` | `DNN_Model` | Keep ours. |
| `NumGPUs` | (per-GPU `Threads_GPUN`) | Map our `NumGPUs=N` to spawning N evaluator+searcher pairs. |
| `MaxGpuBatch` | (n/a) | Dropped; engine profile is authoritative. |

### Step 3: validate on Panda23

1. **Single-worker NPS** ≥ baseline (current ~430 NPS on test engine).
2. **W=2 NPS** > 1.5× W=1 NPS. dlshogi's published numbers say
   1×3090 + 2 workers ≈ 31k NPS, so this should be straightforward
   on the production engine.
3. **W=2 × 2 GPUs** ≥ 60k NPS (the original goal).

### Step 4: port back missing features

In priority order, only as needed:

1. Persistent search + tree reuse (large quality win on long games).
2. MaxMovesToDraw + nyugyoku integration.
3. Watchdog deadline enforcement.
4. VirtualLossWeight USI option.

### Step 5: deprecate lc0_mcts (completed)

`dlshogi_mcts` is the production backend. The opening book was moved to
`book/`, and the retired `lc0_mcts/` search implementation was removed.

## Effort estimate

- Step 1: 1–2 sessions (mostly mechanical translation, the hard part is
  the `Position` → `ShogiBoard` shim).
- Step 2: 0.5 session.
- Step 3: 0.5 session (depends on hardware availability).
- Step 4: 1 session per feature; persistent-search + tree-reuse is
  the biggest re-port.
- Total: 4–6 sessions.

## What to delete from this session's work

The following commits will be superseded by the dlshogi port and can be
considered exploratory/learning rather than production code:

- `156685e` — bucket spawn lock + shared_lock pickers (didn't scale).
- `ae3429a` — lock-free backup with per-node spinlocks (didn't scale).

These don't actively harm anything (single-worker still works) but they
add complexity without delivering the perf goal. Will be removed when
the dlshogi-based MCTS replaces lc0_mcts.

## What to keep from this session's work

- `d2b55ba` — per-worker GPU dispatch infrastructure. Stays useful.
- `c1a5fc2` — MaxMovesToDraw option.
- `e4c52f6` — OUTE_SENNICHITE fix (continuous_check += 2 + 2nd-occurrence).
- `f94b7f6` + `aa314bb` — VirtualLossWeight option + bulk-path coverage.
  (Bulk path will be removed; option carries forward to dlshogi base.)
- `5617a35` — historical benchmark.py --max-gpu-batch flag (removed).
- `564c7c5` — historical MaxGpuBatch USI option (removed).
- `db0797a` — atomic n_in_flight_ (carries forward; matches dlshogi).
