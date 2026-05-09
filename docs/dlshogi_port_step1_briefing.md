# Briefing: Step 1 of dlshogi UctSearch port

This document is the kickoff brief for a fresh agent session. The
goal is **step 1 of the migration plan in `docs/dlshogi_port_plan.md`**:
create a clean port of dlshogi's `UctSearch` into a new
`dlshogi_mcts/` directory, keeping it side-by-side with the existing
`lc0_mcts/` until validated.

Read first (in order):
1. This document.
2. `docs/concurrency_dlshogi_vs_jhbr2.md` — why we're doing this.
3. `docs/dlshogi_port_plan.md` — the full 5-step roadmap; you're
   doing step 1.

---

## Context (what got us here)

jhbr2 is a Shogi engine using lc0-derived MCTS. Per-worker GPU
dispatch already works (`mcts/nn_tensorrt.cc` has multi-slot per
GPU, matching dlshogi's `IExecutionContext`-per-worker model).
However, multi-worker NPS does NOT scale: W=2 is *worse* than W=1.

We confirmed empirically (see commit history `156685e`, `ae3429a`)
that the bottleneck is the **linked-list child layout** in
`lc0_mcts/node.h`. Lock-free backup, atomic `n_in_flight_`, and
bucket spawn mutex all landed without delivering scaling. The
`Edge_Iterator::Actualize` walk of the parent's `child_/sibling_`
chain races with concurrent `GetOrSpawnNode` inserts; cache-line
ping-pong on the same memory between cores is the suspected
(untested) cause of the negative scaling.

dlshogi solves this structurally with **per-edge stats in a flat
pre-allocated array** (`child_node_t[]`), with full subtree Nodes
allocated lazily only when needed. This eliminates the spawn race
on the hot path and lets backup run lock-free with simple
`atomic_fetch_add`.

**Decision:** port dlshogi's `UctSearch.{cpp,h}` and `Node.h`
wholesale, mating with our existing components. Rejected option
A (full copy) loses lc0 algorithm features; rejected option B
(in-place refactor of jhbr2) hit memory blowup issues with
flat-Node-array approach.

---

## What you keep from jhbr2 (do NOT touch these)

- `mcts/nn_tensorrt.{h,cc}` — multi-slot direct TRT. Per-worker
  `IExecutionContext` + CUDA stream + buffers. Already correct.
- `mcts/nn_eval.{h,cc}` — ORT fallback path.
- `shogi/board.{h,cc}`, `shogi_engine/` — board representation,
  move generation, encoder. Includes recent fixes: OUTE_SENNICHITE
  detection (`continuous_check_` += 2, 2nd-occurrence detect),
  `MaxMovesToDraw`, `ply()` accessor, `CanDeclareWin`.
- `shogi/encoder.{h,cc}` — 48-plane input encoder.
- `mate/shallow_mate.h`, `mate/dfpn.{h,cc}` — leaf mate detection.
- `usi/usi_engine.{h,cc}` — USI handler, watchdog, persistent
  search config push, opening book.
- `tools/benchmark.py` — NPS benchmark harness.

## What you create in this step

A **new directory** `dlshogi_mcts/` containing:

1. `dlshogi_mcts/uct_node.{h,cc}` — port of
   `DeepLearningShogi/cppshogi/Node.h` and any `.cpp` it ships
   with. Contains `child_node_t`, `uct_node_t`.
2. `dlshogi_mcts/uct_search.{h,cc}` — port of
   `DeepLearningShogi/usi/UctSearch.{h,cpp}`. Contains
   `UCTSearcher`, `UCTSearcherGroup`, the search loop, PUCT,
   backup, the bucket-hashed position mutex array.
3. `dlshogi_mcts/types.h` — any shared enums (Color, GameResult,
   etc.) needed by uct_search but not in our existing types.
4. `dlshogi_mcts/board_shim.{h,cc}` — adapter functions translating
   between dlshogi's `Position*` API and our `lczero::ShogiBoard`.
   This is the trickiest file; details below.

## What you DON'T do in this step

- Do NOT delete `lc0_mcts/`. Both backends will coexist behind a
  build flag.
- Do NOT add a USI option to switch backends yet — handle that in
  step 2.
- Do NOT re-port lc0 features (persistent search, tree reuse,
  VirtualLossWeight, MaxMovesToDraw enforcement). Those are
  step 4. Use dlshogi's defaults for now.
- Do NOT integrate shallow mate. dlshogi has its own mate hooks;
  leave them as dlshogi has them. We can swap in our shallow mate
  later if it's faster.

## Build flag scheme

Add to `CMakeLists.txt`:
```cmake
option(USE_DLSHOGI_MCTS "Use dlshogi-ported MCTS instead of lc0_mcts" OFF)
if(USE_DLSHOGI_MCTS)
  add_subdirectory(dlshogi_mcts)
  target_link_libraries(jhbr2 PRIVATE dlshogi_mcts)
  target_compile_definitions(jhbr2 PRIVATE USE_DLSHOGI_MCTS=1)
else()
  target_link_libraries(jhbr2 PRIVATE mcts)  # current lc0_mcts
endif()
```

In `usi/usi_engine.cc`:
```cpp
#ifdef USE_DLSHOGI_MCTS
  #include "dlshogi_mcts/uct_search.h"
  using SearchT = dlshogi_shogi::UCTSearcher;
#else
  #include "lc0_mcts/search.h"
  using SearchT = lc0_shogi::Search;
#endif
```

This way both compile and we can A/B by rebuilding.

---

## The board shim — the actual hard part

dlshogi's code uses its own `Position` class with API like:

```cpp
pos->getKey()                        // 64-bit zobrist
pos->gamePly()                       // current ply
pos->moveList<Legal>()               // legal moves
pos->turn()                          // side to move
pos->doMove(m), pos->undoMove(m)     // make/unmake
pos->inCheck()                       // is in check
nyugyoku(pos)                        // can declare win
```

Our `lczero::ShogiBoard` has:
```cpp
board.Hash()
board.ply()
board.GenerateLegalMoves()
board.side_to_move()
board.DoMove(m), board.UndoMove(m, undo)
board.InCheck(color)
board.CanDeclareWin()
```

Move types are different too: dlshogi has `Move` struct with bit
encoding; we have `lczero::Move`. The shim either:

**Option A: Adapt dlshogi's code to call ShogiBoard directly.**
Find/replace `pos->X()` with `board.Y()` throughout the ported
code. Less abstraction, more churn in the port.

**Option B: Write a `Position` class that wraps `ShogiBoard`.**
dlshogi's UctSearch keeps using `Position*`; the wrapper forwards
to ShogiBoard internally. Cleaner if more code to write upfront.

**Recommendation: A.** dlshogi's Position API is small enough
(maybe 15 methods called from UctSearch) that find/replace is
faster than building a wrapper, and the ported code reads more
naturally to a future maintainer.

For move encoding, expect to write a small converter:
```cpp
lczero::Move FromDlshogiMove(uint16_t dl_move);
uint16_t ToDlshogiMove(lczero::Move m);
```
Both encodings are 16-bit-ish; the bit layouts differ. Spend time
here getting it right — silent move corruption will produce a
mysteriously-broken engine.

---

## NN feature encoding

dlshogi has its own input encoder in `cppshogi/cppshogi.cpp`. It's
roughly equivalent to our 48-plane encoder but with different plane
ordering and packed/unpacked feature variants.

For step 1, **don't port dlshogi's encoder.** Use our
`EncodeShogiPosition` from `shogi/encoder.h`. The TRT engine was
trained against our encoder layout; using dlshogi's encoder would
require retraining.

This means the place in dlshogi's code where features get packed
(`make_input_features` in `cppshogi.cpp`) gets replaced with a call
to our encoder, and the resulting `ShogiInputPlanes` get copied
directly to the TRT slot's pinned host buffer.

## TRT integration

Our `mcts/nn_tensorrt.h` already exposes:
```cpp
class NNEvaluator {
  NNEvaluator(string engine_path, bool use_gpu, int device_id, int max_batch, int num_slots);
  std::vector<NNOutput> EvaluateBatchSlot(int slot_id, const batch_t&);
  int num_slots();
};
```

dlshogi's `nn_forward(slot_id, batch_size, x1, x2, y1, y2)` maps
directly to our `EvaluateBatchSlot(slot_id, batch)`. The big
difference: dlshogi works in raw float arrays; we work in
`vector<pair<ShogiBoard, MoveList>>`. The shim converts the worker's
accumulated leaves into our batch format before calling
`EvaluateBatchSlot`.

---

## Validation gates for step 1 (acceptance criteria)

1. **Builds.** `cmake -DUSE_DLSHOGI_MCTS=ON ..` → `make jhbr2` succeeds.
   Also `cmake -DUSE_DLSHOGI_MCTS=OFF ..` (default) keeps building.
2. **USI handshake works.** `usi`, `setoption ...`, `isready`,
   `position startpos`, `go byoyomi 1000` all behave; engine prints
   a `bestmove` line.
3. **Single-position correctness smoke test.** A few hand-crafted
   tactical positions with known best moves. The new backend should
   pick the same move (or a same-evaluation move) as the lc0
   backend. Doesn't have to be byte-identical.
4. **Single-worker NPS** ≥ existing baseline (~430 NPS on the
   `shogi_bt4_epoch13_b128.engine` test engine).
5. **Multi-worker NPS scales.** `WorkersPerGpu=2` should give
   noticeably more than W=1 (target: ≥1.5×). This is the whole
   point of the port — if this gate fails, something is wrong with
   the port, not with dlshogi's design.

Don't worry about Panda23 / 60 KNPS in this step. That's the
acceptance for step 3 in the master plan.

---

## Things that bit me last session — avoid these

- **`std::atomic<T>` deletes implicit move ops.** If you put atomics
  in a struct that gets `std::move`-assigned anywhere, you'll need
  explicit move ops that load+store the atomic. We hit this in
  `Node` when `NodeTree::TrimTreeAtHead` did `*current_head_ =
  Node(...)`. dlshogi uses `unique_ptr<uct_node_t>` widely; you
  may not see this bug if you keep their layout.
- **`std::atomic_flag` with `inline std::array` in a header has subtle
  initialization order issues** if the array is large and the flags
  default to set instead of clear. Use `static` inside an inline
  function (Meyers singleton pattern) or just make sure the array
  initializes to clear.
- **`shogi/` is a symlink to `shogi_engine/`** in our repo. `git add
  shogi/board.h` fails with "beyond a symbolic link" — use the real
  path `shogi_engine/board.h`.
- **TRT engine cache files** are keyed by file path under
  `trt_cache/`. Changing input plane count or max batch invalidates
  the cache (will rebuild on first run, takes a few minutes).
  Cache key already includes `gpu{i}_b{maxbatch}`; if you change
  encoder plane count later, add plane-count to the cache key too.
- **dlshogi assumes `VIRTUAL_LOSS=1`** for non-atomic `move_count`
  increments to be safe. If you ever consider changing this
  constant, also audit those increments — they need to become
  atomic.
- **Don't add Co-Authored-By trailers to commit messages** (user
  preference).

## Repo layout reference

```
/home/ei/Downloads/JHBR2/
├── DeepLearningShogi/         # dlshogi source (read-only, your reference)
│   ├── usi/UctSearch.{h,cpp}  # ← port from
│   ├── usi/main.cpp           # ← USI loop reference
│   └── cppshogi/Node.h        # ← port from
├── shogi_engine/              # our board (real path; shogi/ is a symlink)
├── lc0_mcts/                  # current MCTS (don't touch in this step)
├── mcts/                      # NN evaluator (keep, don't touch)
├── usi/                       # USI engine (touch only for build flag)
├── docs/                      # planning docs
│   ├── concurrency_dlshogi_vs_jhbr2.md
│   ├── dlshogi_port_plan.md
│   └── dlshogi_port_step1_briefing.md  # ← this file
└── dlshogi_mcts/              # ← what you create
```

## When to commit

- After the directory + skeleton files compile (even as stubs).
- After the board shim works (you can call `pos->getKey()` and get
  the right zobrist hash from a `ShogiBoard`).
- After the search loop runs end-to-end and produces a `bestmove`.
- After multi-worker scaling is verified.

Small commits are easier to bisect later if something regresses.

## Out of scope reminders

- One-hot nyugyoku NN features (62 planes) — separate track,
  requires retraining.
- `MakeSolid` / solid-children optimization — deferred.
- Persistent Search across `go` commands and tree reuse — step 4.
- Watchdog deadline enforcement — step 4 (lives in usi_engine.cc,
  needs to call new Search's `Stop()`).

---

## When in doubt

- If something in dlshogi's code is unclear, read the comment or
  ask the user. Don't guess move encodings — verify against
  `cppshogi/move.hpp`.
- If port behavior diverges from lc0 backend, run both side-by-side
  on the same position and look at the output `info` lines. Often
  reveals a bug fast.
- If you find structural problems with the port plan itself,
  push back rather than papering over them. The user is engaged
  and prefers honest assessments.

Good luck.
