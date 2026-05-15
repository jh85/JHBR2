# Technical Design: dlshogi-style MCTS for jhbr2

This document supersedes the mechanical-port plan in
`docs/dlshogi_port_step1_briefing.md`. After completing the
mechanical step 1, ~440 lines of stub baggage accumulated (stub
files for `color.hpp`, `piece.hpp`, `score.hpp`, `position.hpp`,
`generateMoves.hpp`, `move.hpp`, etc.) — code that exists only to
satisfy dlshogi's includes against types we don't have.

**New approach:** design the MCTS module in jhbr2's terms, then
implement it from scratch against jhbr2's existing types
(`ShogiBoard`, `Move`, `MoveList`, etc.). Use dlshogi's
`uct_search.cpp` and `uct_node.h` as a **reference** for the
algorithm — read it carefully, mirror the data structures and
control flow, but write code that fits naturally in jhbr2.

This document is the "what to build." A separate agent session
implements it.

---

## 1. Goal

Replace `lc0_shogi::Search::Run()` with a dlshogi-architecture
MCTS that:

- Pre-allocates per-edge stats in a flat array (`child_node_t[]`)
  at expansion time.
- Lazily allocates per-position subtree nodes (`uct_node_t`) only
  when descended into.
- Uses 65 536-bucket position-hashed mutex (no global tree mutex).
- Lock-free backup via atomic per-edge stats.
- Per-worker direct GPU submission (no central dispatcher).

Target NPS: **30–40k median on Panda23 (2× 3090)**, with
acceptable variance (p10 ≥ 25k). dlshogi reaches 60k median; the
gap is due to BT4 model size, not architecture.

## 2. What we keep from jhbr2

| Component | File | Why |
|---|---|---|
| Board representation | `shogi/board.{h,cc}` | All move-gen, repetition, mate-by-move-limit, nyugyoku. Tested. |
| Move type | `shogi/types.h` | Already used everywhere. |
| Encoder | `shogi/encoder.{h,cc}` | 48-plane input matched to current TRT engine. |
| NN evaluator | `mcts/nn_tensorrt.{h,cc}` | Multi-slot direct TRT, already correct. |
| Shallow mate | `mate/shallow_mate.h` | Optional leaf mate (replaces dlshogi's df-pn at leaf). |
| USI handler | `usi/usi_engine.{h,cc}` | All USI plumbing, watchdog, tree reuse, MaxMovesToDraw, VirtualLossWeight. |
| Opening book | `lc0_mcts/book.{h,cc}` | Move out of `lc0_mcts/` to a top-level `book/` directory. |
| Benchmark | `tools/benchmark.py` | Already produces measurable numbers. |

## 3. What we replace

| Old | New |
|---|---|
| `lc0_mcts/search.{h,cc}` | `dlshogi_mcts/uct_search.{h,cc}` (clean reimplementation) |
| `lc0_mcts/node.{h,cc}` | `dlshogi_mcts/uct_node.{h,cc}` (clean reimplementation) |
| `lc0_mcts/backend.h` | (removed — direct evaluator calls) |
| `lc0_mcts/types.h` | merged into `uct_node.h` or kept minimal in `dlshogi_mcts/types.h` |

Eventually `lc0_mcts/` is deleted entirely once dlshogi_mcts is
verified working.

## 4. Module file structure (clean target)

```
dlshogi_mcts/
├── CMakeLists.txt
├── types.h           # Result/Color enums and helpers (small)
├── uct_node.h        # uct_node_t, child_node_t, NodeTree
├── uct_node.cc       # NodeTree::ResetToPosition (tree reuse)
├── uct_search.h      # UCTSearcher, UCTSearcherGroup, public API
└── uct_search.cc     # PUCT, backup, ParallelUctSearch loop
```

Six files, ~1500 lines total. No stubs. All types are jhbr2's.

## 5. Data structures

### 5.1 `child_node_t` — per-edge statistics

```cpp
struct child_node_t {
  lczero::Move move;             // 4 bytes
  float nnrate;                  // policy prior from NN
  std::atomic<int> move_count;   // visit count, atomic
  std::atomic<float> win;        // accumulated win value, atomic via CAS

  // Win/lose/draw flags packed in unused bits of move.value()
  // (mirror dlshogi's trick if jhbr2's Move has spare bits;
  // otherwise add a separate uint8_t flags_).
  bool IsWin()  const;
  bool IsLose() const;
  bool IsDraw() const;
  void SetWin();
  void SetLose();
  void SetDraw();
};
```

Allocated as a flat array, one entry per legal move at expansion.

### 5.2 `uct_node_t` — per-position node

```cpp
struct uct_node_t {
  std::atomic<int>   move_count{NOT_EXPANDED};   // sentinel for "not yet evaluated"
  std::atomic<float> win{0};
  std::atomic<float> visited_nnrate{0};
  short              child_num{0};
  std::unique_ptr<child_node_t[]>             child;        // edge stats
  std::unique_ptr<std::unique_ptr<uct_node_t>[]> child_nodes; // lazy subtrees

  bool IsEvaled() const;
  void SetEvaled();
  void ExpandNode(const lczero::ShogiBoard* board);
  uct_node_t* CreateChildNode(int i);     // lazy subtree alloc
  void InitChildNodes();
  uct_node_t* ReleaseChildrenExceptOne(lczero::Move m);
};
```

`ExpandNode(board)` calls `board->GenerateLegalMoves()`, allocates
`child[]` for each move with `nnrate=0` (set later by NN result).

### 5.3 `NodeTree` — game tree with reuse

```cpp
class NodeTree {
 public:
  // Try to reuse subtree from previous game state. Returns true if
  // navigated to new position via existing tree, false if rebuilt.
  bool ResetToPosition(uint64_t starting_pos_key,
                       const std::vector<lczero::Move>& moves);
  uct_node_t* GetCurrentHead() const;
  void DeallocateTree();

 private:
  uct_node_t* current_head_ = nullptr;
  std::unique_ptr<uct_node_t> gamebegin_node_;
  uint64_t history_starting_pos_key_ = 0;
};
```

Mirrors dlshogi's `NodeTree`. Already exists conceptually in
jhbr2 (`lc0_mcts/node.cc:NodeTree::ResetToPosition`); reimplement
in the new layout.

### 5.4 `visitor_t` — per-playout state

```cpp
struct trajectory_t {
  uct_node_t* parent;
  unsigned    child_idx;   // which edge from parent
};

struct visitor_t {
  std::vector<trajectory_t> trajectories;  // path from root to leaf
  float value_win;
};
```

Used to record the path of one playout for backup. Pre-allocated
in a pool of size `MinibatchSize` per worker.

### 5.5 `batch_element_t` — per-leaf NN-eval queue entry

```cpp
struct batch_element_t {
  uct_node_t* node;
  Color       color;     // side to move at this leaf
};
```

A worker accumulates these until batch is full, then submits to
NN. Result fills in `node->child[i].nnrate` for each edge and
`visitor.value_win` for backup.

## 6. Components

### 6.1 `UCTSearcherGroup` — one per GPU

```cpp
class UCTSearcherGroup {
 public:
  void Initialize(int threads, int gpu_id, int batch_max);
  void Run();        // launch all worker threads
  void Join();       // wait for completion
  void Term();       // permanent shutdown
  void InitGPU();    // first-call CUDA setup (cudaSetDevice etc.)

 private:
  int gpu_id;
  int threads;
  jhbr2::NNEvaluator* nn;     // shared by all searchers in group
  std::vector<UCTSearcher> searchers;
};
```

Owns one NNEvaluator (one TRT engine) on its assigned GPU. Each
searcher in the group uses one slot of that evaluator. Mirrors
dlshogi's `UCTSearcherGroup`.

### 6.2 `UCTSearcher` — one per worker thread

```cpp
class UCTSearcher {
 public:
  UCTSearcher(UCTSearcherGroup* grp, int thread_id, int batch_max);
  void Run();                  // spawn thread → ParallelUctSearch
  void Join();                 // wait for thread

 private:
  void ParallelUctSearch();    // main worker loop
  float UctSearch(lczero::ShogiBoard* board, child_node_t* parent,
                  uct_node_t* current, visitor_t& visitor);
  unsigned SelectMaxUcbChild(child_node_t* parent, uct_node_t* current);
  void QueuingNode(const lczero::ShogiBoard* board, uct_node_t* node,
                   float* value_win);
  void EvalNode();             // single GPU call for accumulated batch

  UCTSearcherGroup* grp;
  int thread_id;
  int batch_max;
  int current_batch_index;
  std::vector<batch_element_t> policy_value_batch;
  std::vector<float> nn_outputs_policy_buf;  // reused per call
  std::vector<float> nn_outputs_value_buf;
  std::unique_ptr<std::mt19937_64> rng;
  std::thread handle;
};
```

Each searcher has its own batch buffer. `EvalNode` calls
`grp->nn->EvaluateBatchSlot(thread_id, batch)` directly — no
shared queue, no central dispatcher. Multiple searchers on the
same GPU run concurrently on separate slots (= separate TRT
contexts + CUDA streams).

### 6.3 65k-bucket position mutex

```cpp
constexpr uint64_t MUTEX_NUM = 65536;  // 2^16
extern std::mutex mutexes[MUTEX_NUM];

inline std::mutex& GetPositionMutex(const lczero::ShogiBoard* board) {
  return mutexes[board->Hash() & (MUTEX_NUM - 1)];
}

#define LOCK_EXPAND mutexes[0].lock()    // global expand lock
#define UNLOCK_EXPAND mutexes[0].unlock()
```

Held during PUCT walk + leaf expansion at each position.
Different positions → different bucket → no contention.

`LOCK_EXPAND` is a single global lock used only during the
*initial* root-node expansion (rare; once per `go`).

## 7. Algorithm: ParallelUctSearch (the worker loop)

Pseudocode mirroring dlshogi `uct_search.cpp:1082`:

```cpp
void UCTSearcher::ParallelUctSearch() {
  uct_node_t* root = tree->GetCurrentHead();
  
  // First-time root evaluation (one worker does this; others wait)
  LOCK_EXPAND;
  if (!root->IsEvaled()) {
    QueuingNode(pos_root, root, &dummy);
    EvalNode();
  }
  UNLOCK_EXPAND;

  // Main batch loop: keep gathering and evaluating until told to stop
  std::vector<visitor_t> visitor_pool(batch_max);
  std::vector<visitor_t*> visitor_batch;
  std::vector<trajectory_t*> discarded_trajectories;

  while (!IsUctSearchStoped()) {
    visitor_batch.clear();
    discarded_trajectories.clear();
    current_batch_index = 0;

    // Inner: do batch_max playouts, accumulating NN evals
    for (int i = 0; i < batch_max; ++i) {
      lczero::ShogiBoard board(*pos_root);     // copy root board
      visitor_pool[i].trajectories.clear();
      
      float result = UctSearch(&board, nullptr, root, visitor_pool[i]);
      
      if (result != DISCARDED) atomic_fetch_add(&playout_count, 1);
      
      if (result == DISCARDED)
        discarded_trajectories.push_back(&visitor_pool[i].trajectories);
      else if (result == QUEUING)
        visitor_batch.push_back(&visitor_pool[i]);
      // else: terminal value already known, no batch entry needed
    }

    EvalNode();   // ONE GPU call for all queued positions

    // Roll back virtual loss for discarded paths
    for (auto* traj : discarded_trajectories)
      for (auto& step : reverse(*traj))
        SubVirtualLoss(&step.parent->child[step.child_idx], step.parent);

    // Backup actual values for completed paths
    for (auto* visitor : visitor_batch) {
      float value = 1.0f - visitor->value_win;
      for (auto& step : reverse(visitor->trajectories)) {
        UpdateResult(&step.parent->child[step.child_idx], value, step.parent);
        value = 1.0f - value;   // flip perspective each ply
      }
    }

    // One thread monitors time / output info
    if (monitoring_thread) UpdateInfoOutput();
  }
}
```

## 8. Algorithm: UctSearch (one playout)

```cpp
float UCTSearcher::UctSearch(lczero::ShogiBoard* board,
                             child_node_t* parent,
                             uct_node_t* current,
                             visitor_t& visitor) {
  // Terminal cases
  if (board->CanDeclareWin())   return 1.0f;
  if (parent && parent->IsWin()) return 0.0f;  // we'd be losing
  if (parent && parent->IsLose()) return 1.0f; // we'd be winning
  if (parent && parent->IsDraw()) return draw_value;

  // Repetition check
  auto rep = board->CheckRepetition();
  if (rep == kLoss) { parent->SetLose(); return 1.0f; }
  if (rep == kWin)  { parent->SetWin();  return 0.0f; }
  if (rep == kDraw) { parent->SetDraw(); return draw_value; }

  // Move-limit draw
  if (board->ply() > max_moves_to_draw) { return draw_value; }

  // Need to expand?
  std::lock_guard<std::mutex> lk(GetPositionMutex(board));
  if (!current->IsEvaled()) {
    if (parent && current was already queued by another worker)
      return DISCARDED;
    
    // Optionally call shallow mate at this leaf (jhbr2 feature)
    if (CheckLeafMate(*board)) {
      parent->SetWin();   // current side mates → opponent's edge to me wins
      return 0.0f;
    }
    
    current->ExpandNode(board);
    QueuingNode(board, current, &visitor.value_win);
    return QUEUING;
  }
  // (lock released as we exit the if-block — implicit by RAII)

  // Select best child via PUCT
  unsigned next = SelectMaxUcbChild(parent, current);
  
  AddVirtualLoss(&current->child[next], current);
  visitor.trajectories.push_back({current, next});
  
  board->DoMove(current->child[next].move);
  
  // Lazily allocate child subtree if needed
  uct_node_t* next_node = current->child_nodes[next].get();
  if (next_node == nullptr)
    next_node = current->CreateChildNode(next);
  
  // Recurse
  float value = UctSearch(board, &current->child[next], next_node, visitor);
  
  if (value == 1.0f - QUEUING) return value;  // unwound
  return 1.0f - value;  // flip for parent's perspective
}
```

## 9. PUCT: SelectMaxUcbChild

Standard formula (mirror dlshogi `uct_search.cpp:1404`):

```cpp
unsigned SelectMaxUcbChild(child_node_t* parent, uct_node_t* current) {
  const float sqrt_sum = std::sqrt(static_cast<float>(current->move_count));
  const float c = c_init + std::log((current->move_count + c_base) / c_base);
  const float fpu = (parent ? c_fpu_reduction : c_fpu_reduction_root)
                    * std::sqrt(visited_nnrate);

  float best_score = -INFINITY;
  unsigned best = 0;
  for (int i = 0; i < current->child_num; ++i) {
    auto& ch = current->child[i];
    if (ch.IsLose()) return i;  // we win by going here
    if (ch.IsWin()) continue;   // skip — we'd lose
    if (ch.IsDraw() && draw_score_for_us < ...) continue;
    
    int n = ch.move_count;
    float q = (n == 0) ? -fpu : (ch.win / n);
    float u = c * sqrt_sum * ch.nnrate / (1.0f + n);
    float score = q + u;
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return best;
}
```

Plus virtual_loss_weight knob (jhbr2 feature) applied to in-flight
visits in the U denominator if you want to keep that.

## 10. NN integration

Replace dlshogi's `nn->forward(slot, batch_size, x1, x2, y1, y2)`
with our `EvaluateBatchSlot`:

```cpp
void UCTSearcher::EvalNode() {
  if (current_batch_index == 0) return;

  // Build the batch in jhbr2's format
  std::vector<std::pair<lczero::ShogiBoard, lczero::MoveList>> batch;
  batch.reserve(current_batch_index);
  for (int i = 0; i < current_batch_index; ++i) {
    auto& elem = policy_value_batch[i];
    // We saved board snapshots OR can reconstruct from trajectories
    // (tradeoff: store boards in batch_element_t, or replay moves)
    batch.emplace_back(elem.board, elem.legal_moves);
  }

  // Direct evaluator call — no shared queue
  auto results = grp->nn->EvaluateBatchSlot(thread_id, batch);

  // Distribute results back to the queued nodes
  for (int i = 0; i < current_batch_index; ++i) {
    auto& elem = policy_value_batch[i];
    auto& result = results[i];
    
    // Set edge nnrates
    for (int j = 0; j < elem.node->child_num; ++j)
      elem.node->child[j].nnrate = result.policy[j];
    
    // Apply softmax temperature if configured
    softmax_temperature_with_normalize(elem.node->child, elem.node->child_num);
    
    // Store value for the queuing visitor (via batch_element back-ref)
    *elem.value_win_out = (elem.color == BLACK) ? result.value : -result.value;
    
    elem.node->SetEvaled();
  }
}
```

`batch_element_t` will need a `lczero::ShogiBoard board`,
`lczero::MoveList legal_moves`, and `float* value_win_out` to glue
the queue to the playout state.

## 11. Backup: UpdateResult

```cpp
inline void UpdateResult(child_node_t* child, float result, uct_node_t* current) {
  atomic_fetch_add(&current->win, result);   // CAS loop for float
  current->move_count.fetch_add(1 - VIRTUAL_LOSS, std::memory_order_acq_rel);
  atomic_fetch_add(&child->win, result);
  child->move_count.fetch_add(1 - VIRTUAL_LOSS, std::memory_order_acq_rel);
}
```

If `VIRTUAL_LOSS == 1`, `move_count` adjustments are no-ops (the
`+= 1 - 1` cancels the virtual loss that AddVirtualLoss applied).

`atomic_fetch_add` for `float`/`double` doesn't exist natively;
implement as CAS loop:

```cpp
template<typename T>
inline void atomic_fetch_add(std::atomic<T>* obj, T arg) {
  T expected = obj->load(std::memory_order_relaxed);
  while (!obj->compare_exchange_weak(expected, expected + arg)) {}
}
```

## 12. Integration with usi_engine.cc

### 12.1 CmdIsReady

```cpp
void USIEngine::CmdIsReady() {
  if (evaluators_.empty()) {
    ShogiEncoderTables::Init();
    for (int g = 0; g < num_gpus_; g++) {
      evaluators_.push_back(std::make_unique<NNEvaluator>(
          onnx_path_, use_gpu_, g, max_gpu_batch_, workers_per_gpu_));
    }
    
    // Initialize MCTS infrastructure
    InitializeUctSearch(max_nodes_);   // sets uct_node_limit etc.
    SetThread(workers_per_gpu_, batch_size_);  // creates UCTSearcherGroups
  }
  Send("readyok");
}
```

### 12.2 CmdPosition

```cpp
void USIEngine::CmdPosition(const std::vector<std::string>& parts) {
  // Same as today: build board_, replay moves
  board_ = ShogiBoard();
  ...
  for (each move) board_.DoMove(move);
  
  // Reset tree to this position (try to reuse subtree)
  std::vector<lczero::Move> moves_played;  // from start to current
  ...build moves_played from tokens...
  tree->ResetToPosition(starting_pos_key_, moves_played);
}
```

`tree` is the global `NodeTree` from `dlshogi_mcts`.

### 12.3 CmdGo

```cpp
void USIEngine::CmdGo(...) {
  // Compute time limit, set in UCT module:
  SetLimits(&board_, parsed_limits);
  
  // Optionally enable random move at low ply:
  // (jhbr2 doesn't currently use this; can defer)
  
  // Run the search (this blocks until done):
  Move ponder_move;
  Move best_move = UctSearchGenmove(&board_, starting_pos_key_,
                                    moves_played, ponder_move);
  
  Send("bestmove " + best_move.ToUSI());
}
```

`UctSearchGenmove`:
1. Calls `ExpandRoot(&board)` — evaluates root if not yet.
2. Loops `search_groups[i].Run()` for each GPU.
3. Loops `search_groups[i].Join()` to wait.
4. Picks best move via visit-count or value (with optional
   tiebreak by score).
5. Returns the best move.

### 12.4 Watchdog (jhbr2 feature, keep)

The existing watchdog in `usi_engine.cc:CmdGo` calls
`Search::Stop()`. Replace with `TerminateUctSearch()` which sets
the `uct_search_stop` atomic flag the workers check.

## 13. Concrete implementation steps

For the implementing agent, in order:

1. **types.h** (~50 lines). Result enum, draw constants, etc.

2. **uct_node.h, uct_node.cc** (~300 lines). `child_node_t`,
   `uct_node_t`, `NodeTree`. Mirror dlshogi's structures. Test
   manually: create a node, call ExpandNode on startpos, verify
   `child_num` matches expected legal-move count.

3. **uct_search.h** (~150 lines). `UCTSearcher`,
   `UCTSearcherGroup`, public API surface (`InitializeUctSearch`,
   `SetThread`, `SetLimits`, `UctSearchGenmove`,
   `TerminateUctSearch`).

4. **uct_search.cc** (~800 lines). Implementation in this order:
   a. Globals: `mutexes[]`, `tree`, search-control atomics.
   b. `AddVirtualLoss`, `SubVirtualLoss`, `UpdateResult`.
   c. `SelectMaxUcbChild` (PUCT).
   d. `UctSearch` (one playout, recursive).
   e. `QueuingNode`, `EvalNode` (NN integration).
   f. `ParallelUctSearch` (the worker loop).
   g. `UCTSearcherGroup::Run/Join/Term`.
   h. `UctSearchGenmove`, `ExpandRoot`.
   i. `SetLimits`, time management.
   j. `InterruptionCheck` (when to stop searching).

5. **CMakeLists.txt** for `dlshogi_mcts/`. Single static library
   linking against jhbr2's `shogi/`, `mcts/`, `mate/`. ~30 lines.

6. **Wire into usi_engine.cc** behind `USE_DLSHOGI_MCTS` cmake
   flag. Both backends compile; A/B by rebuild.

7. **Smoke test**. `usi`, `setoption`, `position startpos`, `go
   byoyomi 3000`. Should produce `bestmove`.

8. **NPS measurement** on Panda23.

Each step is a commit. Build success per step is the gate.

## 14. What to copy verbatim from dlshogi

These pieces benefit from staying close to dlshogi's exact code:

- The PUCT formula (constants, FPU reduction).
- The bucket mutex array size (65 536) and hash function.
- The `UpdateResult` / virtual loss accounting.
- The trajectory recording shape.
- The atomic_fetch_add CAS loop helper.

Copying these verbatim minimizes the risk of introducing
algorithmic bugs that differ from dlshogi's tested behavior.

## 15. What to write fresh for jhbr2

- Anything touching `Position*` → use `lczero::ShogiBoard*`.
- Anything touching `Move` → use `lczero::Move`.
- Anything touching `MoveList<Legal>` → use
  `board.GenerateLegalMoves()`.
- `nyugyoku()` → `board.CanDeclareWin()`.
- `pos->getKey()` → `board.Hash()`.
- `pos->gamePly()` → `board.ply()`.
- NN encoder calls → `EncodeShogiPosition`.
- NN forward → `evaluator->EvaluateBatchSlot`.
- All `cudaHostAlloc`, `cudaSetDevice` → handled by our `Slot`.

## 16. Validation

After each implementation step:

- Build cleanly.
- Tests still pass: `./build/test_movegen`, etc.
- After step 7: smoke-test passes (some bestmove output).
- After step 8: matches dlshogi's bestmove on a few fixed
  tactical positions (within reason — exact same move not
  required, but value should be in same direction).
- NPS measurement: ≥ 30k median on Panda23 with 2 GPUs × 2
  workers, batch 128.

## 17. Out of scope (for this implementation, defer)

- Persistent search across `go` commands. Phase in post-port.
- VirtualLossWeight USI option. Add post-port.
- Sticky endgames / bound propagation (lc0 feature). Add only if
  a strength regression appears.
- df-pn at leaves (we have shallow mate; dlshogi has df-pn).
  Default to shallow mate; df-pn can be revisited later.
- NYUGYOKU_FEATURES NN planes (separate track, requires
  retraining).
- Multi-PV. Add post-port if needed for analysis.
- Stochastic ponder. Defer.
- Random move at low ply (training selfplay feature). Defer.
- Book integration. Keep the existing book code; just call into
  it from CmdGo before invoking the MCTS, same as today.

## 18. Scrap from existing dlshogi_mcts/ directory

After this clean reimplementation lands:

- Delete `dlshogi_mcts/{color,piece,position,score,move,
  generateMoves,init,overloadEnumOperators,common,fastmath,
  Message,nn_tensorrt,mate,int8_calibrator,search}.h*`.
- Delete `dlshogi_mcts/{cppshogi,uct_node,uct_search}.{cpp,h}`
  (replaced by the new files).
- Keep only the new `dlshogi_mcts/{types,uct_node,uct_search}.{h,cc}`
  + `CMakeLists.txt`.

Net result: ~1500 lines in `dlshogi_mcts/`, all in jhbr2's idioms,
no stubs, no dlshogi-specific includes.

## 19. Why this is better than the mechanical port

| | Mechanical port (current) | Clean reimplementation (this design) |
|---|---|---|
| Stub files needed | ~14 | 0 |
| Code lines that exist only as scaffolding | ~440 | 0 |
| Type conversions on every API call | yes (Move↔Move, Position↔ShogiBoard) | no |
| Maintenance burden | high (must keep stubs in sync with dlshogi assumptions) | low (just our code, our types) |
| Easy to evolve | hard (changes fight against dlshogi's structure) | easy (it's our codebase) |
| Risk of algorithmic divergence from dlshogi | low (literal copy) | medium (we wrote it from algorithm description) |
| Effort | low (mechanical) — but we've done it and it's not done | medium (real engineering) |

The "risk of algorithmic divergence" concern is mitigated by
copying the core algorithm pieces verbatim (PUCT, UpdateResult,
mutex bucket scheme) while writing only the integration/glue
fresh. The algorithm correctness comes from dlshogi's tested
formulas; the code structure comes from jhbr2's idioms.

## 20. Final word for the implementing agent

Read dlshogi's `uct_search.cpp` and `uct_node.h` carefully — they
are the ground truth for the algorithm. Don't try to "improve" the
algorithm during the port; mirror it exactly. The port is about
getting dlshogi's algorithm into jhbr2's codebase, not redesigning
MCTS.

Where this design and dlshogi's code disagree, dlshogi wins (the
design might be wrong). Verify against the source.

Where this design tells you to use jhbr2's types (Move, ShogiBoard)
instead of dlshogi's, follow this design — that's the whole point.

Commit early, commit often. Don't let any single commit grow past
~200 lines of new code. Each commit must build and (where
applicable) pass tests.

Good luck.
