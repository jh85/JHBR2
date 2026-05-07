# Plan: Port dlshogi's 5-Ply Leaf Mate Check to jhbr2

Goal: replace per-leaf df-pn (current `LeafDfpnNodes=10–20`, ~5–20 ms
per call) with dlshogi-style 5-ply shallow mate search (~100–500 ns
per call), recovering NPS in tactical positions while preserving
mate-in-1/3/5 detection.

**Background**: see `docs/architecture_improvements_research.md` for
the cost/coverage analysis that motivates this change.

---

## Scope and non-goals

### In scope

- Implement a `ShallowMateSolver` class in jhbr2 with the same AND/OR
  recursive structure as `DeepLearningShogi/usi/mate.h`
- Hand-tuned mate-in-3 specialization (the 80/20 win)
- Wire it into `lc0_mcts/search.cc` as the per-leaf mate check
- Keep df-pn available behind a flag for fallback / experimentation
- Unit tests with known mate-in-1/3/5/7 positions
- Cross-validation against current df-pn on a corpus of real positions
- Microbenchmark vs. current df-pn

### Out of scope (separate work)

- **Defensive 1-ply lookahead** at our-turn leaves (the day-1 sudden-mate
  fix). Once this port lands, defensive lookahead becomes feasible
  because the per-leaf check is 100× cheaper. But it's a separate change.
- **Background PV mate search** (dlshogi's df-pn-on-PV thread). Would
  catch deep mates that 5-ply misses. Optional follow-up.
- **NN inference cache** and **MakeSolid re-enable**. Separate items
  in the architecture plan.

---

## Files to create / modify

| File | Action | Notes |
|---|---|---|
| `mate/shallow_mate.h` | **NEW** | Templates `mateMoveInOddPly<N>`, `mateMoveInEvenPly<N>`, hand-tuned `mateMoveIn3Ply` |
| `mate/shallow_mate.cc` | **NEW** | Helper functions if any non-template code is needed |
| `mate/move_picker_mate.h` | **NEW** | Equivalent of `ns_mate::MovePicker<or_node, INCHECK>` for ShogiBoard |
| `lc0_mcts/search.cc` | **MODIFY** | Replace df-pn at lines 455 and 822 with a switch on `config_.leaf_mate_mode` |
| `lc0_mcts/search.h` | **MODIFY** | Add `leaf_mate_mode` and `leaf_mate_depth` to config |
| `usi/usi_engine.cc` | **MODIFY** | Add USI options `LeafMateMode` (df-pn / shallow / off), `LeafMateDepth` (default 5) |
| `test/test_shallow_mate.cc` | **NEW** | Unit tests with known mate-in-N positions |
| `test/bench_mate.cc` | **NEW** | Microbenchmark df-pn vs shallow on a position corpus |

---

## API mapping (ShogiBoard ⇄ dlshogi Position)

This is the table we'll work from when porting `mate.h`. We'll add
small adapter helpers where dlshogi's API doesn't match 1:1.

| dlshogi (`Position`) | jhbr2 (`ShogiBoard`) | Notes |
|---|---|---|
| `pos.inCheck()` | `board.InCheck()` | Direct |
| `generateMoves<CheckAll>(pos)` | `GenerateLegalMoves()` filtered by `move_gives_check(b, m)` | jhbr2 doesn't have a check-only generator. Acceptable overhead for now; potential future optimization. |
| `generateMoves<Evasion>(pos)` | `GenerateLegalMoves()` (returns evasions automatically when in check, since they're the only legal moves) | Direct — when in check, all legal moves ARE evasions. |
| `pos.moveGivesCheck(m, ci)` | helper: `MoveGivesCheck(board, m)` | Implement: do move, test `board.InCheck(opponent)`, undo. Slightly slower than dlshogi's pre-computed CheckInfo, but simple. |
| `pos.pinnedBB()` | `board.ComputeBlockersForKing(side_to_move)` | Direct |
| `pos.doMove(m, state, ci, givesCheck)` | `UndoInfo undo = board.DoMove(m)` | jhbr2 returns UndoInfo by value instead of state-by-ref. Minor refactor. |
| `pos.undoMove(m)` | `board.UndoMove(m, undo)` | jhbr2 needs the saved UndoInfo |
| `pos.isDraw(16)` → RepetitionLose/Win/etc. | `board.CheckRepetition()` | Verify the enum values map cleanly |
| `pos.gamePly()` | `board.PlyFromRoot()` or equivalent | Need to confirm exact API name |
| `IsLegal(m, pinned)` | `board.IsLegal(m, pinned)` | Direct (already exists) |

**Key porting gotcha**: dlshogi uses **pseudo-legal** generation +
filter, while `GenerateLegalMoves()` already produces only legal
moves. So we don't need to filter for legality — but we DO need to
filter for "gives check" (and that's the slow part to add).

---

## Implementation phases

### Phase 0: prep & plumbing (~0.5 day)

- [ ] Add `MoveGivesCheck(board, m)` helper in `mate/shallow_mate.h`
  (do/undo + InCheck check). Crude but correct.
- [ ] Add `RepetitionResult ToShallowResult(...)` adapter.
- [ ] Add `mate_mode` enum to `lc0_mcts/search.h::Config`:
  ```cpp
  enum class LeafMateMode { kOff, kDfpn, kShallow };
  LeafMateMode leaf_mate_mode = LeafMateMode::kDfpn;   // default = current behavior
  int leaf_mate_depth = 5;                              // for shallow
  ```
- [ ] Wire USI options.

### Phase 1: port the templates (~1.5 days)

- [ ] Port `ns_mate::MovePicker<or_node, INCHECK>` → `MovePickerMate`
  - OR mode: generate legal moves, keep only those where `MoveGivesCheck`
  - AND mode: generate legal moves (board is in check, so all legal
    moves are evasions automatically)
  - Cap at `MaxCheckMoves = 91` (same as dlshogi)
- [ ] Port `mateMoveInEvenPly<N>` (AND node) — generic recursive template
- [ ] Port `mateMoveInOddPly<N>` (OR node) — generic recursive template
- [ ] Port `mateMoveIn3Ply<INCHECK>` — hand-tuned base case
- [ ] Add template specialization
  `mateMoveInOddPly<3, INCHECK> → mateMoveIn3Ply<INCHECK>`

Use dlshogi's mate.h as the reference — keep variable names and
control flow identical where possible. This minimizes the chance of
introducing bugs in the porting step.

### Phase 2: integration (~0.5 day)

- [ ] In `lc0_mcts/search.cc:455` and `:822`, switch on
  `config_.leaf_mate_mode`:
  ```cpp
  switch (config_.leaf_mate_mode) {
    case kOff:     break;
    case kDfpn:    /* existing df-pn path */ break;
    case kShallow:
      if (jhbr2::ShallowMateSolver::HasMateInOddPly(
            board, config_.leaf_mate_depth)) {
        node->MakeTerminal(GameResult::BLACK_WON);
        return;
      }
      break;
  }
  ```
- [ ] `HasMateInOddPly(board, depth)` dispatches on depth (3, 5, 7)
  by selecting the appropriate template instantiation.

### Phase 3: testing (~1 day)

- [ ] **Unit tests** in `test/test_shallow_mate.cc`:
  - 30+ known mate-in-1 positions (various piece configurations)
  - 30+ known mate-in-3 positions
  - 30+ known mate-in-5 positions
  - 30+ known no-mate positions (to test the negative case)
  - Edge cases: stalemate-like positions (impossible in Shogi but
    near-stalemate), uchifuzume (illegal pawn-drop mate), repetition
- [ ] **Cross-validation** against current df-pn on a corpus of
  real positions (e.g., 10K positions sampled from your training
  data). For positions where df-pn at high budget says "mate in N≤5",
  the new shallow check should agree. Disagreements get logged for
  investigation.
- [ ] **Microbenchmark** in `test/bench_mate.cc`:
  - Run df-pn (LeafDfpnNodes=20) and shallow (depth=5) over the same
    1M positions. Measure wall time and decisions per second.
  - Expectation: shallow is 100–1000× faster on the average position.
  - Also measure on the "tactical position" subset where df-pn was
    tanking NPS to 259/sec — confirm shallow stays fast there.

### Phase 4: validation in real play (~0.5 day)

- [ ] Run a 100-game match: jhbr2-shallow vs jhbr2-dfpn on floodgate
  test positions or against a fixed sparring opponent
- [ ] Confirm: same/better strength, dramatically higher NPS in
  tactical positions
- [ ] Confirm: no new sudden-mate failures (the check still finds
  short forced mates, including the day-1 P*6c position if mate
  exists within 5 plies — though that one was likely deeper)

### Phase 5: rollout (~0.5 day)

- [ ] Default `leaf_mate_mode = kShallow` once Phase 4 passes
- [ ] Keep `kDfpn` available as a USI-tunable fallback
- [ ] Add `leaf_mate_depth` UCI/USI option (default 5, allow 3 or 7)
- [ ] Document in CLAUDE.md or README

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `MoveGivesCheck` slower than dlshogi's CheckInfo, eats most of the speedup | Medium | High | Cross-check microbench. If real, optimize: precompute pinned bitboard once per OR-node and short-circuit obvious non-checks. |
| Repetition handling subtly different between `pos.isDraw(16)` and `board.CheckRepetition()` | Medium | Medium | Carefully test: positions with repetition-in-search-path should match df-pn's behavior. |
| Mate-in-3 specialization has a corner-case bug (uchifuzume, special drop rules) | Low | Medium | Cover with unit tests. dlshogi's reference implementation has been tournament-tested for years. |
| Strength regression in real play (network expects df-pn-style search) | Low | High | Phase 4 game match. Fallback: keep `kDfpn` as default until validated. |
| `MakeSolid` SIGSEGV interacts with new path | Low | High | The shallow check doesn't touch tree structure (operates on a board copy), so this should be unaffected. Verify with stress test. |

---

## Acceptance criteria

The port is "done" when all of these are true:

1. ✅ All unit tests in `test_shallow_mate.cc` pass
2. ✅ On the cross-validation corpus, shallow check agrees with high-budget
   df-pn on **≥99.5%** of mate-in-≤5 verdicts (small disagreements OK
   for genuinely-edge cases like uchifuzume)
3. ✅ Microbenchmark shows shallow is **≥50× faster** than df-pn on
   typical (non-mate) positions
4. ✅ NPS in CP≈0 positions recovers to **≥3000/sec** (vs. observed
   259–695/sec with df-pn)
5. ✅ 100-game match shows ≥0% Elo (no regression) with shallow vs df-pn
6. ✅ The day-1 P*6c position behavior is at least no worse (note: this
   specific position likely needs **defensive lookahead** to fix, which
   is a separate change after this port — but the port enables it)

---

## Estimated total: 4 days of focused work

| Phase | Time |
|---|---|
| 0. Prep & plumbing | 0.5 day |
| 1. Port templates | 1.5 days |
| 2. Integration | 0.5 day |
| 3. Testing | 1.0 day |
| 4. Real-play validation | 0.5 day |
| 5. Rollout | 0.5 day |
| 6. (Optional follow-up) `GenerateCheckingMoves` | 3–5 days |

---

## Phase 6 (optional follow-up): specialized `GenerateCheckingMoves`

This phase is **only triggered if Phase 4 performance measurement
shows that move generation is still the bottleneck**. Phases 0–5
ship a working, fast 5-ply check using a filter (`GenerateLegalMoves()`
+ filter-by-`MoveGivesCheck`). That filter approach is ~50–200× faster
than df-pn — the big win — but it's still ~100–300× slower than
dlshogi's specialized check generator.

### Trade-off captured at the time of writing this plan

| Approach | Per-OR-node cost | 5-ply cost | vs df-pn |
|---|---|---|---|
| Filter (Phase 1 default) | ~5–15 μs | ~25–150 μs | 50–200× faster |
| Specialized generator | ~500 ns – 2 μs | ~5–10 μs | another 100–300× faster |

So Phase 6 is the optimization that closes the remaining gap with
dlshogi. **Decision rule: only do Phase 6 if NPS in tactical positions
is still below ~3000/sec after Phase 5.**

### Why this is a separate phase, not part of Phase 1

- Modifies `ShogiBoard` (the fundamental building block). Higher risk
  than the rest of the port, which only adds new files.
- Bundling it with Phase 1 makes debugging harder: if a strength
  regression shows up, you'd need to disambiguate between bug-in-mate-
  templates vs bug-in-new-move-generator.
- Phase 4 perf measurement may show the filter is fast enough in
  practice, in which case Phase 6 is unnecessary.

### Safety principle: "additive only"

Add `GenerateCheckingMoves()` as a **new method on `ShogiBoard`**.
Never modify `GenerateLegalMoves()` or any existing method. Every
existing call site is untouched, so the worst-case bug is contained
to the new mate check itself.

```cpp
class ShogiBoard {
public:
  MoveList GenerateLegalMoves();      // unchanged
  MoveList GenerateCheckingMoves();   // NEW — only used by mate check
  // ...
};
```

### Implementation sketch

For each position, generate moves that produce check via:

1. **Direct checks** — moves to a square that attacks the enemy king:
   - Compute `direct_check_squares` per piece type using existing
     attack-pattern helpers (`pawn_attacks_from`, `bishop_attacks`, ...)
   - For each of our pieces of type T, intersect its legal-move
     destinations with `direct_check_squares[T]`

2. **Discovered checks** — moves of pieces that block sliding attacks
   on the enemy king:
   - `our_blockers = ComputeBlockersForKing(opponent_color) & our_pieces`
   - Any move of a blocker (to a square not on the king-line) is a
     discovered check

3. **Drops** — drops that attack the enemy king:
   - For each piece type in hand, drops on `direct_check_squares[T]`
   - Filter out illegal drops (uchifuzume, two-pawn rule, etc. — reuse
     existing drop legality logic)

4. **Promotion**: each generated move that's eligible for promotion
   needs to consider both promote and no-promote variants. The check
   status may differ between them (e.g., promoting silver → gold
   changes attack pattern → may or may not still be check).

Reuse existing primitives:

| Primitive | Used for |
|---|---|
| `AttackersTo(sq, occ)` | Sanity-checking after computing direct checks |
| `ComputeBlockersForKing(c)` | Discovered checks |
| `IsLegal(m, pinned)` | Filter pinned-piece illegal moves |
| `ComputeGameResult` | Already handles uchifuzume — verify drop generation respects it |

### Testing strategy (the key safety net)

The filter approach IS the oracle. It's guaranteed correct (just
slow). So the property test is exact equivalence:

```cpp
// Property: for every position, the two methods produce equal sets.
void test_GenerateCheckingMoves_matches_oracle(const ShogiBoard& b) {
    auto specialized = b.GenerateCheckingMoves();
    auto oracle = filter(b.GenerateLegalMoves(),
                         [&](Move m){ return MoveGivesCheck(b, m); });
    ASSERT_EQ(set_of(specialized), set_of(oracle))
        << "Mismatch at position: " << b.sfen();
}
```

#### Test corpus tiers (run all of them):

1. **Hand-curated mate-in-N positions** (~100 positions): correctness
   on cases known to involve discovered checks, double checks,
   promotion-conditional checks, drop checks
2. **Random legal positions** (~10,000): edge cases that real games
   might not exercise
3. **Real-game positions from training data** (~1B): if Phase 5 shards
   exist, run the property test over the full converted training data.
   Effectively exhaustive coverage on positions that occur in practice.
4. **Adversarial positions**: positions with maximum complexity (many
   pinned pieces, many drops, promotion zone, in-check)

The test suite should pass on **every single position in tier 3** before
the new generator is wired into the mate check.

### Acceptance criteria for Phase 6

- ✅ Property test passes on all 4 tiers (especially: zero failures
  across the entire training-data corpus)
- ✅ Microbenchmark shows ≥10× speedup on `GenerateCheckingMoves` vs
  filtered `GenerateLegalMoves`
- ✅ End-to-end mate check is ≥3× faster than the Phase 1 filter
  version
- ✅ Integration test: 5-ply check verdicts (mate / no-mate) are
  identical to Phase 1 filter version on a 100K-position corpus
- ✅ 100-game match shows no Elo regression vs Phase 1 filter version

### Wire-up

Phase 6 is a one-line change in the mate-check code:

```cpp
// Before (Phase 1):
auto moves = filter(b.GenerateLegalMoves(), MoveGivesCheck);

// After (Phase 6):
auto moves = b.GenerateCheckingMoves();
```

Keep both paths behind a flag during validation:

```cpp
auto moves = config_.use_specialized_check_generator
           ? b.GenerateCheckingMoves()
           : filter(b.GenerateLegalMoves(), MoveGivesCheck);
```

Once Phase 6 acceptance criteria pass, default the flag to true and
delete the filter path in a separate commit.

---

## Open questions to resolve before starting

1. **Does jhbr2 have a check-only move generator that I missed?**
   (search `GenerateChecks`, `GenerateChecking`, `CheckList`...) If yes,
   use it; if no, accept the legal-then-filter cost for now.
2. **What does `board.CheckRepetition()` return exactly?** Need the
   enum/values to map dlshogi's RepetitionResult correctly.
3. **Does `ShogiBoard` track game ply for the `draw_ply` parameter?**
   dlshogi uses `pos.gamePly() + 2 > draw_ply` to bail when search
   would extend past max plies — need equivalent.
4. **Is there an existing `MoveGivesCheck` helper anywhere in the
   codebase?** Faster than do/undo if it exists.

These are 30-minute investigations; resolve them at the start of
Phase 0.
