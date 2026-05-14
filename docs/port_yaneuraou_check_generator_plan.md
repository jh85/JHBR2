# Phase 7: Port YaneuraOu's `generateMoves<Check>` to `ShogiBoard`

Goal: replace `ShogiBoard::GenerateCheckingMoves`'s current
`GenerateLegalMoves + filter` implementation with a true
specialized check generator (port of YaneuraOu's `generateMoves<Check>`),
matching dlshogi-class per-call performance.

**Background**:
- Phase 6 added `GenerateCheckingMoves` using a filter approach
  (~2-3 μs per call). Made `LeafMateDepth=3` viable.
- Benchmark showed `LeafMateDepth=5` is still ~5× slower than ideal
  due to the filter overhead. dlshogi at depth=5 doesn't have this
  problem because it inherits YaneuraOu's specialized generator
  (~0.3-0.5 μs per call).
- This phase closes that gap.

Reference: `DeepLearningShogi/cppshogi/generateMoves.cpp:633-1149`
(YaneuraOu's implementation, bundled in dlshogi).

---

## Strategy: incremental port with property-test safety

Same "additive only" principle as Phase 6:

1. Keep current `GenerateCheckingMoves` as-is (rename internally to
   `GenerateCheckingMovesSlow`). It serves as the **oracle**.
2. Implement new `GenerateCheckingMovesFast` step by step.
3. After each step, assert `Fast == Slow` on all corpus positions.
4. Once full port passes, swap `shallow_mate.h` to use Fast.
5. Eventually delete Slow if no longer needed.

The property test (`test/test_check_movegen.cc`) is the safety net.
Any regression is caught immediately on the 5,200+ position corpus.

---

## Files to modify

| File | Action | Notes |
|---|---|---|
| `shogi/bitboard.h`, `bitboard.cc` | Add precomputed check tables to `ShogiTables` | One bitboard per (color, king_sq, piece_type) for steppers; one per (color, king_sq) for lances; per king_sq for bishop/rook |
| `shogi/board.h` | Add `DiscoveredCheckBB()` method | Bitboard of OUR pieces whose move would discover check |
| `shogi/board.h`, `board.cc` | Add `GenerateCheckingMovesFast` | The main port |
| `test/test_check_movegen.cc` | Add Fast vs Slow property test | Existing test extended |
| `mate/shallow_mate.h` | Switch to Fast (after validation) | One-line change per call site |

---

## Sub-phases

### 7a: precompute check tables

**Tables to build at `ShogiTables::Init()` time:**

```cpp
// In ShogiTables namespace (bitboard.h):

// Step piece check tables: squares from which a piece of (pt, c) attacks ksq.
// All step pieces have small finite attack patterns, so these are constructed
// by iterating each square S and testing if PieceAttacks(pt, c, S, Empty).Test(ksq).
extern Bitboard PawnCheckBB  [2][81];   // [color][king_sq]
extern Bitboard KnightCheckBB[2][81];
extern Bitboard SilverCheckBB[2][81];
extern Bitboard GoldCheckBB  [2][81];   // covers gold + tokin + nari-{kyou,kei,gin}

// Sliding piece check tables: squares S where the slider's MAXIMAL attack
// pattern (with empty occupancy) reaches ksq. The actual occupancy-aware
// blocking is checked later during move generation.
extern Bitboard LanceCheckBB [2][81];   // file ray in defender's forward direction
extern Bitboard BishopCheckBB[81];      // four diagonals extended to edges
extern Bitboard HorseCheckBB [81];      // bishop pattern + 1-step orthogonal
// Rook is universal (any of our rooks could potentially check), no table needed.
```

Memory: ~3 KB total. Initialized once in `ShogiTables::Init()`.

**Sanity check**: for every (pt, c, ksq), assert that
`bb := ShogiTables::CheckBB[pt][c][ksq]` matches the slow recomputation
`{S : PieceAttacks(pt, c, S, EMPTY) ∋ ksq}`.

### 7b: `DiscoveredCheckBB()`

Bitboard of OUR pieces that, if moved, would expose a sliding attack
on the enemy king. Uses existing `ComputeBlockersForKing(enemy_color)`
intersected with our pieces.

```cpp
Bitboard ShogiBoard::DiscoveredCheckBB() const {
    return ComputeBlockersForKing(~side_to_move_) & pieces(side_to_move_);
}
```

(May already be inline-computable from existing primitives — verify.)

### 7c: `GenerateCheckingMovesFast` — direct check candidates

Following YaneuraOu's structure
(`generateMoves.cpp:633-`):

```cpp
const Square ksq = king_sq_[~us];
Bitboard direct_check_candidates =
    (pieces(us, kPawn)   & ShogiTables::PawnCheckBB  [us][ksq]) |
    (pieces(us, kLance)  & ShogiTables::LanceCheckBB [us][ksq]) |
    (pieces(us, kKnight) & ShogiTables::KnightCheckBB[us][ksq]) |
    (pieces(us, kSilver) & ShogiTables::SilverCheckBB[us][ksq]) |
    ((pieces(us, kGold)  | promoted_steppers(us)) &
                          ShogiTables::GoldCheckBB  [us][ksq]) |
    (pieces(us, kBishop) & ShogiTables::BishopCheckBB[ksq]) |
    (pieces(us, kRook)   /* any rook — no table needed */)   |
    (pieces(us, kHorse)  & ShogiTables::HorseCheckBB [ksq]) |
    (pieces(us, kDragon) /* any dragon */);

Bitboard discovered_check_candidates = DiscoveredCheckBB();
```

For each piece in `direct_check_candidates ∪ discovered_check_candidates`:
- Iterate destinations using `attacksFrom(piece_type, color, src) & target`
- Filter destinations: must be one that gives check (direct: in
  the per-piece check ray; discovered: not on the line src→ksq)
- Handle promotions (each move may have promote/no-promote variants;
  classify each independently)
- Skip pinned-piece illegal moves

### 7d: drop checks

For each piece type in hand, compute the "drop check zone" — empty
squares from which a dropped piece of that type would attack the
enemy king. Intersect with empty squares, generate drops there.

```cpp
const int hand_pawn = hand(us).count(kPawn);
if (hand_pawn > 0) {
    Bitboard drops = ShogiTables::PawnCheckBB[us][ksq] & ~occupied();
    // Filter for two-pawn rule, uchifuzume, rank constraints
    while (drops.Any()) { ... }
}
// Same for Lance, Knight, Silver, Bishop, Rook, Gold
```

### 7e: validation pass

For each sub-phase, assert `Fast(b) == Slow(b)` on:
- Hand-curated positions (Tier 1, Phase 6)
- 1200 mate puzzle positions (Tier 2)
- ~4000 random-walk positions (Tier 3)
- Synthetic adversarial: maximum pinned/blocker count, maximum drops
  available, maximum check moves possible

Only swap `shallow_mate.h` to Fast after ALL tiers pass with zero
mismatches.

### 7f: re-bench NPS

Re-run `/tmp/bench_nps.py` with depth=3 and depth=5. Acceptance:
- depth=3: ≥1.5× faster than dfpn n=10 (better than current 1.82× allowed; just
  shouldn't regress)
- depth=5: ≥1× of dfpn n=10 across ALL tested positions (i.e., no
  endgame catastrophe)
- Property test still 100% on ALL tiers

### 7g: cleanup

Once Fast is validated and shipped:
- Optionally rename `Fast` → `GenerateCheckingMoves` (drop the suffix)
- Keep `Slow` as `GenerateCheckingMovesViaFilter` for future
  property-test use, OR remove if confidence is high enough.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bitboard-layout mismatch with YaneuraOu causes subtle bugs | High | High | Property test against the slow oracle on every commit |
| Promotion edge case: a move's promote/no-promote variants give different check status | Medium | Medium | Test specifically with promotion-conditional position; use two passes |
| Discovered + direct overlap: a move is both → emitted twice | Medium | Low | Dedupe by tracking emitted destinations per piece, OR use bit operations to subtract overlap |
| Pin-aware legality wrong → emit illegal moves | Medium | High | Use existing `IsLegal(m, pinned)` filter at end, even though it's slower; optimize later |
| Uchifuzume not handled in drop generation | Medium | Medium | Reuse existing uchifuzume detection in `GenerateLegalMoves` |
| Off-by-one in CheckTables | Medium | High | Sanity-check tables at init time against PieceAttacks |

---

## Estimated total: 2-3 days of focused work

| Sub-phase | Time |
|---|---|
| 7a. Precompute check tables | 0.5 day |
| 7b. DiscoveredCheckBB | 0.25 day |
| 7c. Direct check generation | 0.75 day |
| 7d. Drop check generation | 0.5 day |
| 7e. Validation | 0.25 day |
| 7f. Bench + ship | 0.25 day |
| 7g. Cleanup | 0.25 day |

---

## Acceptance criteria

The port is "done" when all are true:

1. ✅ `Fast(b) == Slow(b)` for all 5,200+ corpus positions
2. ✅ All 51 existing shallow_mate tests still pass
3. ✅ Microbench shows Fast is ≥3× faster than Slow per call
4. ✅ NPS bench shows shallow d=3 is ≥1.5× faster than dfpn n=10 on
   average AND no catastrophic regression on any individual position
5. ✅ NPS bench shows shallow d=5 is ≥0.9× of dfpn n=10 across all
   tested positions (eliminating the endgame regression)
6. ✅ shallow_mate.h is using the Fast version
