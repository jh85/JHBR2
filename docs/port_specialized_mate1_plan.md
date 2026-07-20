# Plan: Port the Hand-Specialized Mate-in-1 Routine

## Implementation Outcome

Implemented on 2026-07-20.

The production routine uses a fused, color-specialized check-candidate
walk plus analytical king-escape, capture, pin, discovered-check, and
interposition tests. It does not allocate a checking-move list or make a
full board move. Unlike the upstream contact-focused routine, JHBR2 also
keeps distant slider checks so the public API remains exact.

Correctness results:

- 39,281/39,281 move-oracle positions agree with the independent
  generator-based mate-in-1 oracle.
- 100,000/100,000 deterministic random positions agree.
- Exhaustive cshogi validation has zero JHBR2 failures on 38,088 valid
  non-check positions.
- cshogi's contact-focused `mate_move_in_1ply()` missed two long-range
  mates that both exhaustive oracles and JHBR2 found.
- ASan/UBSan and board-state restoration checks pass.

Performance results:

| Corpus | Previous | Specialized | Improvement |
|---|---:|---:|---:|
| Broad depth 3 | 3.22 us | 2.03 us | 37% |
| Broad depth 5 | 10.96 us | 9.05 us | 17% |
| Broad depth 7 | 63.23 us | 51.56 us | 18% |
| Tactical depth 3 | 26.87 us | 17.03 us | 37% |
| Tactical depth 5 | 227.88 us | 182.96 us | 20% |
| Tactical depth 7 | 1600.05 us | 1274.06 us | 20% |

The standalone specialized routine is approximately 1.5x faster than
the already-optimized JHBR2 oracle, short of the original 3x target.
Profiling a stricter upstream contact-only candidate set reached about
2.2x on tactical positions but missed 46 real mates in the 39,281
position corpus. The exact implementation was retained because it
clears the search-level depth-7 target without that correctness loss.

## Goal

Port dlshogi/YaneuraOu's bitboard-only `mateMoveIn1Ply()` to JHBR2 and
use it as the terminal OR-node check in shallow mate, DFPN, and the
legacy MCTS path.

The port must return a legal mating move or a null move without changing
the board. It must preserve the existing generator-based implementation
as an oracle and as the fallback when the attacker starts in check.

Pinned reference:

- Repository: `TadaoYamaoka/DeepLearningShogi`
- Commit: `5bdf2c8c7ae664651204f29fdbc3d1f2937a8135`
- Source: `cppshogi/position.cpp:577-1408`
- License: GPLv3, compatible with JHBR2

## Design Decisions

### Shared board API

Add a single production API to `ShogiBoard`:

```cpp
Move FindMateInOne();
Move FindMateInOneNonCheck();  // caller guarantees !InCheck()
```

`FindMateInOneNonCheck()` contains the color-specialized port.
`FindMateInOne()` dispatches to it when not in check and uses the current
checking-move plus evasion oracle when in check. The upstream routine
asserts that the attacker's king is not in check, so silently using it
for the in-check case would be incorrect.

The fast routine belongs in `shogi/board.cc` because it needs direct,
carefully scoped access to the piece and color bitboards. Mate search
code should consume the API rather than duplicate board internals.

### Board mutation

The upstream routine temporarily removes a moving piece from its source
bitboards while testing support, captures, and king escapes. Use an RAII
guard for this temporary removal so every early return restores the
bitboards. Do not update the board array, hands, side to move, hash,
history, or continuous-check counters during a probe.

Debug and property tests must verify that SFEN, hash, legal moves,
attacks, and side to move are unchanged after every call.

### Correctness oracle

Keep this independent reference in the test:

1. Generate legal checking moves.
2. Apply each move with known `gives_check=true`.
3. Test `!HasLegalEvasion()`.
4. Undo the move.

Fast and oracle moves need not be identical when several mates exist.
The required properties are:

- Both agree on whether mate-in-1 exists.
- A fast returned move is legal, gives check, and leaves no evasion.
- A pawn-drop mate is never returned.

## Implementation Phases

### Phase 0: Baseline and test harness

- Add a dedicated mate-in-1 property test and microbenchmark.
- Run the oracle over `test/positions.txt`, tactical mate corpora, and
  deterministic random-walk positions.
- Cross-check verdicts with exhaustive cshogi legal moves. Treat
  `Board.mate_move_in_1ply()` as a positive oracle because its upstream
  contact-focused implementation can omit distant slider mates.
- Record current mate-in-1 and depth 3/5/7 timings before integration.

No production call site changes in this phase.

### Phase 1: Analytical helper primitives

Add private helpers corresponding to the upstream implementation:

- `CanKingEscapeAfterCheck`: test defender king destinations using a
  hypothetical occupancy and the checking piece's explicit attack mask.
- `CanDefenderCaptureChecker`: find non-king captures and reject pinned
  defenders that would expose their king.
- `IsPinnedMoveIllegal`: test whether an attacking move leaves its own
  king exposed.
- `IsDiscoveredCheck`: test whether moving a blocker exposes a slider.
- `CheckingMoveIsSupported`: test whether a moved checker can be
  captured by the king.
- A scoped source-piece bitboard removal guard.

Reuse existing `PieceAttacks`, `AttackersTo`, `IsSquareAttacked`,
`ComputeBlockersForKing`, `LineBB`, and the check-zone tables. Do not
introduce a second attack implementation.

Each helper should have focused fixtures for pins, discovered checks,
x-rays through the old king square, and captures on the checking square.

### Phase 2: Port drop mates

Port the upstream order and pruning for:

1. Rook and lance drops
2. Bishop drops
3. Gold drops
4. Silver drops
5. Knight drops

Pawn drops are deliberately excluded because pawn-drop checkmate is
illegal. Validate black and white cases for every drop type before
continuing.

### Phase 3: Port board-move mates

Port piece families incrementally, preserving promotion distinctions:

1. Dragon and rook
2. Horse and bishop
3. Gold and promoted minor pieces
4. Silver
5. Knight
6. Lance
7. Pawn

For every family, cover:

- Capturing and non-capturing mates
- Promoting and non-promoting variants
- Direct and discovered checks
- Supported and unsupported checking pieces
- Attacker pins and defender pinned captures
- Both colors and promotion-zone boundaries

Run the complete oracle comparison after every family. Do not integrate
a partially complete routine into search.

### Phase 4: Integrate one call site at a time

After zero oracle mismatches:

1. Replace the terminal loop in `MateIn3Ply`.
2. Replace `MateInOddPly<1, false>`.
3. Keep `MateInOddPly<1, true>` on the in-check fallback.
4. Make `MateDfpnSolver::Mate1Ply` delegate to `ShogiBoard`.
5. Make the legacy `MCTSSearch::Mate1Ply` delegate to `ShogiBoard`.

Keep pawn-drop legality in `GenerateLegalMoves` on `HasLegalEvasion()`;
calling `FindMateInOne()` there would create the wrong dependency and
could recurse through drop legality.

### Phase 5: Validation and performance gate

Required correctness:

- Zero fast/oracle verdict mismatches on all 39,281 move-oracle
  positions.
- Zero mismatches on at least 100,000 deterministic random positions.
- Zero mismatches against exhaustive cshogi legal-move validation on the
  same valid non-check positions; every positive from cshogi's
  specialized routine must also be found.
- All move-generation, checking-move, shallow-mate, and DFPN tests pass.
- Debug assertions plus ASan/UBSan pass.
- Board state is unchanged after every fast call.

Required performance:

- Profile the specialized mate-in-1 routine against the current oracle;
  retain exact long-range coverage even if the contact-only upstream
  subset benchmarks faster.
- No depth 3, 5, or 7 benchmark regresses by more than 2%.
- Depth 7 improves by at least 15% on the tactical corpus.

If the routine passes correctness but misses the performance gates,
profile before integration. Do not retain the large specialized path
without a measured search-level benefit.

## Delivery Sequence

Use separate reviewable commits:

1. Oracle tests, fixtures, and baseline benchmark
2. Helper primitives and specialized routine
3. Search integrations, final benchmarks, and documentation

The existing generator-based mate-in-1 path remains available until the
final commit, making rollback a call-site change rather than a board
rewrite.

## Main Risks

- Temporary bitboard state escaping through an early return
- Promotion variants with different check geometry
- Treating a pinned defender as able to capture the checker
- Missing an x-ray after the attacking piece leaves its source square
- Accidentally accepting illegal pawn-drop mate
- Calling the upstream non-check routine while the attacker is in check

RAII restoration, dual oracles, piece-family staging, and state-invariant
tests directly cover these risks.
