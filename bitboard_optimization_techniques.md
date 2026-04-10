# Bitboard Optimization Techniques for Shogi Engines

Techniques applied to JHBR2 (9x9 Shogi), ordered by implementation dependency
and impact. Each technique is independent and testable — verify correctness
after each step using a move generation oracle (e.g., cshogi).

## Measured Results (JHBR2, 9x9 Shogi)

| Step | Technique | Speedup | Cumulative |
|------|-----------|---------|------------|
| 1 | Precomputed step attack tables | 2.5x | 2.5x |
| 2 | Zobrist incremental hashing | 1.2x | |
| 3 | Eliminate board copy in IsLegal | 2.5x | |
| 4 | Stack-allocated MoveList | 1.2x | |
| 2-4 combined | | | 5.5x |
| 5 | Qugiy lance + rook vertical | 1.2x | 6.5x |
| 6 | Qugiy rook horizontal + bishop | 1.1x | 7.0x |
| — | (not yet) SSE2/AVX2 SIMD bitboard | ~1.3x | ~9x |
| — | (not yet) Pin-aware legal move gen | ~2x | ~18x |

Starting point: 1.9M nodes/sec. Final: 13.3M nodes/sec. Target (cshogi): 26.8M nodes/sec.

---

## Technique 1: Precomputed Step Attack Tables

**Problem:** Computing step attacks (pawn, knight, silver, gold, king) per-call
with bounds-checking loops over delta arrays.

**Solution:** Precompute attack bitboards at startup for every (square, color)
combination. Replace per-call computation with a single table lookup.

**Tables:**
- PawnEffectBB[num_squares][2]
- KnightEffectBB[num_squares][2]
- SilverEffectBB[num_squares][2]
- GoldEffectBB[num_squares][2] (also used for promoted pawn/lance/knight/silver)
- KingEffectBB[num_squares]
- HorseStepBB[num_squares] (4 cardinal directions)
- DragonStepBB[num_squares] (4 diagonal directions)

**Scales to any board size.** Table memory is O(num_squares x piece_types).

---

## Technique 2: Zobrist Incremental Hashing

**Problem:** ComputeHash() iterates over all squares O(N) after every DoMove.

**Solution:** Maintain hash incrementally via XOR during DoMove:
- Piece placement: `hash ^= Zobrist::Psq[piece][square]`
- Hand changes: `hash ^= Zobrist::Hand[color][piece_type][count]`
- Side flip: `hash ^= Zobrist::Side`

**Key:** XOR is its own inverse, so UndoMove can simply restore the saved hash.

**Scales to any board size.** Zobrist tables grow as O(num_squares x num_piece_types).

---

## Technique 3: Eliminate Board Copy in IsLegal

**Problem:** Checking move legality by copying the entire board, applying DoMove,
checking InCheck, then discarding the copy. This copies hundreds of bytes per
pseudo-legal move (80-200 moves per position).

**Solution:** Use in-place DoMove/UndoMove on the actual board:
```
IsLegal(move):
  undo = DoMove(move)
  legal = !InCheck(us)
  UndoMove(move, undo)
  return legal
```

**Prerequisite:** Technique 2 (Zobrist hashing) -- otherwise DoMove's hash
recomputation makes in-place DoMove/UndoMove just as slow.

**Requires:** Making GenerateLegalMoves() and IsLegal() non-const, since they
temporarily modify the board.

**Scales to any board size.** Impact increases with board size since copies are larger.

---

## Technique 4: Stack-Allocated MoveList

**Problem:** `MoveList = std::vector<Move>` heap-allocates on every
GenerateLegalMoves() call.

**Solution:** Fixed-capacity stack buffer:
```
class MoveList {
  Move moves_[MAX_MOVES];  // 600 for standard shogi
  int count_ = 0;
};
```

**For larger variants:** Increase MAX_MOVES. Chu-shogi may need 1000+.
Tai-shogi may need 2000+. Measure the actual maximum.

---

## Technique 5: Qugiy O(1) Lance and Rook Vertical Effects

**Problem:** Loop-based ray tracing walks one square at a time along a ray
until hitting a blocker. For a lance on an open file, this is up to 8 iterations.

**Solution:** Bit-subtraction trick (Qugiy algorithm).

**For forward direction (WHITE lance, toward higher bits):**
```
mask = precomputed ray mask (all squares forward on the file)
mocc = occupancy & mask    (blockers in the ray)
effect = (mocc ^ (mocc - 1)) & mask
```
Subtraction borrows through zeros until hitting the first set bit (blocker),
flipping all bits in between -- which are exactly the attacked squares.

**For backward direction (BLACK lance, toward lower bits):**
```
mask = precomputed ray mask
mocc = occupancy & mask
msb = 63 - clz(mocc | 1)   // find highest blocker (| 1 avoids UB when empty)
effect = (~0 << msb) & mask
```

**Rook vertical = BLACK lance effect | WHITE lance effect.**

**Prerequisite:** Each file must be contiguous bits within a single machine word.
This is naturally true in vertical bitboard layouts (square = file x board_size + rank).

**Scales to any board size** as long as one file fits in a 64-bit word (max rank <= 64).

---

## Technique 6: Qugiy Rook Horizontal and Bishop Diagonal Effects

**Problem:** Horizontal and diagonal rays span non-adjacent bits in a vertical
bitboard layout (same-rank squares are board_size bits apart).

**Solution:** Qugiy algorithm with byte_reverse + unpack + decrement.

**Core idea:**
1. `byte_reverse(occupancy)` -- mirrors the bitboard horizontally, transforming
   the "right" direction into a "left" direction for arithmetic.
2. `unpack(reversed_occ, occ)` -- rearranges so that left-direction and
   right-direction occupancies are in separate 64-bit lanes.
3. `decrement(hi, lo)` -- subtracts 1 from two independent 128-bit integers
   in parallel, finding the first blocker in each direction.
4. XOR with original -> changed bits = attacked squares.
5. AND with mask -> filter to relevant squares only.
6. `unpack` back -> restore original layout.
7. `byte_reverse` the right-direction result -> un-mirror.
8. OR together -> combined horizontal/diagonal effect.

**For bishop:** Use Bitboard256 (4 x 64-bit) to process all 4 diagonals
as 2 independent 128-bit decrements in parallel.

**Key methods added to Bitboard:**
- `byte_reverse()` -- reverses byte order and swaps halves (__builtin_bswap64)
- `Unpack(hi_in, lo_in, hi_out, lo_out)` -- rearranges 64-bit lanes
- `Decrement(hi_in, lo_in, hi_out, lo_out)` -- 128-bit subtract-1 with borrow

**Precomputed masks:**
- QugiyRookMask[num_squares][2] -- left/right horizontal rays (after unpack)
- QugiyBishopMask[num_squares][2] -- diagonal rays (as Bitboard256, after unpack)

**Scales to any board size** with appropriate SIMD width. The algorithm is
board-size-independent; only the mask tables and register widths change.

---

## Technique 7: SIMD Bitboard Operations (NOT YET IMPLEMENTED)

**Problem:** Bitboard AND/OR/XOR/shift use two scalar 64-bit operations
instead of one SIMD instruction.

**Solution:** Store bitboard as `__m128i` (SSE2) for <=128 squares, or
`__m256i` (AVX2) for <=256 squares. All bitwise operations become single
instructions.

**Also accelerates:** byte_reverse (SSSE3 `_mm_shuffle_epi8`),
decrement (SSE4.1 `_mm_cmpeq_epi64` + `_mm_add_epi64`),
unpack (`_mm_unpackhi/lo_epi64`).

**Board size determines SIMD width:**
- 9x9 (81 squares): __m128i (SSE2)
- 12x12 (144 squares): __m256i (AVX2)
- 15x15 (225 squares): __m256i (AVX2)
- 16x16+ (256+ squares): __m512i (AVX-512)

---

## Technique 8: Pin-Aware Legal Move Generation (NOT YET IMPLEMENTED)

**Problem:** Testing each pseudo-legal move with DoMove + InCheck + UndoMove.
For 80 pseudo-legal moves, that's 80 full board state transitions.

**Solution:** Precompute a "pinned pieces" bitboard (blockers_for_king).
Then legality is two cheap bitboard tests:

```
legal(move):
  if is_drop: return true  (drop legality handled during generation)
  if piece is king: return !attacked(destination)
  if piece not pinned: return true
  return aligned(from, to, king_square)  // pinned piece stays on pin line
```

**Requires:** `blockers_for_king` bitboard and `aligned()`/`line_bb()` tables.
YaneuraOu maintains blockers_for_king incrementally in DoMove.

**Scales to any board size.** This is purely algorithmic.

---

## Verification Strategy

At every step, verify correctness using a reference oracle (cshogi for standard
shogi, or equivalent for other variants):

1. **Generate test positions** -- random playouts + edge cases using the oracle.
   Save SFEN + legal move sets.
2. **Static test** -- for each saved position, set up via SetFromSfen, compare
   GenerateLegalMoves() output against expected moves.
3. **Replay test** -- replay a full game via sequential DoMove, check SFEN and
   legal moves at each step match the oracle (catches DoMove/UndoMove bugs).
4. **Perft** -- recursive move count to fixed depth. Compare node counts against
   the oracle. This is the strongest correctness check.
5. **Benchmark** -- measure raw movegen calls/sec and perft nodes/sec after each
   step to confirm speedup.

---

## Architecture Notes for Larger Variants

**Chu-shogi (12x12, 144 squares):**
- Base bitboard: 3 x uint64_t or __m256i (256 bits)
- Qugiy works with adapted unpack/decrement for 256-bit
- ~46 piece types (including lions, promoted pieces)
- Step attack tables much larger but same principle
- Lion moves (area moves, igui) need special handling

**Tai-shogi (25x25, 625 squares):**
- Bitboard likely impractical (need 10 x uint64_t)
- Consider mailbox + piece lists instead
- Or hybrid: bitboard for occupancy queries, mailbox for piece identity

**General rule:** If num_squares fits in the widest available SIMD register
(512 bits on current hardware), bitboard is viable. Beyond that, mailbox wins.
