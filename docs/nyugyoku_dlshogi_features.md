# dlshogi-style nyugyoku features + GPU bit-unpack (148-plane layout)

This is the **single source of truth** for the input-plane layout. Three
implementations must agree bit-for-bit:

1. `shogi/encoder.cc` — `PackShogiPosition()` (C++ inference packing) and
   `EncodeShogiPosition()` (CPU unpack = reference float planes).
2. `shogi/encoder_unpack.cu` — GPU unpack kernels (the speedup path).
3. `shogi_train.py` — `sfen_to_planes()` (training-side encoder).

## Why packing

dlshogi avoids transferring full float planes (`148 × 81 × 4 B ≈ 48 KB/pos`)
across PCIe. Instead it transfers **packed bits** (`~300 B/pos`) and expands
them on the GPU with a trivial kernel. Two plane families:

- **features1 (positional)** — one bit *per square*. 28 planes × 81 = 2268
  bits = `kPackedF1Bytes = 284`.
- **features2 (uniform / one-hot)** — one bit *per plane*, broadcast to all 81
  squares on the GPU. 120 planes = `kPackedF2Bytes = 15`.

The bit→float trick (branchless): `x = (-(int)bit) & 0x3f800000` reinterpreted
as float gives `1.0f` for bit=1, `0.0f` for bit=0.

## Perspective

All planes are encoded from the **side-to-move's perspective**. The C++ encoder
flips the board 180° (`board.Flipped()`) when WHITE is to move, so on the
working board `b` the mover is always BLACK: `us = BLACK`, `them = WHITE`.
The Python encoder flips coordinates equivalently.

## Plane layout (total `kShogiInputPlanes = 148`)

### features1 — planes 0..27 (positional bitmaps, 1 bit/square)

Cell order within a plane is `rank*9 + file` (C-order `[r][f]`), matching
`data[rank*9+file]` and Python `planes[p, r, f]`.

| Planes | Contents |
|--------|----------|
| 0..13  | our piece types: P, L, N, S, B, R, G, K, +P, +L, +N, +S, +H, +D |
| 14..27 | their piece types (same order) |

Packed bit index: `f1idx * 81 + cell`, `f1idx ∈ [0,28)`.

### features2 — planes 28..147 (uniform one-hot, 1 bit/plane)

Local index `j ∈ [0,120)`; global plane = `28 + j`. Packed bit index = `j`.

| Local j | Contents |
|---------|----------|
| 0..27   | OUR hand, thermometer (unary): P×8, L×4, N×4, S×4, G×4, B×2, R×2 |
| 28..55  | THEIR hand, same order/sizes |
| 56      | check (side-to-move in check) |
| 57..87  | OUR nyugyoku block (31) — see below |
| 88..118 | THEIR nyugyoku block (31) |
| 119     | repetition flag (current position seen before) |

Hand thermometer: for count `n` of a piece with max `M`, set the first
`min(n, M)` planes of its run to 1.

Hand piece order and maxes (matches dlshogi `MAX_PIECES_IN_HAND`):
`P=8, L=4, N=4, S=4, G=4, B=2, R=2` (sum = 28).

#### Nyugyoku block (31 planes, per color), offset `base`

| Offset | Contents |
|--------|----------|
| 0          | king is in the enemy camp (promotion zone) |
| 1 + k      | opp-field piece count: `k = clamp(10 - pieces_in_camp, 0, 9)`, set only if `pieces_in_camp ≥ 1` |
| 11 + k     | declaration score: `k = clamp(threshold - points, 0, 19)`, set only if `threshold - points < 20` |

- `pieces_in_camp` = our pieces in enemy camp, **excluding king**.
- `points` = camp + hand declaration points (R/B/Dragon/Horse = 5, others = 1),
  king excluded. From `ShogiBoard::ComputeEnteringKingInfo`.
- `threshold` = **28 if this color is sente (the original first player), else 27.**
  Because the board may be flipped, sente-ness is derived from the *original*
  `board.side_to_move()`:
  - our threshold  = `orig_stm == BLACK ? 28 : 27`
  - their threshold = `orig_stm == BLACK ? 27 : 28`

This mirrors dlshogi `cppshogi.cpp::make_input_features` (`NYUGYOKU_FEATURES`),
where opp-field/score are single one-hot at the "remaining" index.

## Retraining

Input channels change `48 → 148`, so **existing weights are invalid** — a full
retrain is required. `shogi_train.py` `Config.input_planes` and the shard
`planes` dtype shape `(N, 148, 9, 9)` must match. See the TODO in
`psv_to_shards.py` / `pack_to_shards.py` docstrings (shape comment).

## Note: repetition train/infer skew (pre-existing)

`sfen_to_planes` cannot see game history from a bare SFEN, so plane 147
(repetition) is always 0 in training data. The C++ inference encoder sets it
from real history. This skew predates this change and is left as-is; the model
simply learns to mostly ignore the plane. If it matters, feed repetition via
the shard pipeline.
