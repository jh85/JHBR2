# JHBR2 Shard Generation — Audit Report

Date: 2026-07-09. Scope: `JHBR2/gen_pack_shards.py` vs. the current JHBR2
model (`shogi_model_v2.py`) and training loader (`shogi_train.py`), plus a new
deduplicating/aggregating generator. Sample data used: the `.pack` and PSV
`.bin` files in this directory.

---

## 1. Expected shard format (verified from code, not guessed)

**Loader**: `ShardedDataset` in `shogi_train.py` globs `{--data}_*.npz` and
expects per-shard arrays:

| array | shape / dtype | meaning |
|---|---|---|
| `planes` | (N, 148, 9, 9) float16 | dlshogi-style features, side-to-move perspective (flipped 180° for white). 28 piece planes + 56 hand thermometer + 1 check + 62 nyugyoku + 1 repetition (always 0 in training — `shogi/encoder.h` documents this) |
| `packed1`/`packed2` | (N, 284)+(N, 15) uint8 | alternative bit-packed planes (auto-detected) |
| `policy` | (N,) int32 | **single class index** in [0, 2187); −1 = no label (masked) |
| `wdl` | (N, 3) float16 | soft (W, D, L) **from the side to move's perspective**, trained with soft cross-entropy |
| `mlh` | (N,) int16 | remaining plies; **< 0 = masked**; clipped to `--mlh-clip` (80) at train time; Huber (smooth-L1) loss |

**Policy encoding**: `make_direction_policy_index` — (direction 0–26) × 81
destination squares = 2187; directions 0–9 moves, 10–19 promotions, 20–26
drops; 180°-rotated when white to move. Shared by the trainer, both
generators, and (per `docs/`) verified bit-identical to the C++ engine
encoder. Verified injective over legal moves in sampled positions.

**Value convention**: side to move. Target = 0.7·sigmoid(score/600) blended
with 0.3·hard game result. `.pack` evals are USI `score cp` of the engine to
move (side-to-move ✓); `.pack` results are absolute (0/1/2) and correctly
converted; PSV results are already side-to-move ±1/0 ✓.

**MLH convention**: raw remaining plies, **not** normalized/log/bucketed;
scalar regression head (`MovesLeftHead`, ReLU output). Zero-point: the last
recorded position of a game has mlh = 0 (i.e. "plies left *after* the teacher
move"), consistent across `gen_pack_shards.py`, `pack_to_shards.py`,
`psv_to_shards.py`. At inference the MCTS uses only *relative* deltas
(child_M − parent_M), so the zero-point choice is harmless as long as it stays
consistent. Note: docstrings say "remaining plies to game end", which is off
by one from the code (`n_moves - i - 1`); cosmetic only.

**Policy loss expected one-hot** (before this work): `F.cross_entropy` with a
long class index — i.e. the loader had **no support for policy vectors**.

## 2. Verdict on `gen_pack_shards.py`

**Semantically compatible with the current model/loader — with findings:**

* ✅ PACK decoding matches the reference
  (`YaneuraOu-ScriptCollection/teacher/convert_teacher.py` →
  `TeacherConvertLib.convert_pack_to_hcpe_file`) byte-for-byte: startpos/HCP
  header, (move16, eval16) pairs, `sq1 == sq2` result terminator + 1 reason
  byte. Verified on the sample `.pack`: 300 games replayed, **all 29,381
  moves legal**, results ∈ {0,1,2}. The terminator check is safe because
  cshogi/Apery drop moves carry from-field ≥ 81 (verified empirically).
* ✅ Position reconstruction, feature planes (shared `sfen_to_planes`,
  bit-identical to the C++/GPU path per `verify_shard_format.py`), policy
  index + flip convention, WDL side-to-move convention, MLH convention: all
  correct.
* ❌ **It could not run at all in this checkout** (fixed): it imported
  `ShogiCommonLib` from `JHBR2/YaneuraOu-ScriptCollection/GenSfen`, which
  doesn't exist; the decoder is `YaneShogiLib` in
  `../YaneuraOu-ScriptCollection/CommonLib`. Import resolution now searches
  both layouts.
* ❌ **Policy target**: emits **one row per raw teacher record with a one-hot
  policy target** (answer to audit task 3). No aggregation, no dedup —
  `startpos`-class positions get massive sampling weight and contradictory
  one-hot labels. Addressed by the new aggregating generator (below), not by
  changing `gen_pack_shards.py` (it remains valid for non-aggregated runs).

**Related bug found & fixed — `psv_to_shards.py`**: it decoded the PSV
`move` field with `cshogi.move_to_usi(raw)` **without**
`cshogi.move16_from_psv()`. PSV stores YaneuraOu-format Move16; drops and
promotions use different bit patterns than cshogi. Measured on the sample
`.bin`: only 1201/2000 raw moves were legal — **~40% of policy targets
(every drop/promotion) were silently corrupted**, since
`move_to_policy_index` checks geometry, not legality. The reference converter
does convert. Fixed.

**Legacy paths** (informational): `psv_dataset.py` / `psv_decode_c.c`
(the `--psv-dir` trainer path) build **48-plane** inputs and return 3-tuples —
incompatible with the current 148-plane model and the 4-tuple training loop.
Treat `--psv-dir` as deprecated; convert PSV via shards instead.

## 3. New: `gen_agg_shards.py` — position-aggregated shards

`canonical position → one AlphaZero-style sample`, for `.pack` and `.bin`
together in one run.

* **Canonical key**: cshogi `HuffmanCodedPos` (32 bytes) = board + side to
  move + hands, no move counter/history — identical for the same position
  from either format (repetition is not encoded; the training repetition
  plane is always 0, so the model cannot depend on it). Grouping uses the
  **full 32 raw bytes** (numpy `V32`, deliberately not `S32`, which strips
  trailing NULs and could merge distinct keys; no lossy small-hash is used —
  the crc32 is only for partitioning).
* **Policy**: all observed moves per position are kept as a sparse normalized
  distribution. Methods: `uniform_unique_moves`, `count`, `sqrt_count`,
  `temperature` with `--alpha` (default **0.5**, the recommended default —
  identical to sqrt_count). `--max-moves` (32) caps entries, keeping the
  heaviest and renormalizing (truncations are counted).
* **Value**: mean of the per-record blended WDL targets (exactly
  0.7·mean-winrate + 0.3·result-frequencies); per-position winrate
  mean/std/min/max and result-conflict counts go to the stats report.
  (A "keep highest-depth" policy isn't possible: neither PACK nor PSV records
  carry depth/quality metadata.)
* **MLH**: aggregated **per position, never per (position, move)** — mean
  over records that have MLH; −1 (masked) when none do. PACK provides true
  recorded-game-end MLH; PSV has none, so the default `--psv-mlh none` masks
  it rather than faking it from `gamePly` (`--psv-mlh recorded-end` opts into
  the plies-to-last-recorded-ply approximation).
* **Scalability**: two streaming phases. Phase 1 decodes records into
  compact 40-byte rows (hcp, move-idx, score, mlh, result) hashed into
  `--partitions` files — no tensors stored per duplicate. Phase 2 sorts each
  partition, groups, aggregates, and encodes feature planes **only for unique
  positions** while writing shards. Phase-2 RAM ≈ records × 40 B ÷
  partitions per worker; use `--partitions 4096` for billions of records.
  `--packed` bit-packing is supported (needs `pyext` C++ encoder).
* **Shard schema** (auto-detected by the loader): `planes` (or packed),
  `policy` (top-weight index, back-compat), ragged sparse
  `policy_indices`/`policy_weights`/`policy_offsets`, `wdl`, `mlh` (float16
  mean, −1 sentinel), `count` (raw records per sample; available for future
  frequency-weighted losses), optional `key` (32-byte HCP, `--store-key`).

**Trainer changes** (`shogi_train.py`, backward compatible):
`ShardedDataset` detects the sparse arrays and yields a dense normalized
(2187,) float target; the policy loss uses a masked **soft cross-entropy**
(`−Σ target·log_softmax`) for vector targets and the original hard CE for
index targets, so old and new shards can be mixed shard-by-shard. The
normalized-distribution requirement of the loss is why the generator always
normalizes weights.

## 4. Statistics report (sample run: first 300k records of each file)

`--policy-method temperature --alpha 0.5`, 64 partitions, 8 workers:

```
raw records                 600,103   (2,855 pack games + PSV chunk)
unique positions            598,537   (duplicate ratio 0.26%)
unique (position,move)      599,021   (484 extra moves preserved by aggregation)
positions seen ≥2×              742
value conflicts (mixed W/L)     417
winrate std over dups       mean 0.014, max 0.138
MLH std over dups           mean 22.0 plies
samples with / without MLH  299,474 / 299,063   (PSV side masked)
unmappable / illegal moves  0
moves-per-position          1: 598,146   2: 324   3: 46   4: 16   5: 5
shards                      64 (≤500k samples each)
```

Duplication is low in this slice because both sample files start games from
varied book positions and `startpos` itself never occurs; on full teacher sets
(hundreds of millions of records sharing openings) the dedup and multi-move
aggregation are exactly what prevents `startpos` domination. Top duplicated
positions in the report show the intended behavior, e.g. one ×15 position kept
4 moves `[2b3c 10 → 0.459, 5a4b 3 → 0.251, 2b8h+ 1 → 0.145, 2a3c 1 → 0.145]`,
and pack+PSV records merged into single samples (partial MLH → position-level
mean over the records that have it).

## 5. Smoke tests

`python test_agg_shards.py --pack-file <file.pack> --psv-file <file.bin>` —
**44/44 pass**:

1. Synthetic `.pack` with known duplicates → exact aggregation math
   (temperature-0.5 weights √2:1, count weights 2/3:1/3, WDL mean blend, MLH
   mean, both observed startpos moves preserved).
2. Real `.pack` → schema/dtypes, unique keys (dedup complete), weights
   normalized, **every nonzero policy entry is a legal move**, WDL rows sum
   to 1 in [0,1], MLH ≥ 0.
3. Real `.bin` → same, plus MLH masked (−1) and PSV→cshogi move conversion
   legality.
4. `ShardedDataset` loads aggregated shards → dense soft policy batches;
   tiny-model forward pass; full loss (soft policy CE + WDL CE + masked MLH
   Huber) + optimizer step, finite. Regression: `gen_pack_shards.py` one-hot
   shards still load and train through the hard-CE path.

End-to-end: `python shogi_train.py --data <out>/aggshard --epochs 1 ...` ran a
full epoch on aggregated shards (loss 11.58 → 9.21; policy/value/mlh all
active).

## 6. Files changed / added

| file | change |
|---|---|
| `JHBR2/gen_agg_shards.py` | **new** — aggregating generator + stats report |
| `JHBR2/test_agg_shards.py` | **new** — smoke tests above |
| `JHBR2/shogi_train.py` | sparse-policy support in `ShardedDataset` + soft policy CE (backward compatible) |
| `JHBR2/psv_to_shards.py` | **bugfix**: `move16_from_psv` conversion (~40% of policy targets were corrupt) |
| `JHBR2/gen_pack_shards.py` | **fix**: `GameDataDecoder` import resolution (script was unrunnable in this layout) |

## 7. Assumptions & open questions

* **Repetition**: plane 147 is always 0 in training data (documented in
  `shogi/encoder.h`), so the canonical key ignores repetition history. If the
  engine sets it at inference, the model sees an input pattern it never
  trained on — pre-existing, out of scope here.
* **MLH zero-point**: kept the existing "0 at the last recorded position"
  convention (off-by-one vs. the docstrings). Inference uses relative deltas,
  so consistency matters more than the offset.
* **Aggregated value is an unweighted per-record mean**; positions with many
  records get a lower-variance target but the same *sample* weight as
  singletons. The `count` array is stored if you later want frequency- or
  confidence-weighted losses.
* **Mixed WDL blend across sources**: pack and PSV records aggregate into one
  target using each record's own score/result — intended, but means one
  sample can blend evals from engines of different strength.
* **`--psv-mlh recorded-end`** measures plies to the *last recorded* ply, not
  the true game end (recording often stops early) — that's why it is off by
  default.
* **Old aggregated data + old loaders**: shards from `gen_agg_shards.py`
  require the updated `shogi_train.py` (older copies would read only the
  top-1 `policy` index — trainable, but the point of aggregation is lost).
* Startpos never appears in the two sample files; the synthetic test covers
  the startpos aggregation path explicitly.
* The pure-Python encoder is the throughput bottleneck (~1k pos/s/worker);
  build `pyext` (`bash pyext/build.sh`) before large runs.
