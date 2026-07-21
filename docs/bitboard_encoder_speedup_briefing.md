# Briefing: speed up bitboard and encoder

This document is the kickoff brief for a fresh agent session to
optimize jhbr2's `shogi/bitboard.{h,cc}` and `shogi/encoder.{h,cc}`
for raw CPU throughput.

This work is **independent of the in-progress dlshogi MCTS port**
(see `docs/dlshogi_port_step1_briefing.md`). Both efforts will land
into the same eventual jhbr2 binary; you don't need to coordinate
with the MCTS work beyond standard git practice (work on a separate
branch, rebase before merging).

Read first:
1. This document.
2. `docs/concurrency_dlshogi_vs_jhbr2.md` — for context on jhbr2's
   overall NPS gap and where bitboard/encoder fit in the picture.

---

## Goal

Make jhbr2's bitboard and encoder as fast as possible without changing
their **observable behavior** (any move generated, attack computed,
or input plane produced must be identical to the current
implementation).

**Success criteria:**

- All existing tests pass (`./build/test_movegen`,
  `./build/test_check_movegen`, `./build/test_replay`,
  `./build/test_mate`).
- Single-threaded benchmark of move-generation throughput improves
  by ≥ 20%. Encoder throughput improves by ≥ 30%.
- No regressions in the broader engine NPS — measure
  `tools/benchmark.py` before and after.

The bitboard and encoder are CPU-bound and run frequently:
- Move generation: every PUCT walk step, every leaf extension,
  every mate-detection probe.
- Encoder: every NN evaluation (so once per leaf for per-leaf
  gathering — a lot).

A 20–30% speedup in either translates to a measurable but bounded
NPS improvement in the overall engine (probably 5–10% of total NPS,
since GPU + tree walk cost is also real). The point isn't a 2×
engine speedup — it's stacking incremental wins.

---

## What you keep working on

After the recent refactor (`shogi_engine/` → `shogi/`), the relevant
files are:

```
shogi/
├── bitboard.h
├── bitboard.cc
├── board.h
├── board.cc
├── encoder.h
├── encoder.cc
├── types.h
└── ...
```

`board.cc` calls into `bitboard.cc` for attack computation and into
`encoder.cc` for plane production. Test suite is in `test/`:
`test_movegen.cc`, `test_check_movegen.cc`, `test_replay.cc`.

## What's out of scope

- **Don't touch the MCTS code** (`lc0_mcts/`, `dlshogi_mcts/`). That's
  a separate effort.
- **Don't change the public API** of `bitboard.h` / `encoder.h`. Other
  code (board, search, mate solvers) calls these by signature.
  Internal implementation can change freely.
- **Don't add new functionality.** Algorithmic correctness must remain
  identical — same moves generated, same encoder output bit-for-bit.
- **Don't rewrite in a different language.** Same C++17 codebase.

## Profile first, optimize second

Before changing any code, **measure where time is actually spent.**
Speculation about hot spots is reliably wrong; data isn't.

```bash
# Build with debug symbols + optimizations
cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_TENSORRT=ON
make -j jhbr2

# Run a representative workload under perf
sudo sysctl kernel.perf_event_paranoid=1   # if not already set

cat > /tmp/profile_input.txt << 'EOF'
usi
setoption name OnnxModel value /path/to/model.onnx
setoption name UseGPU value true
setoption name NumGPUs value 1
setoption name WorkersPerGpu value 1
setoption name MinibatchSize value 64
isready
position startpos
go byoyomi 30000
quit
EOF

perf record -g -F 999 -- ./jhbr2 < /tmp/profile_input.txt
perf report --stdio | head -100 > /tmp/profile.txt
less /tmp/profile.txt
```

Look for:
- Time in `lczero::Bitboard::*` methods.
- Time in `EncodeShogiPosition` / `ShogiInputPlane::SetFromBitboard`.
- Time in `ShogiBoard::GenerateLegalMoves` / `GenerateCheckingMoves`.
- Anything else that surprises you.

**Useful sub-benchmarks** (write these if they don't exist; small
focused timing programs):

```bash
# Move-gen throughput
./test/bench_movegen --positions 1000000   # rough goal: million-positions/sec

# Encoder throughput
./test/bench_encoder --positions 1000000   # rough goal: million-encodes/sec
```

If those binaries don't exist, write them first. Without
single-purpose benchmarks you can't reliably attribute speedups.

## Specific optimization ideas to investigate

### 1. BMI2 PEXT for sliding-piece attacks (highest expected value)

Current bitboard probably uses **magic bitboards** for sliding pieces
(rook, bishop, lance, dragon, horse): a multiplication + shift to
index a precomputed attack table.

YaneuraOu and modern Stockfish use **BMI2 PEXT** instead:
`pext_u64(occupancy, mask)` extracts the relevant occupancy bits
into a small index, with no multiplication needed. ~15–25% faster
on x86 with BMI2 (Panda23 has it; check `cat /proc/cpuinfo | grep bmi2`).

Decision tree:
1. Check if jhbr2 already uses PEXT. `grep -rn "_pext_u64\|_BMI2\|HAVE_BMI2" shogi/`
2. If yes: skip.
3. If no: port. Look at YaneuraOu's `bitboard.cpp` for reference (it
   has both code paths under #ifdef USE_BMI2).

Build flag: usually `-mbmi2` to gcc/clang. Worth gating behind a
runtime CPU detection so the binary still works on older CPUs.

### 2. SIMD for 81-bit shogi bitboards

Shogi has 81 squares — doesn't fit in `uint64_t`. Common
representations:

- **Two `uint64_t` halves** (current jhbr2; one for ranks 1-7, one
  for ranks 8-9, or similar split).
- **`__m128i`** SIMD register (128 bits, plenty of room for 81).
- **`__m256i`** (256 bits) — overkill for one bitboard, useful for
  operating on multiple bitboards at once.

The "float-vs-int" thing the user remembers from YaneuraOu likely
relates to **`__m128i` operations sharing hardware with float-vector
operations** (XMM registers are used for both). Single `__m128i`
operation replaces two `uint64_t` operations on the bitboard halves.

YaneuraOu's `bitboard.h` is the reference. Look at how they
implement `Bitboard::operator&`, `operator|`, `operator^`, `popcount`,
`isSet`, `popLsb`.

Decision: profile shows whether bitboard logical ops are a hot spot
worth SIMD-izing. If they are 5%+ of CPU time, this is a real win.

### 3. Move-list construction allocation

`MoveList` may be allocating on each `GenerateLegalMoves` call. If
so, pre-allocating a thread-local fixed-size buffer (shogi has at
most ~600 legal moves, typically < 100) and recycling it would
eliminate allocator pressure.

Check: `grep -n "std::vector\|reserve\|push_back" shogi/board.cc | head`.

Pattern to consider: `std::array<Move, 600> + size_t count`
instead of `std::vector<Move>`. No heap, fixed cost, fits on stack.

### 4. Encoder plane construction

`ShogiInputPlane::SetFromBitboard` (in `shogi/encoder.h`) walks a
bitboard and writes 1.0 to corresponding squares of a 81-float
plane. Likely loop-with-pop-lsb pattern.

Optimizations to try:
- Vectorize: instead of pop-lsb-by-bit, decode the whole 64-bit
  half at once via lookup tables (256 entries each → 32-byte mask
  per byte). Expand to 8 floats with SIMD.
- Memset-then-set-some: if the bitboard is sparse, current is fine.
  If dense, `std::fill(plane, plane+81, 0)` then mark bits.
- Avoid bit-by-bit Pop loop entirely with `_mm256_maskstore_ps` or
  similar gather/scatter.

Encoder runs *per leaf* — even a 30% speedup here translates to
real engine gain.

### 5. Repetition-detection hash check

`ShogiBoard::CheckRepetition` walks `history_` linearly. If history
is long (200+ moves), this matters. Consider: maintaining a
small hash-map of seen positions for O(1) lookup.

Profile dependency: only worth it if `CheckRepetition` shows up in
the profile.

### 6. Hand piece compression

Each hand has 7 piece types with bounded counts. Currently encoded
as 7 floats (one per piece type) in the encoder. If the dlshogi
one-hot expansion is ever adopted, that's 56 floats — but for now,
the tight encoding is cheap.

## Workflow

1. **Profile** the current code (see above). Identify the top 3
   hot functions.
2. **Pick one** and write a focused micro-benchmark for it.
3. **Try one optimization.** Measure.
4. **Verify correctness:** all tests still pass.
5. **Verify net engine NPS:** run `tools/benchmark.py` and compare
   median NPS before/after.
6. **Commit** the change with a brief summary of the speedup
   measured.
7. **Repeat** for the next hot function.

Small, measured commits are better than one big rewrite. Each
commit should isolate one optimization with a measured impact, so
regressions can be bisected later.

## Things that bit prior agents — avoid these

- **Don't make the bitboard a different size.** Many places assume
  `Bitboard` is exactly 16 bytes (two uint64_t). Going to `__m128i`
  may keep that size but watch alignment.
- **Don't change the iteration order** of `GenerateLegalMoves`.
  Move ordering affects PUCT visit order, which affects search
  paths, which affects test outputs. Tests check exact move lists.
- **Don't break BMI2-less CPUs without a fallback.** If you add
  PEXT, gate it behind `#ifdef __BMI2__` or runtime check, with
  the existing path as fallback.
- **Don't over-optimize a cold function.** A 5× speedup of code
  that runs 0.1% of the time is invisible.
- **Don't change the encoder output layout** (plane order, plane
  count, scale). The TRT engine was built for a specific input
  shape; changing it breaks all model loading.

## Validation gates

After each optimization:

```bash
# 1. All tests pass
cd build
make -j test_movegen test_check_movegen test_replay test_mate
./test_movegen ../test/positions.txt
./test_check_movegen
./test_replay ../test/game1_replay.txt
./test_mate

# 2. Engine still produces same outputs on a known position
echo -e "usi\nsetoption name OnnxModel value /path/to/model.onnx\nsetoption name MaxNodes value 10000\nisready\nposition startpos\ngo byoyomi 5000\nquit" | ./jhbr2 | grep -E "^info|^bestmove"
# Compare bestmove + score with pre-change run. Should match within
# noise (small Q-value differences are OK; bestmove differences
# shouldn't happen for the same input.)

# 3. Net engine NPS measurement
python3 ../tools/benchmark.py --threads 1 --gpus 1 --minibatch 64 \
  --byoyomi 10000 --leaf-mate-mode off --limit 30 \
  ./jhbr2 /path/to/model.onnx
# Compare median NPS with baseline.
```

## What "done" looks like for this work

This is **incremental optimization**, not a project with a clear
end state. Each commit either lands a measured speedup or doesn't.
Candidates for stopping:

- All hot functions are within 20% of theoretical optimum (you've
  squeezed everything obvious out).
- Engine median NPS improvement plateaus (further bitboard/encoder
  speedups don't translate to engine NPS — bottleneck is elsewhere).
- The MCTS port lands and rebases this work — good time to measure
  and decide whether to continue.

A concrete intermediate goal: **bring the encoder-per-call cost down
by 30% and move-gen-per-call cost down by 20%.** That's enough to be
worth a release.

## Repo layout reference (post-rename)

```
/home/ei/Downloads/JHBR2/
├── shogi/                  # ← board, bitboard, encoder, types (your focus)
├── lc0_mcts/               # current MCTS (don't touch)
├── dlshogi_mcts/           # in-progress port (don't touch)
├── inference/              # NN evaluator (don't touch)
├── usi/                    # USI handler (don't touch)
├── mate/                   # mate detection (don't touch)
├── test/                   # tests + corpora
├── tools/                  # benchmark.py
└── docs/                   # planning + briefing docs
```

## When to commit

- After each individual optimization with measured speedup.
- Don't commit a "WIP rewrite" with no measured gain — that adds
  risk without reward.
- Don't commit a regression — if a change makes engine NPS worse,
  back it out.

## When in doubt

- Profile data > intuition. Always.
- Verify against tests before claiming a speedup.
- Read YaneuraOu's `bitboard.h` and `bitboard.cpp` for reference
  implementations of fast shogi bitboards. They're the gold
  standard and well-commented (in Japanese, but readable).
- Push back on the briefing if something here doesn't make sense
  — fresh eyes catch bad assumptions.

Good luck. Small wins compound.
