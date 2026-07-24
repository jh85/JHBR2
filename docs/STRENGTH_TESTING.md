# Reliable strength testing

`tools/run_strength_test.sh` provides a portable A/B match environment for two
USI engines. It installs `cshogi` into `build-strength/venv` when necessary,
detects visible NVIDIA GPUs, runs games in opening pairs with colors reversed,
and continuously saves results and paired confidence intervals.

## 1. Create a fixed opening suite

Create the suite once and reuse the exact file for every comparison:

```bash
./tools/run_strength_test.sh openings \
  /data/new_jhbr2/user_book1.ybb \
  build-strength/openings-512.txt \
  --count 512 --min-ply 8 --max-ply 12
```

The generator is deterministic by default. It follows varied moves within 100
centipawns of the best YBB move and rejects final positions whose best stored
evaluation is more than 250 centipawns from equality. The file records the
source-book SHA-256 and generation parameters.

This opening suite is only a test input. JHBR2's `BookFile` and
`UseGoteExitBook` remain disabled during normal A/B tests.

## 2. Run a fast screening match

The following example compares one option while holding everything else fixed:

```bash
MODEL=/workspace/JHBR2/engines/shogi_bt4_epoch3.engine

./tools/run_strength_test.sh match \
  --engine-a ./build-trt/jhbr2 \
  --engine-b ./build-trt/jhbr2 \
  --openings build-strength/openings-512.txt \
  --pairs 200 --nodes 100000 \
  --option-a OnnxModel="$MODEL" \
  --option-b OnnxModel="$MODEL" \
  --option-a WorkersPerGpu=2 \
  --option-b WorkersPerGpu=1 \
  --gpus-per-worker 1 \
  --output strength-runs/workers-2-vs-1
```

On an eight-GPU host, `--gpus-per-worker 1` automatically launches eight
parallel pair workers. Each worker exposes one GPU to two persistent engine
processes and automatically sets `NumGPUs=1`. The engines move alternately, so
only one searches at a time, although both models occupy GPU memory.

Use fixed nodes for initial algorithm and parameter comparisons. It greatly
reduces noise from machine load and NPS changes. Change only one factor in each
match.

## 3. Confirm using the production GPU topology and time control

A change that wins in one-GPU screening should be confirmed in the actual
deployment topology:

```bash
./tools/run_strength_test.sh match \
  --engine-a ./build-trt/jhbr2 \
  --engine-b ./build-trt/jhbr2 \
  --openings build-strength/openings-512.txt \
  --pairs 100 --main-time-ms 300000 --byoyomi-ms 10000 \
  --option-a OnnxModel="$MODEL" \
  --option-b OnnxModel="$MODEL" \
  --option-a WorkersPerGpu=2 \
  --option-b WorkersPerGpu=1 \
  --gpus-per-worker 8 \
  --output strength-runs/workers-production
```

With eight visible GPUs, `--gpus-per-worker 8` runs one pair at a time and
automatically sets `NumGPUs=8`. This is slower but captures multi-GPU scaling,
batching, and real time-management effects.

Use `--gpu-devices 2,3,4,5` to select devices explicitly. The wrapper respects
an existing `CUDA_VISIBLE_DEVICES`. Use `--workers N` only when overriding the
automatic one-worker-per-GPU-group choice.

## Results and resuming

Each run directory contains:

- `config.json`: commands, options, executable/model hashes, opening hash, Git
  revision, host, and GPUs.
- `pairs.jsonl`: append-only game records, including moves and final info lines.
- `summary.json`: W-L-D, paired score, Elo estimate, 95% interval, LOS, and
  pentanomial counts.
- `logs/`: engine stderr, plus complete USI traffic when `--protocol-log` is
  requested.

If a cloud allocation ends, rerun the identical command with `--resume`. The
runner verifies the saved configuration and skips completed pairs.

The approximate confidence interval uses a Jeffreys-smoothed pentanomial model
and treats each color-reversed opening pair as one sample, not two independent
games. This avoids both zero-width intervals in tiny runs and overstating
certainty when an opening strongly favors one side. A result is actionable
when the interval is narrow enough for the decision; nine games are generally
not enough. Keep the raw run directory whenever a result influences an engine
change.

## Practical test order

1. Run a short smoke test (`--pairs 2 --nodes 1000`).
2. Screen candidates on one GPU per worker at fixed nodes.
3. Extend promising tests until the paired interval is useful.
4. Confirm the winner with the production GPU count and production clock.
5. Test the final binary against a frozen previous binary, not merely two
   option sets in a newly rebuilt executable.

For a binary-versus-binary test, set `--engine-a` to the saved baseline and
`--engine-b` to the candidate. Absolute model paths are recommended.
