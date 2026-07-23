# Gote early-exit YBB book

JHBR2 can use a generated opening policy that tries to force Sente out of a
source YaneuraOu Binary Book (YBB) as early as possible. The policy is used
only when JHBR2 is Gote. Normal `BookFile` behavior remains available for
Sente and whenever the feature is disabled.

## Generate the monthly book

From the JHBR2 repository:

```bash
./tools/generate_gote_exit_book.sh /path/to/new_user_book1.ybb
```

The default output is `user_book1_gote_exit.ybb` in the repository root. If
that file already exists, rerun with `--force`:

```bash
./tools/generate_gote_exit_book.sh /path/to/new_user_book1.ybb --force
```

`--force` preserves the previous output as a timestamped
`.previous.<timestamp>` file before installing the new one. To choose another
output path, supply it as the second positional argument:

```bash
./tools/generate_gote_exit_book.sh \
  /path/to/new_user_book1.ybb \
  /path/to/user_book1_gote_exit.ybb
```

The default strength guard allows only Gote moves within 30 centipawns of the
best stored evaluation at every Gote node. Override it directly on the command
line:

```bash
./tools/generate_gote_exit_book.sh /path/to/new_user_book1.ybb \
  --gote-exit-eval-margin 20 --force
```

The build uses all available CPU threads. Use `--threads 16` to override that
number. The shorter `--eval-margin` spelling and the
`GOTE_EXIT_EVAL_MARGIN`/`GOTE_EXIT_THREADS` environment variables remain
available for compatibility and automation. `GOTE_BOOK_BUILD_DIR` changes the
CMake build directory, and `GOTE_BOOK_BUILD_JOBS` changes compiler
parallelism. The lower-level generator also accepts `--validate-moves` for a
slower full legality check. `--max-positions` is only for development tests;
its output is incomplete and must not be used for play.

The source must be a distributed/PetaShock-style YBB whose stored move lists
contain the in-book transpositions that should participate in minimax
processing. JHBR2 now reads YBB V1 directly; the old text `.db` format is no
longer supported.

## Enable the policy in USI

Configure the engine before `isready`:

```text
setoption name BookFile value /path/to/original_user_book1.ybb
setoption name GoteExitBookFile value /path/to/user_book1_gote_exit.ybb
setoption name UseGoteExitBook value true
isready
```

`UseGoteExitBook` can be toggled at any time. Changing either book path takes
effect on the next `isready`.

When enabled:

- Sente uses `BookFile`, if configured.
- Gote uses only `GoteExitBookFile`.
- A Gote specialized-book miss falls through to MCTS. It deliberately does
  not fall back to `BookFile`, because the miss marks the intended book exit.

The generated file contains only Gote positions and one selected move per
position. It retains the selected source move's evaluation and depth.

## Selection algorithm

The converter resolves each stored move to its successor PackedSfen and solves
an asymmetric reachability game over the book graph:

- Gote minimizes the number of plies until an out-of-book edge.
- Sente maximizes that number and may maintain a cycle.
- Gote moves below the evaluation-margin threshold are excluded.
- If Gote cannot force a finite exit, the generator keeps the highest-valued
  plausible Gote move as a conservative fallback.

This is a graph-aware retrograde minimax calculation, so transpositions and
cycles are handled explicitly rather than by assuming the book is a tree.
