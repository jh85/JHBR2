# JHBR2 engine parameter and constant inventory

Last audited: 2026-07-24

## Scope and classification

This inventory follows the source files that are compiled into the `jhbr2`
binary: `main.cc`, `usi/`, `dlshogi_mcts/`, `inference/`, `shogi/`, `mate/`,
and the runtime YBB reader in `book/`. It intentionally excludes training,
conversion, match-runner, and Gote-book-generator scripts.

It records every externally configurable engine setting and every hard-coded
constant that can materially affect playing behavior, throughput, time use,
diagnostics, or file/model compatibility. Mechanical loop bounds that merely
implement the 9x9 board are grouped under structural constants instead of
being repeated for every loop.

The classifications used below are:

- **Tune:** a plausible strength or hardware parameter. Change it only in a
  paired experiment.
- **Operational:** deployment, reporting, or safety behavior; normally do not
  optimize it for Elo.
- **Rule:** shogi or USI semantics. Do not tune it.
- **Format:** model, move, PackedSfen, or YBB compatibility. Changing it
  requires a coordinated format migration.
- **Implementation:** data-structure or synchronization design. Profile and
  test before changing it.

Defaults in this document are the actual C++ defaults, not example script
values.

## Active USI options

### Search policy and result selection

| USI option | Default | Parser limit | Class | Active effect |
|---|---:|---:|---|---|
| `CInit` | 1.25 | 0--100 | Tune | Non-root PUCT exploration constant |
| `CBase` | 19652 | 1--1e9 | Tune | Non-root visit-dependent PUCT scale |
| `FpuReduction` | 0.27 | 0--100 | Tune | Non-root first-play urgency reduction |
| `CInitRoot` | 1.25 | 0--100 | Tune | Root PUCT exploration constant |
| `CBaseRoot` | 19652 | 1--1e9 | Tune | Root visit-dependent PUCT scale |
| `FpuReductionRoot` | 0.0 | 0--100 | Tune | Root first-play urgency reduction |
| `DrawValueBlack` | 0.5 | 0--1 | Tune/ruleset | Value backed up for a draw when Black traversed the edge |
| `DrawValueWhite` | 0.5 | 0--1 | Tune/ruleset | Value backed up for a draw when White traversed the edge |
| `ResignThreshold` | 0.01 | 0--0.5 | Tune/safety | Resign when the selected root move's win probability is below this |
| `MaxMovesToDraw` | 100000 | 1--100000 | Rule/operational | Search returns a draw once `board.ply() > value`; no practical default cap |
| `MovesLeftWeight` | 0.0 | 0--100 | Tune | Master strength of the MLH selection term; zero disables it |
| `MovesLeftThreshold` | 0.0 | 0--0.5 | Tune | Apply MLH only when `abs(Q - 0.5)` exceeds this |
| `MovesLeftCap` | 20.0 | 0--10000 | Tune | Clamp the child-minus-parent moves-left delta in plies |

The parser also accepts dlshogi-compatible aliases `c_init`, `c_base`,
`c_fpu_reduction`, `c_init_root`, `c_base_root`, and
`c_fpu_reduction_root`. The very wide parser limits are safety bounds, not
recommended experiment ranges. Suggested initial ranges are in
`docs/ENGINE_STRENGTH_ROADMAP.md`.

The active PUCT score in `dlshogi_mcts/search_primitives.cc` is:

```text
c = c_init + log((parent_visits + c_base + 1) / c_base)
score = Q + c * U * policy_prior + moves_left_effect
FPU = max(0, parent_Q - fpu_reduction * sqrt(visited_policy_mass))
```

Changing any of the options in this table rebuilds the persistent `Search`
object, so a new search cannot retain a tree built under different semantics.

### Search work, parallelism, cache, and reporting

| USI option | Default | Advertised range | Class | Active effect |
|---|---:|---:|---|---|
| `MaxNodes` | 800 | 1--10,000,000 | Tune/test | Default playout limit when `go nodes` is absent |
| `NumGPUs` | 1 | 1--8 | Operational/tune | One evaluator and worker group per visible device index |
| `WorkersPerGpu` | 2 | 1--64 | Tune | Search threads and TensorRT execution slots per GPU |
| `Threads` | 2 | 1--64 | Compatibility | Exact alias of `WorkersPerGpu` |
| `MinibatchSize` | 128 | 1--4096 | Tune | Maximum gathered leaves per worker iteration |
| `NNCacheSize` | 0 | 0--100,000,000 | Tune | Shared cached positions; zero disables the cache |
| `InfoIntervalMs` | 1000 | 100--10,000 | Operational | Minimum interval between periodic structured `info` records |

`MinibatchSize` may exceed the TensorRT engine profile maximum. The backend
then splits the request into profile-sized chunks, so values above the profile
maximum do not create a larger inference batch.

`MaxNodes` counts terminal and queued playouts, but not trajectories discarded
because another thread is expanding the same node. Parallel workers can
overshoot the exact limit by work already gathered into their batches.

### Mate search and move-time controls

| USI option | Default | Advertised range | Class | Active effect |
|---|---:|---:|---|---|
| `LeafMateMode` | `shallow` | `off`, `shallow` | Tune | Enables or disables depth-bounded mate checks before NN evaluation |
| `LeafMateDepth` | 5 | 1--7 | Tune | Rounded down to one of 1, 3, 5, 7 |
| `RootMateDepth` | 7 | 0--7 | Tune/safety | Rejects root candidates allowing an opponent mate; zero disables, positive values round down to odd |
| `DfPnMaxTime` | 4000 ms | 100--60,000 | Tune/time | Wall-time cap for concurrent root DFPN |
| `MaxMoveTime` | 0 | 0--300,000 ms | Operational | Hard move cap; zero disables |
| `MaxMoveTime1m` | 0 | 0--60,000 ms | Operational | Replaces `MaxMoveTime` when remaining main time is below 60 seconds; zero disables |

`LeafMateMode=dfpn` is accepted only for old configuration compatibility and
is treated as `off`. It is no longer advertised.

Root mate rejection normally checks only the MCTS-selected move. If that move
allows mate, it marks the move losing and checks the next-best candidate,
continuing for at most the number of root legal moves.

### Model/backend and opening-book controls

| USI option | Default | Values | Class | Active effect |
|---|---|---|---|---|
| `OnnxModel` | `shogi_bt4.onnx` | path | Operational | ONNX path in ORT builds or serialized engine path in TensorRT builds |
| `ModelFormat` | `auto` | `auto`, `jhbr2`, `dlshogi` | Format | Selects tensor names and feature encoder |
| `DlshogiModel` | `false` | check | Compatibility | `true` selects dlshogi; `false` restores auto detection |
| `UseGPU` | `true` | check | Operational | ORT tries CUDA then CPU; native TensorRT always uses its GPU |
| `BookFile` | empty | YBB path | Operational | Normal book; empty disables it |
| `UseGoteExitBook` | `false` | check | Operational | Enables the specialized book only when White/Gote is to move |
| `GoteExitBookFile` | `user_book1_gote_exit.ybb` | YBB path | Operational | Specialized Gote-book path |

Changing a model, format, GPU count, worker count, or `UseGPU` destroys both
the search and evaluator objects. Book files are reopened at the next
`isready`. When `UseGoteExitBook=true`, a specialized-book miss falls through
to MCTS and never falls back to the normal book. With `BookFile` empty, Sente
always uses MCTS.

Model-load exceptions and a TensorRT evaluator with zero worker slots are
contained: `isready` reports `info string Model load failed...`, returns
`readyok`, and subsequent `go` returns `bestmove resign`.

## USI `go` parameters and protocol behavior

These are command parameters rather than persistent options:

| Input | Behavior | Class |
|---|---|---|
| `go nodes N` | Replaces `MaxNodes` for that search | Test/tune |
| `go infinite` | Uses 10,000,000 nodes | Operational limitation |
| `btime`, `wtime` | Selects remaining main time for the side to move | Operational |
| `binc`, `winc` | Selects per-move increment for the side to move | Operational |
| `byoyomi` | Shared per-move byoyomi | Operational |
| `go mate infinite` | DFPN limit of 10,000,000 nodes | Operational |
| `go mate T` | DFPN limit `max(T_ms * 200, 100,000)` plus wall-time stop | Operational |
| `ponder` | Parsed but does not alter search | Protocol limitation |

`go infinite` is not truly unbounded. More importantly, `CmdGo` is currently
synchronous, so the main input loop cannot process `stop` or `ponderhit` until
the search has returned. The watchdog can stop a timed search from another
thread, but an external `stop` cannot interrupt an active synchronous search.
This is a known architectural limitation, not a tuning parameter.

The timed `go mate` loop polls every 10 ms.

## Hard-coded time-manager constants

Source: `usi/time_manager.cc`.

| Constant/branch | Current value | Class | Meaning |
|---|---:|---|---|
| Byoyomi allocation | 0.90 | Tune/time | MCTS seconds are 90% of byoyomi |
| Main-time fraction | 0.05 | Tune/time | Used only when byoyomi is zero |
| Increment fraction | 0.80 | Tune/time | Added only in the no-byoyomi branch |
| Minimum timed search | 0.1 s | Operational | Floor in the main/increment branch |
| Explicit-cap reserve | 0.5 s | Operational | Subtracted from `MaxMoveTime`/`MaxMoveTime1m` |
| Explicit-cap floor | 0.5 s | Operational | Minimum MCTS budget after that subtraction |
| Watchdog grace | 2000 ms | Operational/safety | Added when no explicit move cap exists |
| DFPN deadline reserve | 50 ms | Operational/safety | DFPN cap stays this far inside hard deadline |

Root DFPN uses bands based on
`available_ms = main_time + increment + byoyomi`:

| Available time | Post-MCTS grace | DFPN node limit |
|---:|---:|---:|
| no clock / node search | 300 ms | 100,000 |
| below 10 s | 100 ms | 10,000 |
| 10--59.999 s | 300 ms | 100,000 |
| 60--299.999 s | 500 ms | 500,000 |
| 300 s or more | 1000 ms | 2,000,000 |

The current branch order means a control such as 300 seconds main time plus
10-second byoyomi spends about 9 seconds on MCTS and does not deliberately
invest any main time. Replacing this simple policy is a higher-priority
strength project than statistically tuning these constants in place.

The watchdog checks every 50 ms. Root DFPN's cancellation timer and final
grace loop check every 1 ms.

## Hard-coded MCTS constants and policies

Sources: `dlshogi_mcts/uct_search.*`, `search_primitives.*`, `types.h`, and
`uct_node.*`.

| Constant/policy | Current value | Class | Notes |
|---|---:|---|---|
| Virtual loss | 1 | Implementation/tune later | Compile-time `kVirtualLoss`; no active USI multiplier |
| Position mutexes | 65,536 | Implementation | Hash-striped global mutex array; count must remain a power of two |
| Initial unevaluated node marker | -1 | Implementation | `kNotExpanded` |
| New leaf value | 0.5 | Semantics | Neutral win probability before NN/terminal result |
| Invalid inference containment value | value 0, MLH 0, uniform policy | Safety | Marks node evaluated, refuses backup/cache, and stops search |
| WDL/policy normalization tolerance | 1e-3 | Safety | Used for both fresh and cached output validation |
| Processed value tolerance | -1.001--1.001 | Safety | Allows small floating-point overshoot |
| CP logistic scale | 756 | Presentation | `-log(1/p - 1) * 756` |
| CP probability clamp | 0.001--0.999 | Safety/presentation | Prevents infinite CP |
| Non-finite final Q fallback | 0.5 / 0 CP | Safety | Logs one diagnostic and contains the value |
| Maximum PV walk | effectively 257 moves | Operational | Loop stops once `pv.size() > 256` |
| Trajectory initial reserve | 128 edges | Performance | Vector may still grow |
| Root tie break | higher prior | Search policy | Applied after equal visit counts |
| Proven move ordering | proven win first, proven loss skipped | Rule/search | Overrides visit count |
| Single legal root move | returned without MCTS | Search policy | No score/PV search is performed |
| Tree garbage collection | one background FIFO thread | Performance | Large discarded branches are destroyed off the move path |

The final move is the legal non-proven-losing root child with the most visits,
breaking ties by policy prior. Search `depth` and `seldepth` are the displayed
PV length, not alpha-beta depth. `multipv` is always 1.

Tree reuse requires the same starting-position hash and a new move history
that extends the previously searched history. Moving backward or changing the
game discards the affected tree. The tree currently has no node or memory
cap.

Search treats the first prior occurrence below the root as a repetition so it
can avoid cycles without looking ahead to the official fourth occurrence.
The live root is exempt from this independent search adjudication; the game
controller owns the actual game result.

## NN cache constants and policies

Source: `inference/nn_cache.h`.

| Constant/policy | Current value | Class | Notes |
|---|---:|---|---|
| Maximum shards | 256 | Implementation | Actual shard count is largest power of two not exceeding `min(capacity, 256)` |
| Hash-table reserve factor | 1.3 plus one | Performance | Allocation hint only |
| Eviction policy | FIFO per shard | Implementation/tune later | Hits do not refresh order |
| Cache key | 64-bit position hash | Format/safety | Cached value also stores legal-move count |
| In-flight key | hash plus legal-move count | Safety | Deduplicates simultaneous inference |
| Default capacity | 0 | Tune | Disabled until explicitly configured |
| `hashfull` scale | 0--1000 | USI | Cache occupancy in permill; zero when disabled |

The cache is shared by all GPUs and workers in one `Search`. Statistics reset
at each `go`; stored entries survive through the persistent `Search`.

## Neural inference and model constants

### Common post-processing

| Constant/policy | Current value | Class |
|---|---:|---|
| Policy softmax temperature | implicit 1.0 | Tune; not yet exposed |
| WDL softmax temperature | implicit 1.0 | Format/model |
| Invalid policy-label logit | -1000 | Safety |
| JHBR2 value | `P(win) - P(loss)` | Format |
| dlshogi scalar value | clamp to 0--1, draw probability 0 | Format |
| Missing MLH head | 0 plies | Format |

Only legal-move logits participate in policy softmax. A shared,
configurable `PolicyTemperature` is a planned tuning feature but is not
implemented in either backend yet.

### TensorRT backend

| Constant/policy | Current value | Class |
|---|---:|---|
| Fallback input channels before engine inspection | 48 | Defensive placeholder |
| Fallback maximum batch | 32 | Defensive placeholder when profile metadata is absent |
| Fallback policy width | 2187 | Format |
| One context/stream/buffer set | per worker slot | Performance |
| Oversized request | split at engine maximum batch | Performance |
| Static engine request | pad to engine batch | Format/performance |
| Tensor names, JHBR2 | `input_planes`, `policy`, `wdl`, optional `mlh` | Format |
| Tensor names, dlshogi | `input1`, `input2`, `output_policy`, `output_value` | Format |
| Diagnostic input hash | 64-bit FNV-1a | Diagnostics |
| FNV offset basis | 1469598103934665603 | Format/diagnostics |
| FNV prime | 1099511628211 | Format/diagnostics |

The JHBR2 encoder-channel mismatch is currently a warning, not a load error.
CUDA allocation helper failures are logged; later inference validation is the
final containment. These are safety-review items rather than tunable values.

### ONNX Runtime backend

| Constant/policy | Current value | Class |
|---|---:|---|
| Intra-op threads | 1 | Performance |
| Graph optimization | `ORT_ENABLE_ALL` | Performance |
| Provider order when `UseGPU=true` | CUDA, then CPU fallback | Operational |
| Required input | one dynamic-batch `(batch, C, 9, 9)` tensor | Format |
| Required outputs | policy and WDL; extra outputs accepted | Format |

A build with neither TensorRT nor ONNX Runtime now fails model initialization
explicitly. It no longer runs a silent uniform-policy, neutral-value engine.

## Mate-search constants

Sources: `mate/shallow_mate.h` and `mate/dfpn.*`.

| Constant/policy | Current value | Class | Notes |
|---|---:|---|---|
| Shallow depths compiled | 1, 3, 5, 7 | Tune/implementation | Only odd attacker-to-move mate depths |
| Three-ply countercheck handling | defender escape | Search approximation | Matches the documented dlshogi simplification |
| DFPN default constructor budget | 100,000 nodes | Operational | Callers normally pass an explicit limit |
| DFPN pool scale | 8x node limit | Performance | Capped before allocation |
| DFPN minimum pool | 1,024 nodes | Performance |
| DFPN maximum pool | 2,000,000 nodes | Performance/memory |
| Proof-number `INF` | `0xFFFFFFF0` | Implementation |
| Proof-number `MATE` | `INF - 2000` | Implementation |
| Unexpanded child sentinel | 255 | Implementation |
| Maximum recorded children | 254 | Implementation/safety | `uint8_t` reserves 255 as sentinel |
| DFPN repetition policy | any hash already on current path is no-mate | Search/rule approximation |
| OR-node choice | minimum proof number | Algorithm |
| AND-node choice | minimum disproof number | Algorithm |

DFPN allocates from a linear pool with no hash table or garbage collection.
The 254-child cap is structural and should receive focused correctness review
before being changed; it is not a strength knob.

## Opening-book constants and policies

Sources: `book/opening_book.*`, `book/book_selection.h`, and
`book/ybb_format.h`.

| Constant/policy | Current value | Class |
|---|---:|---|
| Runtime format | YBB V1 only | Format |
| Position lookup | binary search over PackedSfen index | Implementation |
| Move choice | highest stored signed 16-bit eval among legal moves | Search/book |
| Eval perspective | side to move | Format |
| Ties | first highest-eval record | Search/book |
| Stored YBB ply | ignored | Search/book |
| Malformed/illegal moves | skipped | Safety |
| Header size | 32 bytes | Format |
| Index record size | 44 bytes | Format |
| Move record size | 4 bytes, or 6 with depth flag | Format |
| Known flag | bit 0 means move depth is present | Format |
| PackedSfen key | 32 bytes | Format |

There is no runtime book randomness, minimum depth, move-count weighting, or
evaluation margin. The Gote exit margin belongs to offline generation and is
not part of the `jhbr2` binary.

## Shogi-rule and representation constants

These values are fixed by rules or model/file compatibility and must not be
included in an Elo parameter sweep.

| Area | Fixed values |
|---|---|
| Board | 9x9, 81 squares, three-rank promotion zones |
| Colors | Black/Sente = 0, White/Gote = 1 |
| Piece types | 15 indices including empty and promoted types |
| Hand types | 7 |
| Legal-move array capacity | 593, the proved exact maximum |
| Move encoding | 16 bits; 7-bit source, 7-bit destination, drop bit 14, promotion bit 15 |
| Bitboard | 81 bits split as 63 plus 18 |
| Entering king | king in enemy camp, not in check, at least 10 non-king camp pieces |
| Entering-king points | major pieces 5, others 1; Black threshold 28, White threshold 27 |
| PackedSfen | exactly 256 bits / 32 bytes |
| Physical set | P18, L4, N4, S4, B2, R2, G4 |
| Native input | 28 positional plus 120 uniform planes = 148 |
| Native packed input | 284 bytes positional plus 15 bytes uniform per position |
| dlshogi input | 62 feature-1 plus 57 feature-2 planes |
| Policy | 27 directions x 81 destinations = 2187 labels |
| Policy directions | 10 non-promotion, 10 promotion, 7 drops |
| Hand feature maxima | P8, L4, N4, S4, G4, B2, R2 per color |
| Nyugyoku features | 31 planes per color: king-in-camp 1, field remainder 10, point remainder 20 |

Zobrist hashes use deterministic SplitMix64-generated constants. Hash mixing
constants and the golden-ratio combine constant in local/cache keys are
implementation details, not tuning candidates.

## Build and diagnostic constants

Source: `CMakeLists.txt` and `inference/nn_diagnostics.h`.

| Setting | Current value | Class |
|---|---|---|
| C++ language | C++20 | Build |
| CUDA language | C++17 | Build |
| Release flags | `-O3 -DNDEBUG -march=native` | Performance/portability |
| Debug flags | `-g -O0 -fsanitize=address` | Diagnostics |
| CUDA architecture default | native when supported | Build/portability |
| `ENABLE_NN_DIAGNOSTICS` | ON | Diagnostics |
| Diagnostic emission | first failure process-wide only | Diagnostics |
| Containment when diagnostics OFF | remains active | Safety |
| TensorRT log threshold | warning and more severe | Diagnostics |

TensorRT engine-building profile sizes, FP16 selection, and workspace size are
not runtime `jhbr2` constants; they belong to the serialized engine artifact.
They still must be frozen and recorded in any performance or strength test.

## Retired compatibility options

These names are not advertised. If an old GUI sends one, JHBR2 reports that it
is retired and ignores it:

| Name | Reason |
|---|---|
| `NoiseEpsilon` | No root Dirichlet noise exists in USI play |
| `PerLeafGathering` | The active worker loop always gathers leaves |
| `LeafDfpnNodes` | Active leaf mate search is depth-bounded, not DFPN |
| `VirtualLossWeight` | The active virtual loss is the compile-time value 1 |
| `MaxGpuBatch` | Superseded by `MinibatchSize` plus the engine profile limit |
| `BookOnTheFly` | No active alternative runtime path |

Unknown options and invalid numeric values are now reported and ignored
instead of being falsely acknowledged as set. Floating-point tuning options
also reject NaN and infinity, and string paths may contain spaces.

## Audit cleanup completed with this inventory

The audit removed only code with no runtime effect:

- an unused ray-attack implementation superseded by the bitboard tables;
- an unused opponent-piece bitboard in general move generation;
- a no-op fixed-array `MoveList::reserve()` compatibility method and its only
  caller;
- an unused piece-character lookup;
- unused `SearchConfig::num_gpus`, `SearchResult::ponder_move`,
  `SearchResult::root_q`, and `USIEngine` game-ply/noise fields;
- a duplicate thread termination wrapper and a `Search::Run` overload whose
  argument was ignored;
- the silent no-backend uniform evaluator.

It also corrected stale policy-encoder and repetition comments and made the
advertised default Gote-book path match the actual member default.

## Recommended tuning order

1. Replace and test the current time manager before fine-tuning its fractions.
2. Tune `WorkersPerGpu`, `MinibatchSize`, and `NNCacheSize` for each GPU/model
   profile using fixed-time throughput and paired games.
3. Implement `PolicyTemperature`; screen it before final PUCT tuning.
4. Tune non-root `CInit`, `CBase`, and `FpuReduction` at fixed nodes.
5. Tune root PUCT/FPU only after non-root values stabilize.
6. Reconfirm `LeafMateDepth` and `RootMateDepth` with the tactical suite and
   paired games.
7. Tune MLH settings only for a model whose moves-left head was trained and
   validated.
8. Treat draw values and resign threshold as competition-specific safety
   experiments, with false-resign and repetition logs.

Do not tune format/rule constants, CP scale, info interval, diagnostic
thresholds, or hash constants for Elo. Every selected parameter must win a
fresh held-out paired match; choosing the best noisy result from a wide sweep
is not confirmation.
