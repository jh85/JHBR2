# JHBR2 engine-strength roadmap and agent handoff

Last updated: 2026-07-24
Branch at handoff: `feat/dlshogi-nyugyoku-148planes`

## Purpose

This is the durable handoff for improving JHBR2's playing strength without
assuming that a stronger self-play network will solve everything. Read this
document before starting another search change. Older design documents remain
useful for implementation detail, but some of their status sections predate
the work listed below.

The user's approximate external estimate is 3500--3600 Elo for JHBR2, versus
roughly 4200 for the top engines in that pool. Those numbers are useful as a
goal, not as a controlled measurement: hardware, opponents, books, clocks, and
rating-pool composition differ. Internal paired matches are the decision
instrument for code changes.

The governing rule is:

> Correctness first, then measured strength per node, then measured speed, and
> finally production-clock strength. NPS alone is not a strength result.

## Current repository state and user-owned files

- The current search backend is `dlshogi_mcts/`; the retired lc0 MCTS backend
  has already been removed.
- `build.sh` has an uncommitted user change selecting a different model and
  TensorRT workspace. Do not overwrite or include it in unrelated commits.
- `check_shard.py` is an untracked user file. Do not add, edit, or delete it
  without an explicit request.
- TensorRT `.engine` files and generated strength-test runs are intentionally
  untracked.
- One-shot NN diagnostics and containment are still present. They were added
  while investigating the rare `score cp -2147483648` failure. The failure did
  not recur in 24 diagnostic games, but its original software/hardware cause
  was not conclusively proved. Keep the containment until equivalent cloud
  exposure has run long enough and all diagnostic counters remain clean.
- The Gote-only early-exit YBB feature is implemented. Sente can play without
  a book while Gote uses `user_book1_gote_exit.ybb`.

## Completed foundations -- do not redo

| Area | Status | Commit or path |
|---|---|---|
| Exact legal-move capacity bound | Complete | `91ec837` |
| Concurrent NN cache with in-flight deduplication and statistics | Complete | `c9a0655`, `inference/nn_cache.h` |
| Persistent subtree reuse | Complete | `d127b53`, `dlshogi_mcts/uct_node.*` |
| Independent root-repetition correction | Complete | `dd1d546` |
| Standard USI `info` reporting for depth, seldepth, CP, nodes, NPS, time, and PV | Complete | `dd1d546`, `usi/search_info.*` |
| Terminal-value backup and dlshogi PUCT semantics | Complete | `d72cefd`, `dlshogi_mcts/search_primitives.*` |
| Non-finite NN/tree containment and one-shot diagnostics | Complete, investigation remains open | `fc5212f` |
| Shallow mate search through 7 plies and root mate guard | Complete | `mate/shallow_mate.h`, `5f36892` |
| Specialized mate-in-one routine | Complete | `0d18270` |
| Root DFPN timing/proof handling | Complete foundation, still tunable | `4790bff` |
| Native YBB reader and Gote early-exit book generator | Complete | `7d1c188`, `94e541b` |
| 148-plane nyugyoku feature path | Implemented on this branch | `docs/nyugyoku_dlshogi_features.md` |
| Portable paired strength-test harness | Complete in the commit containing this document | `tools/strength_test.py`, `docs/STRENGTH_TESTING.md` |

Related detailed documents:

- `docs/STRENGTH_TESTING.md`
- `docs/ENGINE_PARAMETER_INVENTORY.md`
- `docs/port_5ply_mate_check_plan.md`
- `docs/port_specialized_mate1_plan.md`
- `docs/port_yaneuraou_check_generator_plan.md`
- `docs/GOTE_EXIT_BOOK.md`
- `docs/nyugyoku_dlshogi_features.md`
- `docs/shard_generation_audit_2026-07.md`

`docs/architecture_improvements_research.md` is historical. In particular, its
statements that JHBR2 lacks an NN cache or shallow mate implementation are no
longer current.

## Phase 0 -- establish the strength laboratory

This phase is implemented, but it must now be used consistently.

### Required workflow

1. Freeze a baseline binary, its TensorRT engine, its Git revision, and all USI
   options. Never rebuild a file in place and continue calling it the
   baseline.
2. Generate one deterministic YBB-derived opening suite and retain its hash.
3. Run two color-reversed games from every selected opening.
4. Screen algorithmic changes with a fixed node count.
5. Confirm candidates under the real GPU topology and real clock.
6. Change one factor per match.
7. Retain `config.json`, `pairs.jsonl`, `summary.json`, and engine logs for
   every decision.
8. Use the paired interval and LOS, not a small raw W-L record. Nine games do
   not establish a strength change.

The command recipes and resume behavior are in
`docs/STRENGTH_TESTING.md`.

### Two different questions require two tests

- **Fixed nodes:** did search quality improve for the same nominal work?
- **Fixed time:** did the complete engine become stronger after including
  speed, batching, GPU scaling, and time management?

A speed optimization can lose fixed-node Elo but win fixed-time Elo. A search
change can do the opposite. Record both instead of reducing everything to NPS.

### Calibration before trusting a new environment

- Run a two-pair smoke test.
- Run baseline versus the same baseline. The result should be compatible with
  50%, with no illegal moves, time forfeits, or failed pairs.
- Confirm that all requested GPUs are visible in `config.json`.
- Watch peak GPU memory: each match worker keeps two engine processes and two
  model copies resident on its GPU group.
- Start the eight-RTX-5090 screening topology with
  `--gpus-per-worker 1`; reserve `--gpus-per-worker 8` for final
  production-topology confirmation.

### Harness limitations to remember

- It currently reports a fixed-size paired confidence interval, not SPRT.
- It does not perform evaluation adjudication; games finish by shogi rules,
  resign/declaration, time forfeit, or `--max-plies`.
- The generated opening suite is test data, not an instruction to enable the
  engine's playing book.
- Results from one-GPU workers are excellent for screening but are not a
  substitute for the final eight-GPU topology.

## Phase 1 -- correctness audit and refactoring with oracles

Refactoring is valuable when it exposes invariants, removes misleading
interfaces, or makes differential testing possible. Refactoring by itself is
not assumed to add Elo.

### 1.1 Search invariants

Add deterministic tests around:

- value perspective at every edge and backup;
- terminal win/loss/draw flags;
- fourfold repetition and perpetual-check win/loss;
- entering-king declaration;
- max-move draws and checkmate precedence;
- virtual-loss add/remove balance on every exit path;
- finite `win`, `sum_m`, priors, values, and CP conversion;
- visit accounting for queued, cached, terminal, discarded, and invalid
  evaluations;
- root move legality after mate rejection;
- cold-tree versus reused-tree behavior from the same position;
- cancellation and stop paths during TensorRT failures.

Prefer small pure functions such as those in
`dlshogi_mcts/search_primitives.*`. Every discovered bug should receive the
smallest reproducing position and a permanent regression test.

### 1.2 Board differential testing

Generate deterministic random legal games and compare JHBR2 with cshogi and,
where practical, YaneuraOu:

- complete legal-move sets;
- check/evasion/checking-move sets;
- do/undo and position hashes;
- SFEN and PackedSfen round trips;
- repetition classification;
- mate-in-one and shallow-mate verdicts;
- declaration-win eligibility;
- policy-index conversion.

Run this on at least hundreds of thousands of positions before changing a
core move generator. Keep a seed and first failing SFEN in the output.

### 1.3 Concurrency and lifetime testing

- Build a CPU/mock-inference configuration suitable for ThreadSanitizer.
- Stress tree pruning/reuse while workers stop and restart.
- Stress NN-cache owner/waiter cancellation.
- Vary workers, batch size, and GPU count.
- Assert that no in-flight virtual losses remain after a completed search.
- Add fault injection for short inference result arrays and non-finite
  policy/value/moves-left outputs.

### 1.4 Keep USI options truthful

The 2026-07-24 parameter audit stopped advertising `NoiseEpsilon`,
`PerLeafGathering`, `LeafDfpnNodes`, and `VirtualLossWeight`. Old
configurations receive an explicit retired-option message instead of a false
acknowledgement. Core PUCT/FPU, draw, resign, and info-interval settings are
now active USI options. Keep
`docs/ENGINE_PARAMETER_INVENTORY.md` synchronized whenever a default, bound,
or search constant changes.

### Acceptance gate

- All unit and property tests pass in release and sanitizer builds.
- A long baseline match has zero protocol failures, illegal moves, non-finite
  containment events, and unexplained engine restarts.
- Search semantic changes are committed separately from mechanical
  refactoring.

## Phase 2 -- expose and statistically tune search parameters

Do not optimize hard-coded values by running an enormous Cartesian grid.
First expose them as USI options, validate that changing each option changes
the intended code path, then use staged paired experiments.

### Current high-value parameters

| Parameter | Current value | Initial experiment range | Notes |
|---|---:|---:|---|
| Non-root `c_init` | 1.25 | 0.75--2.0 | Primary PUCT exploration strength |
| Non-root `c_base` | 19652 | 1,000--100,000 log scale | Controls visit-dependent exploration |
| Non-root FPU reduction | 0.27 | 0.0--0.6 | Strong network-dependent parameter |
| Root `c_init` | 1.25 | 0.75--2.0 | Tune separately only after non-root |
| Root `c_base` | 19652 | 1,000--100,000 log scale | Often weakly identifiable at short searches |
| Root FPU reduction | 0.0 | 0.0--0.4 | Can affect policy lock-in |
| Resign threshold | 0.01 win probability | 0--0.03 | Measure false resignations, not only Elo |
| Draw value, Black/White | 0.5/0.5 | 0.47--0.53 | Tune only for the target competition rules |
| Leaf mate depth | 5 | 1, 3, 5, 7 | Measure tactical Elo and node cost |
| Root mate depth | 7 | 0, 5, 7 | Already subjectively valuable; confirm |
| NN cache size | 0 default | 0, 250K, 1M, 4M | Record hit rate, RAM, NPS, and Elo |
| Workers per GPU | 2 | 4, 8, 12, 16, 24, 32 on RTX 5090 | Hardware/model/profile specific; monitor GPU memory |
| Minibatch size | 128 | 32, 64, 96, 128, 192, 256 | Stop at the TensorRT profile maximum; rebuild the engine profile to test a true larger batch |
| Moves-left weight | 0 | small positive values | Only with a valid trained MLH head |
| Moves-left threshold | 0 | 0--0.3 | Tune jointly after enabling MLH |
| Moves-left cap | 20 | 5--40 | Tune jointly after enabling MLH |
| Policy softmax temperature | implicit 1.0 | 0.7--1.5 | Must be implemented before testing |

Other constants are classified in
`docs/ENGINE_PARAMETER_INVENTORY.md`. Important examples are:

- `kVirtualLoss = 1` (there is no advertised runtime multiplier);
- 65,536 position mutexes;
- CP display scale 756;
- info interval;
- root-DFPN time/node bands;
- time-manager fractions and safety margins;
- max moves to draw.

The CP scale and info interval mostly affect presentation and should not be
treated as playing-strength parameters.

### Parameter-tuning protocol

1. Tune GPU topology (`WorkersPerGpu`, `MinibatchSize`, cache size) per hardware
   and TensorRT profile first.
2. Tune search semantics at fixed nodes using the frozen topology.
3. Use coarse one-factor or fractional-factorial screening.
4. Race only promising regions; do not spend equal games on clearly losing
   candidates.
5. Confirm the selected setting on a held-out opening suite.
6. Confirm again at production time control and production GPU count.
7. Re-test interactions only among the few surviving parameters.
8. Correct for multiple comparisons: a parameter chosen as the best of many
   noisy trials must win a fresh confirmation match.

SPSA, CLOP, Bayesian optimization, or a racing scheduler can reduce cost, but
the objective must remain paired game score. Do not optimize CP agreement or
NPS as a proxy for Elo.

## Phase 3 -- replace the simple time manager

This is probably the highest-value unimplemented non-network improvement.

### Current weakness

`usi/time_manager.cc` currently behaves as follows:

- if byoyomi is present, it allocates 90% of byoyomi;
- otherwise it allocates 5% of remaining main time plus 80% of increment;
- it does not model moves to go, root uncertainty, or best-move stability.

Consequently, under a 300+10 control, the first branch wins and the nominal
MCTS allocation is about 9 seconds per move. The 300-second main clock is not
deliberately invested. This can leave a large amount of playing strength
unused.

### Proposed design

Create independently testable soft and hard budgets:

1. Reserve a configurable emergency margin.
2. Estimate moves remaining from game ply; later blend in a calibrated MLH
   prediction when available.
3. Allocate a base share of remaining main time plus byoyomi/increment.
4. Permit extensions when:
   - the top root moves have close visit counts or values;
   - the best move changes late;
   - root entropy is high;
   - the position is tactical or a root mate search remains active.
5. Stop early when the best move and value have been stable for a meaningful
   window.
6. Apply increasingly conservative caps below 60, 30, and 10 seconds.
7. Fit root DFPN inside the same hard deadline instead of giving it a loosely
   coupled budget.
8. Record the reason for every early stop or extension for offline analysis.

### Required tests

- Simulated full games never exceed the clock, including OS scheduling grace.
- 300+10 actually spends some main time in important positions.
- Low-time behavior cannot request a negative or zero accidental budget.
- Mate search cannot overrun the hard deadline.
- Fixed-position tests cover opening, quiet middlegame, tactical middlegame,
  entering-king, and last-minute states.
- A/B confirmation uses the real 300+10 clock, not only fixed nodes.

## Phase 4 -- search-quality improvements

Work through these as isolated experiments, not one rewrite.

### 4.1 Policy calibration

Implement `PolicyTemperature` in both ONNX Runtime and TensorRT paths. Apply it
to legal logits before softmax, with one shared tested function. Compare policy
entropy and paired fixed-node strength. Network policy calibration often
changes the best PUCT constants, so tune temperature before the final PUCT
sweep.

### 4.2 Root decision quality

- Record the top root moves, visits, Q, and prior in diagnostic match data.
- Test more robust final selection when visits are close.
- Investigate a small tactical verification budget for the top two or three
  candidates.
- Preserve the existing root mate-7 guard and measure how often it rejects the
  original most-visited move.
- Consider sequential-halving or Gumbel-style root search only as a separate,
  reversible experiment; it is not automatically superior for this network.

### 4.3 Deep tactical verification

Do not run 9/11-ply exhaustive checks at every leaf. Prefer:

- cheap 3/5-ply checks at leaves;
- the current 7-ply root blunder guard;
- a concurrent PV/root DFPN worker for deeper mates;
- proof reuse between consecutive searches where safe;
- a strict shared hard time deadline.

Build a tactical regression suite from every observed sudden mate, missed
mate, invalid declaration, and repetition error. Track solved rate and time in
addition to game Elo.

### 4.4 Parallel-search efficiency

- Tune workers and batch size separately on RTX 3090 and RTX 5090.
- Measure GPU utilization, inference latency distribution, batch occupancy,
  cache hit/wait rates, duplicate leaves, and discarded trajectories.
- Implement a real virtual-loss parameter before attempting to tune it.
- Investigate adaptive batching near the end of a time budget.
- Revisit mutex count or lock layout only after profiling shows contention.

### 4.5 Tree and transposition work

- Measure retained node count and useful reused visits after every played move.
- Add a memory cap/pruning policy if long games grow without bound.
- Test reuse-on versus reuse-off at fixed time as well as fixed nodes.
- A full MCTS transposition DAG is a high-risk project because backup and
  repetition context become more complicated. Investigate it only after the
  tree, cache, and correctness measurements show enough duplicate search work
  to justify the complexity.

## Phase 5 -- tactical and endgame strength

### Tactical corpus

Maintain versioned suites for:

- mate in 1, 3, 5, 7, 9, and 11;
- avoiding an opponent mate in those distances;
- sacrifice/recapture positions;
- perpetual-check win/loss;
- declaration-win races;
- positions extracted from JHBR2 losses.

The suite should include the expected best move or acceptable move set, the
proof source, and a fixed time/node budget. It supplements game matches; it
does not replace them.

### Endgame search

- Verify declaration rules and repetition before adding heuristics.
- Use deeper root/PV DFPN selectively in low-material tactical endings.
- Investigate small material-limited retrograde databases only where the state
  space is tractable. A general shogi tablebase is not.
- Add endgame-specific time extensions only after the main time manager exists.
- Measure false resignations and declaration opportunities missed.

## Phase 6 -- opening strategy

The specialized Gote YBB pipeline is complete, but its playing policy still
needs empirical selection.

### Experiments

- Generate margins such as 0, 10, 20, 30, 50, and 100 CP.
- Measure median and distribution of the ply where the game leaves the source
  book.
- Measure Gote score, not only exit depth.
- Confirm that Sente has `BookFile` empty and does not use the Gote book.
- Rebuild from the publisher's monthly YBB with
  `tools/generate_gote_exit_book.sh`.
- Keep book experiments out of general search A/B tests unless the book itself
  is the factor under test.

The objective is constrained optimization: leave early while remaining within
an acceptable book-evaluation margin. Earliest exit alone can choose losing
lines.

## Phase 7 -- nyugyoku and mixture-of-experts proposal

The user's idea of a regular-game model plus a nyugyoku expert is feasible,
but a hard two-model switch should not be the first intervention.

### First establish the opportunity

1. Classify a large game corpus by entering-king likelihood and game phase.
2. Measure the current model's calibration and move accuracy in that subset.
3. Verify that the 148-plane nyugyoku features are populated identically in
   training, ONNX export, TensorRT, cache, and inference.
4. Quantify how many losses are actually attributable to nyugyoku evaluation
   rather than search, declaration, or time management.

### Safer progression

1. Oversample/fine-tune on nyugyoku and long endgames.
2. Try a shared trunk with an auxiliary nyugyoku head or phase-conditioned
   features.
3. Try a small expert adapter.
4. Only then try two complete models.

If two full models are used:

- use a continuous, hysteretic gate rather than switching on one brittle
  threshold;
- include side-to-move, king zones, declaration points, hand material, ply,
  and both kings' progress in the gate;
- test positions near the boundary for discontinuous value/policy changes;
- account for doubled GPU memory and model-loading cost;
- version and hash both models in match configuration;
- compare against an ensemble/blend near the gate.

The expert must win paired games on a nyugyoku-heavy held-out suite and not
regress the general suite.

## Phase 8 -- targeted training other than ordinary self-play

These are network projects, but they capture the user's proposed directions
and should remain in the roadmap.

### Opening-book training

- Use book positions for coverage, not blindly as ground-truth policy.
- Weight moves by book evaluation, depth, and visit reliability.
- Include alternative good moves rather than a single one-hot label.
- Hold out entire opening families to detect memorization.
- Mix with normal positions so opening training does not damage middlegames.

### Mate and endgame curricula

- Generate proved mate-in-1/3/5/7/9/11 positions and defender-to-move
  counterparts.
- Store mate distance, winning move set, and proof provenance.
- Train value and policy consistently; avoid assigning a mating value to
  positions whose label is only "not solved within the limit."
- Include near-mate negatives so the network does not hallucinate attacks.
- Add entering-king races, declaration positions, long endgames, and
  repetition choices.

### Data quality before model size

- Deduplicate by position while preserving a multi-move policy target.
- Track age, source engine/network, search budget, and result reliability.
- Balance openings, middlegames, tactical endings, and nyugyoku.
- Audit feature planes and labels after every format change.
- Evaluate every checkpoint through the frozen strength harness.

Larger networks, gated attention, MoE, or deeper residual stacks should be
compared at equal wall-clock search strength, not equal training loss.

## Phase 9 -- performance engineering

The NN cache and tree reuse raised NPS substantially without an equally large
observed strength gain. This is plausible: extra visits have diminishing
returns, and search correctness, tactics, time allocation, or policy quality
can dominate.

Optimize only measured bottlenecks:

- CPU move generation and encoding;
- TensorRT host/device copies;
- actual rather than configured batch occupancy;
- synchronization and mutex contention;
- duplicate evaluations and cache waits;
- end-of-search underfilled batches;
- tree allocation/pruning.

For every performance patch:

1. run correctness/property tests;
2. measure the focused microbenchmark;
3. measure end-to-end NPS and GPU utilization;
4. run fixed-node A/B to detect semantic drift;
5. run fixed-time A/B to establish actual benefit.

Do not bundle a search-policy change into a performance commit.

## Experiment record template

Create one short Markdown or JSON record per consequential experiment:

```text
Hypothesis:
Baseline Git/binary/model hashes:
Candidate Git/binary/model hashes:
Only changed factor:
Opening-suite hash:
Node or time control:
GPU model/count and topology:
USI options:
Pairs completed / failed:
W-L-D and pentanomial:
Score, Elo interval, LOS:
NPS, cache, batching, and memory observations:
Protocol/diagnostic failures:
Decision:
Run directory:
```

Negative results are valuable and should be retained so a later agent does not
repeat them.

## Recommended order for the next agent

1. Read this file and `docs/STRENGTH_TESTING.md`.
2. Freeze the current binary/model as the baseline.
3. Run harness calibration and a short eight-GPU fixed-node baseline.
4. Implement and test a 300+10-aware adaptive time manager.
5. Confirm it under the actual clock.
6. Implement and expose `PolicyTemperature`; the core PUCT/FPU constants are
   already truthful USI options.
7. Run coarse fixed-node screening, then held-out and production-clock
   confirmation.
8. Continue the correctness differential suite in parallel with parameter
   work.
9. Build the tactical-loss corpus from real JHBR2 records.
10. Investigate the nyugyoku expert only after quantifying its addressable loss
    rate.

## Definition of a successful change

A change is ready to become the default only when:

- its implementation has targeted regression tests;
- it produces no new diagnostic/protocol failures;
- it has a clean isolated commit;
- fixed-node behavior is understood;
- paired fixed-time evidence supports it on a held-out suite;
- the result is reproduced with the production GPU topology;
- model, binary, options, opening hash, and raw match data are retained;
- any hardware-specific default is documented as such.
