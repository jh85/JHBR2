# HOW TO TRAIN

End-to-end recipe for training the **148-plane JHBR2 model** (dlshogi-style
input + WDL value + MLH moves-left head) from YaneuraOu `.pack` files, and
turning the result into a TensorRT engine the `jhbr2` binary can run.

Pipeline:

```
.pack files ──► gen_pack_shards.py ──► .npz shards ──► shogi_train.py ──► .onnx ──► trtexec ──► .engine
```

Two scripts do the work: **`gen_pack_shards.py`** (data) and
**`shogi_train.py`** (training). `trtexec` builds the engine. See
`docs/nyugyoku_dlshogi_features.md` for the plane layout and `HOW_TO_START.md`
for building/running the engine.

---

## 0. Prerequisites

- Python with PyTorch (CUDA build), NumPy, and `cshogi`.
- A CUDA GPU for training (CPU works but is only useful for a tiny smoke test).
- For the engine step: CUDA + TensorRT (see `HOW_TO_START.md`).

---

## 1. Pack files → training shards

Point `--pack-dir` at the directory holding **all** your `.pack` files; they
are processed in parallel.

```bash
python gen_pack_shards.py \
    --pack-dir /path/to/all_packs/ \
    --output-dir /workspace/pack_shards/ \
    --shard-size 500000 \
    --workers 16 \
    --eval-coef 600.0
```

Output: `/workspace/pack_shards/shard_000000.npz, shard_000001.npz, …`. Each
shard holds `planes (N,148,9,9) f16`, `policy int32` (∈[0,2187)), `wdl (N,3) f16`,
and `mlh int16` (remaining plies to game end — the MLH target).

Sanity-check first (a couple thousand positions, one worker):

```bash
python gen_pack_shards.py --pack-dir data --output-dir /tmp/probe \
    --limit 2000 --workers 1
```

> **Shard size vs. RAM.** Under DDP every GPU process loads a full shard into
> RAM, so peak ≈ `shard_size × ~24 KB × num_gpus`. For an 8-GPU run, prefer
> `--shard-size 100000` (~2.4 GB/shard → ~19 GB across 8 ranks) over 500k.
> Generate shards **once** and reuse them for every training run.

`gen_pack_shards.py` is pure-Python (~900 pos/s/worker); 16 workers ≈ 14k/s, so
~100M positions ≈ 2 h. (Other generators — `pack_to_shards.py`, `psv_to_shards.py`
— also produce 148-plane shards, but `gen_pack_shards.py` is the canonical,
verified one for packs.)

---

## 2. Train

The model is already configured for 148 planes + MLH; "from scratch" just means
**don't pass `--resume`**.

> ⚠️ **`--data` is a file PREFIX, not a directory.** The trainer detects shards
> by globbing `{--data}_*.npz`, so use `…/pack_shards/shard` (matches
> `…/pack_shards/shard_*.npz`). Passing the folder silently trains on synthetic
> data instead.

### Single GPU

```bash
python shogi_train.py \
    --data /workspace/pack_shards/shard \
    --epochs 20 --batch 1024 --lr 1e-3 \
    --save-dir checkpoints/ --save-every 1 --workers 8 \
    --export-onnx model_148.onnx
```

### Multiple GPUs (recommended: DDP via torchrun)

DDP runs one process per GPU with NCCL all-reduce — far better scaling than the
single-process `DataParallel` fallback. Just launch with `torchrun`; the script
auto-detects it.

```bash
# 2× GPU (e.g. test box)
torchrun --nproc_per_node=2 shogi_train.py \
    --data /workspace/pack_shards/shard \
    --epochs 20 --batch 2048 --lr 1e-3 \
    --save-dir checkpoints/ --save-every 1 --workers 8 \
    --export-onnx model_148.onnx

# 8× GPU
torchrun --nproc_per_node=8 shogi_train.py \
    --data /workspace/pack_shards/shard \
    --epochs 20 --batch 8192 --lr 2e-3 \
    --save-dir checkpoints/ --save-every 1 --workers 8 \
    --export-onnx model_148.onnx
```

- **`--batch` is per-GPU under DDP**, so global batch = `batch × nproc`. Keep
  ~1024/GPU and scale `--lr` up as the global batch grows.
- Pick GPUs with `CUDA_VISIBLE_DEVICES=0,1 torchrun …`.
- Only rank 0 logs, checkpoints, and exports. Checkpoints have no `module.`
  prefix issues — `--resume` handles them.

### Losses (what to watch)

The per-epoch line reports all three heads; **all should trend down**:

```
Epoch 3/20  loss=…  policy=…  value=…  mlh=…  lr=…  speed=… samples/sec
```

- `policy` — cross-entropy over the 2187 move labels.
- `value`  — cross-entropy over WDL.
- `mlh`    — Huber loss on clipped remaining plies. Tunable:
  - `--mlh-weight` (default `0.1`; set `0` to disable MLH training),
  - `--mlh-clip` (default `80` plies).

Add `--log-csv run.csv` to record per-epoch metrics.

---

## 3. Export → TensorRT engine

`--export-onnx` writes the ONNX at the end of training (or run
`checkpoint2onnx.py` on a checkpoint). It emits the tensor names the native
backend expects: `input_planes` / `policy` / `wdl` / `mlh`, dynamic batch,
148 channels.

Build the engine — **note `148`, not `48`** (the templates in `HOW_TO_START.md`
predate the 148-plane encoder):

```bash
$TENSORRT_PATH/bin/trtexec \
  --onnx=model_148.onnx \
  --saveEngine=engines/model_148.engine \
  --fp16 \
  --minShapes=input_planes:1x148x9x9 \
  --optShapes=input_planes:128x148x9x9 \
  --maxShapes=input_planes:128x148x9x9 \
  --memPoolSize=workspace:8192M
```

---

## 4. Run

```
setoption name OnnxModel value /path/to/engines/model_148.engine
```

The backend reads the engine's tensor names, auto-detects the **JHBR2** format,
and uses the packed-bits + GPU-unpack input path. (`ModelFormat` defaults to
`auto`; the `dlshogi` value is for the separate external-net validation path.)

### Optional: enable the moves-left (MLH) effect in search

Off by default. Once you've trained a model with the MLH head, you can have MCTS
prefer shorter wins / longer losses:

```
setoption name MovesLeftWeight value 0.03   # 0 disables; start small
setoption name MovesLeftThreshold value 0.1 # only act when clearly won/lost
```

Tune `MovesLeftWeight` with real games (e.g. sweep 0.01–0.05 vs. the baseline);
too high can cost strength.

---

## Quick reference

| Step | Command |
|------|---------|
| Shards | `python gen_pack_shards.py --pack-dir P --output-dir S --workers 16` |
| Train (1 GPU) | `python shogi_train.py --data S/shard --epochs 20 --batch 1024 --export-onnx m.onnx` |
| Train (N GPU) | `torchrun --nproc_per_node=N shogi_train.py --data S/shard --batch 1024×perGPU …` |
| Engine | `trtexec --onnx=m.onnx --saveEngine=m.engine --fp16 --*Shapes=input_planes:…x148x9x9` |
| Run | `setoption name OnnxModel value m.engine` |
