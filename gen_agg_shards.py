"""
Generate DEDUPLICATED / AGGREGATED training shards from YaneuraOu .pack and
.bin (PSV) teacher data for the JHBR2 model (148-plane input, 2187 policy,
WDL value head, MLH head).

Why: raw teacher data repeats positions (startpos: millions of times). Naive
per-record shards give those positions enormous sampling weight AND a one-hot
policy that flip-flops between different good moves across records. This
script aggregates by canonical position (AlphaZero-style):

    canonical position (HuffmanCodedPos, 32 bytes: board + side-to-move +
    hands — no move counter, no history)
        -> ONE training sample:
           policy  = distribution over ALL observed next moves
           wdl     = mean of the per-record blended WDL targets
           mlh     = mean remaining plies over records that have MLH
           count   = number of raw records aggregated

Policy aggregation methods (--policy-method):
    uniform_unique_moves  every observed move gets equal probability
    count                 probability proportional to raw count
    sqrt_count            proportional to sqrt(count)
    temperature           proportional to count ** alpha   (--alpha, default 0.5)
The vector is normalized over the observed moves (the training loss is a soft
cross-entropy over a normalized distribution).

Output shard format (aggshard_NNNNNN.npz), consumed by ShardedDataset in
shogi_train.py (which auto-detects the sparse policy arrays):
    planes          (N, 148, 9, 9) float16   (or packed1/packed2 with --packed)
    policy          (N,)      int32    top-weight move index (back-compat)
    policy_indices  (M,)      int16    ragged sparse policy: move indices
    policy_weights  (M,)      float16  ragged sparse policy: probabilities
    policy_offsets  (N+1,)    int64    sample i owns [offsets[i], offsets[i+1])
    wdl             (N, 3)    float16  (W, D, L), side-to-move's perspective
    mlh             (N,)      float16  mean remaining plies, -1 = no MLH label
    count           (N,)      uint32   raw records aggregated into this sample
    key             (N, 32)   uint8    HuffmanCodedPos (only with --store-key)

Scalability: two streaming phases, never holds full tensors for duplicates.
  Phase 1 (parallel over input files): decode records, write compact 40-byte
    intermediate rows (hcp, policy_idx, score, mlh, result) into partition
    files by hash(hcp) % --partitions.
  Phase 2 (parallel over partitions): sort each partition by hcp, group,
    aggregate, encode planes only for unique positions, write shards.
Memory per phase-2 worker ~ (total records * 40 B) / partitions; raise
--partitions for very large datasets (e.g. 4096 for billions of records).

MLH sources:
  .pack  true remaining plies to the recorded game end (game structure known).
  .bin   PSV has no game-end info. Default --psv-mlh none stores -1 (the
         training loss masks MLH for those samples). --psv-mlh recorded-end
         reconstructs games by gamePly continuity and uses plies to the last
         RECORDED position (an approximation; recording often stops early).

Usage:
    python gen_agg_shards.py \
        --pack-dir data/ --psv-dir data/ \
        --output-dir /workspace/agg_shards/ \
        --shard-size 500000 --workers 16 \
        --policy-method temperature --alpha 0.5 \
        --stats-json /workspace/agg_shards/stats.json

    # smoke test on the first 3000 records of each file
    python gen_agg_shards.py --pack-dir data/ --output-dir /tmp/probe \
        --limit 3000 --workers 1 --partitions 8
"""

import argparse
import glob as globmod
import heapq
import json
import os
import shutil
import struct
import sys
import time
from collections import Counter
from multiprocessing import Pool
from zlib import crc32

import numpy as np
import cshogi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shogi_train import move_to_policy_index, sfen_to_planes
import jhbr2_encoder

POLICY_SIZE = 2187
FAST_ENCODER = jhbr2_encoder.available()

# Compact intermediate record: everything needed to aggregate, no tensors.
# hcp is V32 (raw void bytes), NOT S32 — numpy's S dtype strips trailing
# NUL bytes, which would silently merge distinct HuffmanCodedPos keys.
REC_DTYPE = np.dtype([
    ("hcp", "V32"),      # canonical position: HuffmanCodedPos bytes
    ("move", "<i2"),     # policy index in [0, 2187)
    ("score", "<i2"),    # eval (cp), side-to-move's perspective
    ("mlh", "<i2"),      # remaining plies, -1 = unavailable
    ("result", "i1"),    # game result, side-to-move: +1 win / 0 draw / -1 loss
    ("pad", "u1"),
])
assert REC_DTYPE.itemsize == 40
_REC_PACK = struct.Struct("<32shhhbB")   # same 40-byte layout, for writing
assert _REC_PACK.size == REC_DTYPE.itemsize

TOP_K = 20               # "top duplicated positions" reported in stats


# =====================================================================
# Phase 1: streaming extraction into partition files
# =====================================================================

class PartitionWriter:
    """Buffered append-writer: record -> partition file by hash(hcp)."""

    def __init__(self, tmp_dir, n_partitions, file_id):
        self.tmp_dir = tmp_dir
        self.n = n_partitions
        self.file_id = file_id
        self.bufs = [bytearray() for _ in range(n_partitions)]
        self.written = 0

    def add(self, hcp_bytes, move_idx, score, mlh, result):
        p = crc32(hcp_bytes) % self.n
        buf = self.bufs[p]
        buf += _REC_PACK.pack(hcp_bytes, move_idx,
                              max(-32000, min(32000, int(score))),
                              max(-1, min(32000, int(mlh))), result, 0)
        self.written += 1
        if len(buf) >= (1 << 18):        # 256 KB
            self._flush(p)

    def _flush(self, p):
        if not self.bufs[p]:
            return
        path = os.path.join(self.tmp_dir, f"part_{p:04d}",
                            f"f{self.file_id:06d}.bin")
        with open(path, "ab") as f:
            f.write(self.bufs[p])
        self.bufs[p] = bytearray()

    def close(self):
        for p in range(self.n):
            self._flush(p)


def result_stm(game_result_abs, black_to_move):
    """Absolute result (0 draw / 1 black / 2 white) -> side-to-move ±1/0."""
    if game_result_abs == 0:
        return 0
    black_won = (game_result_abs == 1)
    return 1 if black_won == black_to_move else -1


def extract_pack_file(task):
    """Phase 1 for one .pack file. Mirrors gen_pack_shards.py's decoder."""
    (path, file_id, tmp_dir, n_partitions, limit) = task
    from YaneShogiLib import GameDataDecoder   # resolved via --script-lib path

    with open(path, "rb") as f:
        decoder = GameDataDecoder(bytearray(f.read()))

    writer = PartitionWriter(tmp_dir, n_partitions, file_id)
    board = cshogi.Board()
    hcp_arr = np.empty(1, dtype=cshogi.HuffmanCodedPos)
    stats = Counter()

    while not decoder.eof():
        if limit and writer.written >= limit:
            break
        # Read one whole game; a decoder failure here loses the game boundary.
        try:
            sfen = decoder.get_sfen()
            game_kif = []
            game_result_abs = 0
            while True:
                move = decoder.read_uint16()
                if (move & 0x7F) == ((move >> 7) & 0x7F):
                    game_result_abs = move & 0x7F
                    decoder.read_uint8()
                    break
                eval16 = decoder.read_int16()
                game_kif.append((move, eval16))
        except Exception as e:
            print(f"  {path}: decoder broke: {type(e).__name__}: {e}",
                  file=sys.stderr)
            break

        stats["games"] += 1
        n_moves = len(game_kif)
        if n_moves == 0:
            continue

        try:
            board.set_sfen(sfen)
            for i, (move, eval16) in enumerate(game_kif):
                if move == 0:
                    stats["bad_move"] += 1
                    break
                black_to_move = (board.turn == cshogi.BLACK)
                policy_idx = move_to_policy_index(
                    cshogi.move_to_usi(move), not black_to_move)
                if 0 <= policy_idx < POLICY_SIZE:
                    board.to_hcp(hcp_arr)
                    writer.add(hcp_arr.tobytes(), policy_idx, eval16,
                               n_moves - i - 1,   # plies left after this move
                               result_stm(game_result_abs, black_to_move))
                    stats["records"] += 1
                else:
                    stats["unmappable_move"] += 1
                board.push_move16(move)
        except Exception:
            stats["skipped_games"] += 1

    writer.close()
    return path, dict(stats)


def extract_psv_file(task):
    """Phase 1 for one .bin (PSV) file."""
    (path, file_id, tmp_dir, n_partitions, limit, psv_mlh) = task

    n_records = os.path.getsize(path) // cshogi.PackedSfenValue.itemsize
    if limit:
        n_records = min(n_records, limit)
    records = np.memmap(path, dtype=cshogi.PackedSfenValue, mode="r",
                        shape=(n_records,))

    writer = PartitionWriter(tmp_dir, n_partitions, file_id)
    board = cshogi.Board()
    hcp_arr = np.empty(1, dtype=cshogi.HuffmanCodedPos)
    stats = Counter()

    def emit(rec, mlh):
        move_raw = int(rec["move"])
        if move_raw == 0:
            stats["bad_move"] += 1
            return
        try:
            board.set_psfen(np.ascontiguousarray(rec["sfen"], dtype=np.uint8))
        except Exception:
            stats["bad_position"] += 1
            return
        black_to_move = (board.turn == cshogi.BLACK)
        # PSV stores YaneuraOu-format Move16 — convert to cshogi encoding
        # first (drops/promotions differ), like the reference converter does.
        move16 = cshogi.move16_from_psv(move_raw)
        policy_idx = move_to_policy_index(
            cshogi.move_to_usi(move16), not black_to_move)
        if not (0 <= policy_idx < POLICY_SIZE):
            stats["unmappable_move"] += 1
            return
        board.to_hcp(hcp_arr)
        writer.add(hcp_arr.tobytes(), policy_idx, int(rec["score"]), mlh,
                   int(rec["game_result"]))   # already side-to-move ±1/0
        stats["records"] += 1

    if psv_mlh == "recorded-end":
        # Reconstruct games by gamePly continuity (see psv_to_shards.py) and
        # use plies-to-last-RECORDED-position as an approximate MLH.
        game = []
        prev_ply = None
        for i in range(n_records):
            rec = records[i]
            ply = int(rec["gamePly"])
            if prev_ply is not None and ply != prev_ply + 1:
                final = int(game[-1]["gamePly"])
                for r in game:
                    emit(r, max(0, final - int(r["gamePly"])))
                stats["games"] += 1
                game = []
            game.append(rec)
            prev_ply = ply
        if game:
            final = int(game[-1]["gamePly"])
            for r in game:
                emit(r, max(0, final - int(r["gamePly"])))
            stats["games"] += 1
    else:
        # True moves-left is not recoverable from PSV records: store -1 and
        # let the training loss mask MLH for these samples.
        for i in range(n_records):
            emit(records[i], -1)

    writer.close()
    return path, dict(stats)


# =====================================================================
# Phase 2: per-partition aggregation -> shards
# =====================================================================

def policy_weights_from_counts(counts, method, alpha):
    """counts: (K,) int array -> normalized (K,) float64 weights."""
    c = counts.astype(np.float64)
    if method == "uniform_unique_moves":
        w = np.ones_like(c)
    elif method == "count":
        w = c
    elif method == "sqrt_count":
        w = np.sqrt(c)
    elif method == "temperature":
        w = c ** alpha
    else:
        raise ValueError(f"unknown policy method: {method}")
    return w / w.sum()


def stable_winrate(scores, eval_coef):
    return 1.0 / (1.0 + np.exp(-np.clip(
        scores.astype(np.float64) / eval_coef, -30.0, 30.0)))


class ShardBuffer:
    """Accumulates aggregated samples and writes aggshard_NNNNNN.npz files."""

    def __init__(self, output_dir, shard_id_base, shard_size, packed,
                 store_key):
        self.output_dir = output_dir
        self.shard_id = shard_id_base
        self.shard_size = shard_size
        self.packed = packed
        self.store_key = store_key
        self.paths = []
        self._reset()

    def _reset(self):
        self.sfens, self.keys = [], []
        self.p_idx, self.p_wt, self.p_len = [], [], []
        self.top1, self.wdl, self.mlh, self.count = [], [], [], []

    def add(self, sfen, key, idxs, wts, wdl, mlh, count):
        self.sfens.append(sfen)
        self.keys.append(key)
        self.p_idx.append(idxs)
        self.p_wt.append(wts)
        self.p_len.append(len(idxs))
        self.top1.append(int(idxs[int(np.argmax(wts))]))
        self.wdl.append(wdl)
        self.mlh.append(mlh)
        self.count.append(count)
        if len(self.sfens) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.sfens:
            return
        n = len(self.sfens)
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(self.p_len, out=offsets[1:])
        arrays = dict(
            policy=np.asarray(self.top1, dtype=np.int32),
            policy_indices=np.concatenate(self.p_idx).astype(np.int16),
            policy_weights=np.concatenate(self.p_wt).astype(np.float16),
            policy_offsets=offsets,
            wdl=np.asarray(self.wdl, dtype=np.float16),
            mlh=np.asarray(self.mlh, dtype=np.float16),
            count=np.asarray(self.count, dtype=np.uint32),
        )
        if self.store_key:
            arrays["key"] = np.frombuffer(
                b"".join(self.keys), dtype=np.uint8).reshape(n, 32)
        # Encode planes only now, only for unique positions, in batches.
        if self.packed:
            p1, p2 = [], []
            for i in range(0, n, 8192):
                a, b = jhbr2_encoder.pack_sfens(self.sfens[i:i + 8192])
                p1.append(a)
                p2.append(b)
            arrays["packed1"] = np.concatenate(p1)
            arrays["packed2"] = np.concatenate(p2)
        else:
            planes = np.empty((n, 148, 9, 9), dtype=np.float16)
            for i in range(0, n, 8192):
                chunk = self.sfens[i:i + 8192]
                if FAST_ENCODER:
                    planes[i:i + len(chunk)] = \
                        jhbr2_encoder.encode_sfens(chunk).astype(np.float16)
                else:
                    for j, s in enumerate(chunk):
                        planes[i + j] = sfen_to_planes(s).astype(np.float16)
            arrays["planes"] = planes
        path = os.path.join(self.output_dir,
                            f"aggshard_{self.shard_id:06d}.npz")
        np.savez_compressed(path, **arrays)
        self.paths.append(path)
        self.shard_id += 1
        self._reset()


def aggregate_partition(task):
    """Phase 2 for one partition: group by hcp, aggregate, write shards."""
    (part_idx, tmp_dir, output_dir, shard_size, shard_id_base, eval_coef,
     method, alpha, max_moves, packed, store_key, startpos_hcp) = task

    part_dir = os.path.join(tmp_dir, f"part_{part_idx:04d}")
    files = sorted(globmod.glob(os.path.join(part_dir, "*.bin")))
    if not files:
        return part_idx, {}, []
    recs = np.concatenate([np.fromfile(f, dtype=REC_DTYPE) for f in files])

    order = np.argsort(recs["hcp"], kind="stable")
    recs = recs[order]
    keys, starts, counts = np.unique(recs["hcp"], return_index=True,
                                     return_counts=True)
    winrate = stable_winrate(recs["score"], eval_coef)

    board = cshogi.Board()
    hcp_arr = np.empty(1, dtype=cshogi.HuffmanCodedPos)
    shards = ShardBuffer(output_dir, shard_id_base, shard_size, packed,
                         store_key)

    stats = {
        "raw_records": int(len(recs)),
        "unique_positions": int(len(keys)),
        "unique_pairs": 0,
        "moves_per_pos_hist": Counter(),
        "dup_positions": 0,                 # count >= 2
        "result_conflict_positions": 0,     # >1 distinct game result
        "winrate_std_sum": 0.0,
        "winrate_std_max": 0.0,
        "mlh_with": 0,
        "mlh_without": 0,
        "mlh_std_sum": 0.0,
        "mlh_std_n": 0,
        "truncated_moves": 0,
        "top": [],                          # (count, sfen, [(usi, cnt)...],
                                            #  value stats, mlh stats)
        "startpos": None,
    }
    top_heap = []

    for g in range(len(keys)):
        s, c = int(starts[g]), int(counts[g])
        e = s + c
        moves, mcounts = np.unique(recs["move"][s:e], return_counts=True)
        stats["unique_pairs"] += len(moves)
        stats["moves_per_pos_hist"][min(len(moves), 10)] += 1

        w = policy_weights_from_counts(mcounts, method, alpha)
        if len(moves) > max_moves:                 # keep heaviest moves
            keep = np.argsort(w)[::-1][:max_moves]
            stats["truncated_moves"] += len(moves) - max_moves
            moves, w, mcounts = moves[keep], w[keep], mcounts[keep]
            w = w / w.sum()

        wr = winrate[s:e]
        mean_wr = float(wr.mean())
        res = recs["result"][s:e]
        n_win = int(np.count_nonzero(res == 1))
        n_draw = int(np.count_nonzero(res == 0))
        n_loss = c - n_win - n_draw
        wdl = (0.7 * mean_wr + 0.3 * n_win / c,
               0.3 * n_draw / c,
               0.7 * (1.0 - mean_wr) + 0.3 * n_loss / c)

        mlh_vals = recs["mlh"][s:e]
        mlh_vals = mlh_vals[mlh_vals >= 0]
        if len(mlh_vals):
            mlh = float(mlh_vals.mean(dtype=np.float64))
            stats["mlh_with"] += 1
        else:
            mlh = -1.0
            stats["mlh_without"] += 1

        if c >= 2:
            stats["dup_positions"] += 1
            if len(np.unique(res)) > 1:
                stats["result_conflict_positions"] += 1
            wstd = float(wr.std())
            stats["winrate_std_sum"] += wstd
            stats["winrate_std_max"] = max(stats["winrate_std_max"], wstd)
            if len(mlh_vals) >= 2:
                stats["mlh_std_sum"] += float(mlh_vals.std())
                stats["mlh_std_n"] += 1

        key_bytes = bytes(keys[g])           # V32 -> full 32 raw bytes
        board.set_hcp(np.frombuffer(key_bytes, dtype=cshogi.HuffmanCodedPos))
        sfen = board.sfen()

        is_startpos = (key_bytes == startpos_hcp)
        if is_startpos or len(top_heap) < TOP_K or c > top_heap[0][0]:
            flip = (board.turn == cshogi.WHITE)
            by_weight = np.argsort(w)[::-1]
            entry = {
                "count": c, "sfen": sfen,
                "moves": [[_policy_idx_to_usi(int(moves[j]), flip, board),
                           int(mcounts[j]), round(float(w[j]), 4)]
                          for j in by_weight],
                "winrate_mean": round(mean_wr, 4),
                "winrate_std": round(float(wr.std()), 4),
                "winrate_min": round(float(wr.min()), 4),
                "winrate_max": round(float(wr.max()), 4),
                "results_wdl": [n_win, n_draw, n_loss],
                "mlh_mean": round(mlh, 2),
                "mlh_std": round(float(mlh_vals.std()), 2) if len(mlh_vals)
                           else -1.0,
            }
            if is_startpos:
                stats["startpos"] = entry
            heapq.heappush(top_heap, (c, g, entry))
            if len(top_heap) > TOP_K:
                heapq.heappop(top_heap)

        shards.add(sfen, key_bytes, moves.astype(np.int16),
                   w.astype(np.float32), wdl, mlh, c)

    shards.flush()
    stats["top"] = [t[2] for t in sorted(top_heap, reverse=True,
                                         key=lambda t: t[0])]
    stats["moves_per_pos_hist"] = dict(stats["moves_per_pos_hist"])
    return part_idx, stats, shards.paths


def _policy_idx_to_usi(policy_idx, flip, board):
    """Best-effort reverse mapping (for the stats report only): find the legal
    move on `board` whose policy index matches."""
    for mv in board.legal_moves:
        usi = cshogi.move_to_usi(mv)
        if move_to_policy_index(usi, flip) == policy_idx:
            return usi
    return f"idx{policy_idx}"


# =====================================================================
# Driver
# =====================================================================

def merge_stats(parts, phase1):
    g = {
        "raw_records": sum(s.get("records", 0) for s in phase1.values()),
        "games": sum(s.get("games", 0) for s in phase1.values()),
        "unmappable_moves": sum(s.get("unmappable_move", 0)
                                for s in phase1.values()),
        "bad_moves": sum(s.get("bad_move", 0) for s in phase1.values()),
        "bad_positions": sum(s.get("bad_position", 0)
                             for s in phase1.values()),
        "skipped_games": sum(s.get("skipped_games", 0)
                             for s in phase1.values()),
        "unique_positions": 0, "unique_pairs": 0, "dup_positions": 0,
        "result_conflict_positions": 0, "mlh_with": 0, "mlh_without": 0,
        "truncated_moves": 0,
        "moves_per_pos_hist": Counter(),
        "winrate_std_max": 0.0,
        "_wstd_sum": 0.0, "_mlh_std_sum": 0.0, "_mlh_std_n": 0,
        "top": [], "startpos": None,
    }
    for s in parts:
        if not s:
            continue
        for k in ("unique_positions", "unique_pairs", "dup_positions",
                  "result_conflict_positions", "mlh_with", "mlh_without",
                  "truncated_moves"):
            g[k] += s[k]
        g["moves_per_pos_hist"].update(s["moves_per_pos_hist"])
        g["winrate_std_max"] = max(g["winrate_std_max"], s["winrate_std_max"])
        g["_wstd_sum"] += s["winrate_std_sum"]
        g["_mlh_std_sum"] += s["mlh_std_sum"]
        g["_mlh_std_n"] += s["mlh_std_n"]
        g["top"].extend(s["top"])
        if s["startpos"]:
            g["startpos"] = s["startpos"]
    g["top"] = sorted(g["top"], key=lambda t: -t["count"])[:TOP_K]
    g["duplicate_ratio"] = (1.0 - g["unique_positions"] /
                            max(1, g["raw_records"]))
    g["winrate_std_mean_dup"] = (g.pop("_wstd_sum") /
                                 max(1, g["dup_positions"]))
    n = g.pop("_mlh_std_n")
    g["mlh_std_mean_dup"] = g.pop("_mlh_std_sum") / max(1, n)
    g["moves_per_pos_hist"] = {str(k): v for k, v in
                               sorted(g["moves_per_pos_hist"].items())}
    return g


def print_report(g, n_shards, shard_size):
    print("\n================ AGGREGATION REPORT ================")
    print(f"raw records processed        : {g['raw_records']:,}")
    print(f"games (pack / psv boundaries): {g['games']:,}")
    print(f"unique positions             : {g['unique_positions']:,}")
    print(f"unique (position, move) pairs: {g['unique_pairs']:,}")
    print(f"duplicate ratio              : {g['duplicate_ratio']:.4f}")
    print(f"positions seen >= 2 times    : {g['dup_positions']:,}")
    print(f"value conflicts (mixed results): "
          f"{g['result_conflict_positions']:,}")
    print(f"winrate std (dup positions)  : mean {g['winrate_std_mean_dup']:.4f}"
          f", max {g['winrate_std_max']:.4f}")
    print(f"MLH std (dup positions)      : mean {g['mlh_std_mean_dup']:.2f}")
    print(f"samples with / without MLH   : {g['mlh_with']:,} / "
          f"{g['mlh_without']:,}")
    print(f"unmappable / bad moves       : {g['unmappable_moves']:,} / "
          f"{g['bad_moves']:,}")
    print(f"bad positions, skipped games : {g['bad_positions']:,}, "
          f"{g['skipped_games']:,}")
    print(f"policy moves truncated       : {g['truncated_moves']:,}")
    print(f"shards written               : {n_shards} "
          f"(<= {shard_size:,} samples each)")
    print("moves-per-position histogram (10 = '10 or more'):")
    for k, v in g["moves_per_pos_hist"].items():
        print(f"    {k:>3} moves: {v:,}")
    if g["startpos"]:
        sp = g["startpos"]
        print(f"startpos count               : {sp['count']:,}")
        print(f"startpos policy              : {sp['moves']}")
    print("top duplicated positions:")
    for t in g["top"]:
        print(f"    x{t['count']:<8,} {t['sfen']}")
        print(f"        moves={t['moves'][:6]}")
        print(f"        winrate mean/std/min/max = {t['winrate_mean']}/"
              f"{t['winrate_std']}/{t['winrate_min']}/{t['winrate_max']}  "
              f"W-D-L={t['results_wdl']}  mlh={t['mlh_mean']}"
              f"±{t['mlh_std']}")
    print("====================================================")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack-dir", help="Directory of .pack files")
    p.add_argument("--pack-glob", default="*.pack")
    p.add_argument("--psv-dir", help="Directory of .bin (PSV) files")
    p.add_argument("--psv-glob", default="*.bin")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-size", type=int, default=500_000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--eval-coef", type=float, default=600.0)
    p.add_argument("--policy-method", default="temperature",
                   choices=["uniform_unique_moves", "count", "sqrt_count",
                            "temperature"])
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Exponent for --policy-method temperature")
    p.add_argument("--max-moves", type=int, default=32,
                   help="Keep at most this many moves per position "
                        "(heaviest first, renormalized)")
    p.add_argument("--psv-mlh", default="none",
                   choices=["none", "recorded-end"],
                   help="MLH for PSV records: 'none' masks it (-1); "
                        "'recorded-end' approximates via gamePly continuity")
    p.add_argument("--partitions", type=int, default=256,
                   help="Hash partitions; phase-2 RAM ~ records*40B/partitions")
    p.add_argument("--tmp-dir", default=None,
                   help="Intermediate partition files "
                        "(default: <output-dir>/agg_tmp; deleted on success)")
    p.add_argument("--keep-tmp", action="store_true")
    p.add_argument("--packed", action="store_true",
                   help="Bit-packed planes (needs pyext C++ encoder)")
    p.add_argument("--store-key", action="store_true",
                   help="Store the 32-byte HuffmanCodedPos per sample "
                        "(for verification/debugging)")
    p.add_argument("--limit", type=int, default=0,
                   help="Max records per input file (testing)")
    p.add_argument("--stats-json", default=None)
    p.add_argument("--script-lib", default=None,
                   help="Path to YaneuraOu-ScriptCollection/CommonLib "
                        "(for the .pack GameDataDecoder)")
    args = p.parse_args()

    if not args.pack_dir and not args.psv_dir:
        p.error("need --pack-dir and/or --psv-dir")
    if args.packed and not FAST_ENCODER:
        print("ERROR: --packed requires the C++ encoder (bash pyext/build.sh)",
              file=sys.stderr)
        return 1

    pack_files = sorted(globmod.glob(os.path.join(
        args.pack_dir, args.pack_glob))) if args.pack_dir else []
    psv_files = sorted(globmod.glob(os.path.join(
        args.psv_dir, args.psv_glob))) if args.psv_dir else []
    if not pack_files and not psv_files:
        print("No input files found.", file=sys.stderr)
        return 1

    if pack_files:
        # GameDataDecoder lives in YaneuraOu-ScriptCollection/CommonLib.
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [args.script_lib] if args.script_lib else [
            os.path.join(here, "YaneuraOu-ScriptCollection", "CommonLib"),
            os.path.join(here, "YaneuraOu-ScriptCollection", "GenSfen"),
            os.path.join(os.path.dirname(here), "YaneuraOu-ScriptCollection",
                         "CommonLib"),
        ]
        for c in candidates:
            if c and os.path.isdir(c):
                sys.path.insert(0, c)
        try:
            from YaneShogiLib import GameDataDecoder  # noqa: F401
        except ImportError:
            print("ERROR: cannot import YaneShogiLib.GameDataDecoder — pass "
                  "--script-lib /path/to/YaneuraOu-ScriptCollection/CommonLib",
                  file=sys.stderr)
            return 1

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = args.tmp_dir or os.path.join(args.output_dir, "agg_tmp")
    if os.path.isdir(tmp_dir) and os.listdir(tmp_dir):
        print(f"ERROR: tmp dir {tmp_dir} is not empty (stale run?). "
              f"Remove it or pass a fresh --tmp-dir.", file=sys.stderr)
        return 1
    for pidx in range(args.partitions):
        os.makedirs(os.path.join(tmp_dir, f"part_{pidx:04d}"), exist_ok=True)

    print(f"Inputs: {len(pack_files)} .pack + {len(psv_files)} .bin | "
          f"workers={args.workers} partitions={args.partitions} "
          f"policy={args.policy_method}"
          f"{f'(alpha={args.alpha})' if args.policy_method == 'temperature' else ''} "
          f"psv-mlh={args.psv_mlh}")
    print(f"Encoder: {'C++ (fast)' if FAST_ENCODER else 'pure Python (slow — bash pyext/build.sh)'}"
          f" | shard planes: {'bit-packed' if args.packed else 'float16'}")

    # ---------------- Phase 1 ----------------
    t0 = time.time()
    tasks = []
    fid = 0
    for f in pack_files:
        tasks.append(("pack", (f, fid, tmp_dir, args.partitions, args.limit)))
        fid += 1
    for f in psv_files:
        tasks.append(("psv", (f, fid, tmp_dir, args.partitions, args.limit,
                              args.psv_mlh)))
        fid += 1

    phase1_stats = {}
    with Pool(args.workers) as pool:
        results = []
        for kind, t in tasks:
            fn = extract_pack_file if kind == "pack" else extract_psv_file
            results.append(pool.apply_async(fn, (t,)))
        for r in results:
            path, st = r.get()
            phase1_stats[path] = st
            print(f"  extracted {os.path.basename(path)}: "
                  f"{st.get('records', 0):,} records "
                  f"({st.get('games', 0):,} games, "
                  f"{st.get('unmappable_move', 0)} unmappable)")
    print(f"Phase 1 done in {time.time() - t0:.0f}s: "
          f"{sum(s.get('records', 0) for s in phase1_stats.values()):,} records")

    # ---------------- Phase 2 ----------------
    t1 = time.time()
    startpos_hcp = np.empty(1, dtype=cshogi.HuffmanCodedPos)
    cshogi.Board().to_hcp(startpos_hcp)
    startpos_hcp = startpos_hcp.tobytes()

    SHARD_ID_BUDGET = 10_000
    tasks2 = [
        (pidx, tmp_dir, args.output_dir, args.shard_size,
         pidx * SHARD_ID_BUDGET, args.eval_coef, args.policy_method,
         args.alpha, args.max_moves, args.packed, args.store_key,
         startpos_hcp)
        for pidx in range(args.partitions)
    ]
    part_stats, all_paths = [], []
    with Pool(args.workers) as pool:
        for pidx, st, paths in pool.imap_unordered(aggregate_partition,
                                                   tasks2):
            part_stats.append(st)
            all_paths.extend(paths)
    print(f"Phase 2 done in {time.time() - t1:.0f}s")

    g = merge_stats(part_stats, phase1_stats)
    print_report(g, len(all_paths), args.shard_size)
    if args.stats_json:
        with open(args.stats_json, "w") as f:
            json.dump(g, f, indent=2)
        print(f"Stats JSON: {args.stats_json}")

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Output: {args.output_dir} "
          f"(train with --data {os.path.join(args.output_dir, 'aggshard')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
