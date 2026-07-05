"""
Generate training shards from YaneuraOu .pack files for the improved JHBR2
model (148-plane dlshogi-style input, MLH head, 2187 direction policy).

This is the canonical generator for the current encoder. It routes every
position through shogi_train.sfen_to_planes(), which is verified bit-identical
to the C++ PackShogiPosition() / GPU unpack kernel (see
docs/nyugyoku_dlshogi_features.md), so training input == inference input.

What each position produces:
    planes  (148, 9, 9) float16  features1 piece bitmaps (28) + features2
                                 one-hot: hand thermometer (56), check (1),
                                 nyugyoku (62), repetition (1, always 0 here —
                                 a bare position has no history).
    policy  int32                direction-based index in [0, 2187), or the
                                 record is dropped if the move can't be mapped.
    wdl     (3,) float16         (W, D, L) from the side-to-move's perspective,
                                 blended 0.7*eval-winrate + 0.3*game-result.
    mlh     int16                raw remaining plies to game end (NO clipping;
                                 apply --mlh-clip at training time so it's tunable).

Output: shard_NNNNNN.npz files with arrays {planes, policy, wdl, mlh}, the same
format ShardedDataset in shogi_train.py consumes (and mixable with psv shards).

Usage:
    python gen_pack_shards.py \
        --pack-dir data/ \
        --output-dir /workspace/pack_shards/ \
        --shard-size 500000 \
        --workers 16 \
        --eval-coef 600.0

    # quick smoke test: only the first 2000 positions of each file
    python gen_pack_shards.py --pack-dir data/ --output-dir /tmp/probe \
        --limit 2000 --workers 1
"""

import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import cshogi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "YaneuraOu-ScriptCollection", "GenSfen"))
from shogi_train import sfen_to_planes, move_to_policy_index
from ShogiCommonLib import GameDataDecoder
import jhbr2_encoder

POLICY_SIZE = 2187
# Fast C++ encoder (pyext/libjhbr2_encoder.so) if built, else Python fallback.
# ~50x faster; produces bit-identical planes. Build with: bash pyext/build.sh
FAST_ENCODER = jhbr2_encoder.available()


def encode_planes(sfens):
    """Batch-encode a list of SFENs -> (N, 148, 9, 9) float16.

    Uses the compiled C++ encoder when available (bit-identical to
    sfen_to_planes), else falls back to pure Python."""
    if not sfens:
        return np.empty((0, 148, 9, 9), dtype=np.float16)
    if FAST_ENCODER:
        return jhbr2_encoder.encode_sfens(sfens).astype(np.float16)
    return np.stack([sfen_to_planes(s) for s in sfens]).astype(np.float16)


def position_meta(board, move_raw, score, game_result_abs, eval_coef):
    """
    Compute (sfen, policy_idx, wdl) for the current `board` state, or None if
    the move is missing/unmappable. Planes are encoded separately, in a batch
    per game (the C++ encoder computes the check plane itself, so we no longer
    read it from the cshogi board here).

    game_result_abs: 0=draw, 1=BLACK win, 2=WHITE win (absolute).
    """
    if move_raw == 0:
        return None

    sfen = board.sfen()
    flip = sfen.split()[1] == 'w'          # WHITE to move
    is_black = not flip

    policy_idx = move_to_policy_index(cshogi.move_to_usi(move_raw), flip)
    if policy_idx < 0 or policy_idx >= POLICY_SIZE:
        return None

    # WDL target from the side-to-move's perspective.
    if game_result_abs == 0:
        hard = (0.0, 1.0, 0.0)
    elif (game_result_abs == 1 and is_black) or \
         (game_result_abs == 2 and not is_black):
        hard = (1.0, 0.0, 0.0)
    else:
        hard = (0.0, 0.0, 1.0)

    win_rate = 1.0 / (1.0 + math.exp(-score / eval_coef))
    wdl = (0.7 * win_rate + 0.3 * hard[0],
           0.0 + 0.3 * hard[1],
           0.7 * (1.0 - win_rate) + 0.3 * hard[2])

    return sfen, policy_idx, wdl


def flush_shard(shard_id, output_dir, planes, policy, wdl, mlh):
    out_path = os.path.join(output_dir, f"shard_{shard_id:06d}.npz")
    np.savez_compressed(
        out_path,
        planes=np.asarray(planes, dtype=np.float16),
        policy=np.asarray(policy, dtype=np.int32),
        wdl=np.asarray(wdl, dtype=np.float16),
        mlh=np.asarray(mlh, dtype=np.int16),
    )
    return out_path


def process_pack_file(task):
    """Process one .pack file into one or more shards."""
    pack_path, shard_id_base, output_dir, shard_size, eval_coef, limit = task

    with open(pack_path, "rb") as f:
        decoder = GameDataDecoder(bytearray(f.read()))

    planes_buf, policy_buf, wdl_buf, mlh_buf = [], [], [], []
    shard_id = shard_id_base
    written = errors = games = skipped_games = 0
    board = cshogi.Board()

    while not decoder.eof():
        if limit and written + len(planes_buf) >= limit:
            break

        # PHASE 1: read one whole game. A decoder failure here is unrecoverable
        # (we lose the next game boundary), so stop this file.
        try:
            sfen = decoder.get_sfen()
            game_kif = []
            game_result_abs = 0
            while True:
                move = decoder.read_uint16()
                if (move & 0x7f) == ((move >> 7) & 0x7f):
                    game_result_abs = move & 0x7f
                    decoder.read_uint8()
                    break
                eval16 = decoder.read_int16()
                game_kif.append((move, eval16))
        except Exception as e:
            print(f"  {pack_path}: decoder broke at game {games}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            break

        games += 1
        n_moves = len(game_kif)
        if n_moves == 0:
            continue

        # PHASE 2: replay + encode. We're at a game boundary, so a failure here
        # only loses this one game.
        try:
            board.set_sfen(sfen)
            # Collect per-position metadata; encode planes in one batch per game.
            g_sfens, g_policy, g_wdl, g_mlh = [], [], [], []
            for i, (move, eval16) in enumerate(game_kif):
                mlh_target = n_moves - i - 1   # raw remaining plies (no clip)
                rec = position_meta(board, move, eval16, game_result_abs,
                                    eval_coef)
                if rec is None:
                    errors += 1
                else:
                    s, policy_idx, wdl = rec
                    g_sfens.append(s)
                    g_policy.append(policy_idx)
                    g_wdl.append(wdl)
                    g_mlh.append(mlh_target)
                try:
                    board.push_move16(move)
                except Exception:
                    errors += 1
                    break

            # Batch-encode this game's planes (fast C++ path), then buffer/flush.
            if g_sfens:
                g_planes = encode_planes(g_sfens)
                for k in range(len(g_sfens)):
                    planes_buf.append(g_planes[k])
                    policy_buf.append(g_policy[k])
                    wdl_buf.append(g_wdl[k])
                    mlh_buf.append(g_mlh[k])
                    if len(planes_buf) >= shard_size:
                        flush_shard(shard_id, output_dir, planes_buf,
                                    policy_buf, wdl_buf, mlh_buf)
                        written += len(planes_buf)
                        planes_buf, policy_buf, wdl_buf, mlh_buf = [], [], [], []
                        shard_id += 1
                if limit and written + len(planes_buf) >= limit:
                    break
        except Exception as e:
            skipped_games += 1
            if skipped_games <= 10:
                print(f"  {pack_path}: skipped game {games}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)

    if planes_buf:
        flush_shard(shard_id, output_dir, planes_buf,
                    policy_buf, wdl_buf, mlh_buf)
        written += len(planes_buf)
        shard_id += 1

    return pack_path, games, written, errors, skipped_games


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack-dir", required=True, help="Directory of .pack files")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-size", type=int, default=500_000,
                   help="Positions per shard (default 500k)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--eval-coef", type=float, default=600.0,
                   help="Sigmoid coefficient mapping centipawns -> win rate")
    p.add_argument("--limit", type=int, default=0,
                   help="Max positions per pack file (0 = unlimited). For tests.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pack_files = sorted(
        os.path.join(args.pack_dir, f)
        for f in os.listdir(args.pack_dir)
        if f.endswith(".pack")
    )
    if not pack_files:
        print(f"No .pack files found in {args.pack_dir}", file=sys.stderr)
        return

    print(f"Found {len(pack_files)} .pack files, workers={args.workers}, "
          f"shard_size={args.shard_size}, limit={args.limit or 'none'}")
    if FAST_ENCODER:
        print("Encoder: C++ (pyext/libjhbr2_encoder.so) — fast")
    else:
        print("Encoder: pure Python (SLOW). Build the fast one: bash pyext/build.sh")

    # Each file gets its own block of shard IDs so workers never collide.
    PER_FILE_SHARD_BUDGET = 100_000
    tasks = [
        (path, idx * PER_FILE_SHARD_BUDGET, args.output_dir,
         args.shard_size, args.eval_coef, args.limit)
        for idx, path in enumerate(pack_files)
    ]

    t0 = time.time()
    total_games = total_written = total_errors = total_skipped = 0
    with Pool(args.workers) as pool:
        for pack_path, games, written, errors, skipped in pool.imap_unordered(
                process_pack_file, tasks):
            total_games += games
            total_written += written
            total_errors += errors
            total_skipped += skipped
            elapsed = time.time() - t0
            print(f"  done {os.path.basename(pack_path)}: {games:,} games "
                  f"({skipped} skipped), {written:,} positions "
                  f"(errors={errors}; total {total_written:,} in {elapsed:.0f}s, "
                  f"{total_written/max(elapsed,1):.0f}/s)")

    print(f"\nFinished: {total_games:,} games, {total_skipped} skipped games, "
          f"{total_written:,} positions, {total_errors} dropped positions")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
