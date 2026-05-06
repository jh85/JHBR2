"""
Pre-compute YaneuraOu .pack files directly into numpy shards
(skipping the HCPE intermediate so we can preserve remaining-plies
information for MLH training).

Pipeline replaces:
    pack file --pack2hcpe.py--> hcpe --precompute_cshogi.py--> shards
with:
    pack file --pack_to_shards.py--> shards (with mlh_target)

Each shard is a .npz with arrays:
    planes  (N, 48, 9, 9)  float16
    policy  (N,)           int32   in [0, 2187), or -1 if move missing
    wdl     (N, 3)         float16 (W, D, L) from side-to-move's perspective
    mlh     (N,)           int16   remaining plies from this position to
                                   game end (clamped to [0, 200])

Usage:
    python pack_to_shards.py \
        --pack-dir /path/to/pack_files/ \
        --output-dir /workspace/pack_precomputed/ \
        --shard-size 500000 \
        --workers 16 \
        --eval-coef 600.0 \
        --mlh-clip 200
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


def encode_one_position(board, move_raw, score, game_result_abs,
                        eval_coef=600.0):
    """
    Build (planes, policy_idx, wdl) for the current board state.
    `game_result_abs` is the absolute game result: 0=draw, 1=BLACK win,
    2=WHITE win. The wdl target is from the side-to-move's perspective.
    Returns None if the move is missing or invalid.
    """
    sfen = board.sfen()
    flip = sfen.split()[1] == 'w'
    is_black = not flip

    planes = sfen_to_planes(sfen)

    if move_raw == 0:
        return None
    move_usi = cshogi.move_to_usi(move_raw)
    policy_idx = move_to_policy_index(move_usi, flip)
    if policy_idx < 0 or policy_idx >= 2187:
        return None

    # Side-to-move WDL from absolute game result
    if game_result_abs == 0:
        hard = [0.0, 1.0, 0.0]
    elif (game_result_abs == 1 and is_black) or \
         (game_result_abs == 2 and not is_black):
        hard = [1.0, 0.0, 0.0]
    else:
        hard = [0.0, 0.0, 1.0]

    win_rate = 1.0 / (1.0 + math.exp(-score / eval_coef))
    wdl = [0.7 * win_rate + 0.3 * hard[0],
           0.0 + 0.3 * hard[1],
           0.7 * (1.0 - win_rate) + 0.3 * hard[2]]

    return planes, policy_idx, wdl


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


def process_pack_file(args):
    """Process a single .pack file → one or more shards."""
    pack_path, shard_id_base, output_dir, shard_size, eval_coef, mlh_clip = args

    with open(pack_path, "rb") as f:
        data = bytearray(f.read())
    decoder = GameDataDecoder(data)

    # Buffers for the current shard
    planes_buf, policy_buf, wdl_buf, mlh_buf = [], [], [], []
    shard_id = shard_id_base
    written = 0
    errors = 0
    games = 0

    board = cshogi.Board()

    try:
        while not decoder.eof():
            # Game start: get initial SFEN, set up board
            sfen = decoder.get_sfen()
            board.set_sfen(sfen)

            # Phase 1: read all (move, eval) until the game-result marker
            game_kif = []
            game_result_abs = 0
            while True:
                move = decoder.read_uint16()
                # End-of-game marker: from/to squares are equal
                sq1 = move & 0x7f
                sq2 = (move >> 7) & 0x7f
                if sq1 == sq2:
                    game_result_abs = sq1   # 0=draw, 1=BLACK win, 2=WHITE win
                    decoder.read_uint8()    # status / reason byte
                    break
                eval16 = decoder.read_int16()
                game_kif.append((move, eval16))

            games += 1
            n_moves = len(game_kif)
            if n_moves == 0:
                continue

            # Phase 2: replay the game, emit one record per move
            for i, (move, eval16) in enumerate(game_kif):
                remaining_plies = n_moves - i - 1
                # Clip large values: a draw-shuffle of 300 plies isn't
                # informative; capping at e.g. 200 lets MLH focus on
                # near-end positions where the signal matters.
                mlh_target = min(remaining_plies, mlh_clip)

                rec = encode_one_position(board, move, eval16,
                                          game_result_abs, eval_coef)
                if rec is None:
                    errors += 1
                else:
                    planes, policy_idx, wdl = rec
                    planes_buf.append(planes)
                    policy_buf.append(policy_idx)
                    wdl_buf.append(wdl)
                    mlh_buf.append(mlh_target)

                # Advance the board (use the original raw move from pack)
                try:
                    board.push_move16(move)
                except Exception:
                    errors += 1
                    break

                # Flush full shards
                if len(planes_buf) >= shard_size:
                    flush_shard(shard_id, output_dir, planes_buf,
                                policy_buf, wdl_buf, mlh_buf)
                    written += len(planes_buf)
                    planes_buf, policy_buf, wdl_buf, mlh_buf = [], [], [], []
                    shard_id += 1

    except Exception as e:
        print(f"  {pack_path}: stopped at game {games}: {e}", file=sys.stderr)

    # Flush any partial final shard
    if planes_buf:
        flush_shard(shard_id, output_dir, planes_buf,
                    policy_buf, wdl_buf, mlh_buf)
        written += len(planes_buf)
        shard_id += 1

    return pack_path, games, written, errors, shard_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack-dir", required=True, help="Directory of .pack files")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-size", type=int, default=500_000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--eval-coef", type=float, default=600.0)
    p.add_argument("--mlh-clip", type=int, default=200,
                   help="Clamp remaining_plies to this value for MLH target")
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

    print(f"Found {len(pack_files)} .pack files, workers={args.workers}")

    # Each pack file gets its own block of shard IDs (rough upper bound:
    # file_size / 40 / shard_size shards per file).
    PER_FILE_SHARD_BUDGET = 10_000
    tasks = []
    for idx, path in enumerate(pack_files):
        shard_id_base = idx * PER_FILE_SHARD_BUDGET
        tasks.append((path, shard_id_base, args.output_dir,
                      args.shard_size, args.eval_coef, args.mlh_clip))

    t0 = time.time()
    total_games = total_written = total_errors = 0
    with Pool(args.workers) as pool:
        for pack_path, games, written, errors, _ in pool.imap_unordered(
                process_pack_file, tasks):
            total_games += games
            total_written += written
            total_errors += errors
            elapsed = time.time() - t0
            print(f"  done {os.path.basename(pack_path)}: "
                  f"{games:,} games, {written:,} positions "
                  f"(errors={errors}, total {total_written:,} "
                  f"in {elapsed:.0f}s, {total_written/max(elapsed,1):.0f}/s)")

    print(f"\nFinished: {total_games:,} games, {total_written:,} positions, "
          f"{total_errors} errors")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
