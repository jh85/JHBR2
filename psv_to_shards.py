"""
Pre-compute YaneuraOu .bin (PSV — PackedSfenValue) files directly into
numpy shards with MLH targets included.

Sister script to pack_to_shards.py. The two scripts produce shards with
identical structure so they can be mixed in training.

PSV format vs pack format differs in:
  - PSV is a flat array of fixed 40-byte records (no game markers).
  - PSV records ARE stored sequentially within games (gamePly increases
    by exactly 1 across consecutive records). Game boundaries are
    detected by gamePly NOT increasing by 1.
  - PSV game_result is already from the side-to-move's perspective
    (+1 = side-to-move won, -1 = lost, 0 = draw), unlike pack/HCPE
    where it's absolute.

Output shard format (same as pack_to_shards.py):
    planes  (N, 48, 9, 9)  float16
    policy  (N,)           int32   in [0, 2187), or -1 if move missing
    wdl     (N, 3)         float16 (W, D, L) from side-to-move's view
    mlh     (N,)           int16   raw remaining plies to (recorded)
                                   game end. Apply --mlh-clip at training
                                   time, not here.

Caveat for MLH: in PSV data, the LAST recorded position of a game is
not necessarily the actual game-end ply (selfplay often stops recording
when eval is decisive). So `remaining_plies` here measures "plies until
the recording stops," which is an approximation of "plies until the
game ends." For most training purposes this is close enough.

Usage:
    python psv_to_shards.py \\
        --psv-dir /mnt/shogi_data/psv_files/ \\
        --output-dir /mnt/shogi_data/psv_shards/ \\
        --shard-size 500000 \\
        --workers 4 \\
        --eval-coef 600.0
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
from shogi_train import sfen_to_planes, move_to_policy_index


PSV_DTYPE = cshogi.PackedSfenValue   # itemsize=40
PER_FILE_SHARD_BUDGET = 100_000      # large because PSV files can be huge (1B+)


def encode_one_psv(board, psv_record, eval_coef):
    """
    Build (planes, policy_idx, wdl) for one PSV record.
    Returns None if the move is missing or invalid.

    PSV game_result is already from the side-to-move's perspective:
        +1 = side-to-move won
         0 = draw
        -1 = side-to-move lost
    """
    board.set_psfen(np.ascontiguousarray(psv_record['sfen'], dtype=np.uint8))
    sfen = board.sfen()
    flip = sfen.split()[1] == 'w'

    planes = sfen_to_planes(sfen)

    move_raw = int(psv_record['move'])
    if move_raw == 0:
        return None
    move_usi = cshogi.move_to_usi(move_raw)
    policy_idx = move_to_policy_index(move_usi, flip)
    if policy_idx < 0 or policy_idx >= 2187:
        return None

    score = int(psv_record['score'])
    game_result = int(psv_record['game_result'])

    # PSV game_result is already side-to-move-relative.
    if game_result == 1:
        hard = [1.0, 0.0, 0.0]
    elif game_result == 0:
        hard = [0.0, 1.0, 0.0]
    else:                         # -1
        hard = [0.0, 0.0, 1.0]

    win_rate = 1.0 / (1.0 + math.exp(-score / eval_coef))
    wdl = [0.7 * win_rate + 0.3 * hard[0],
           0.0           + 0.3 * hard[1],
           0.7 * (1 - win_rate) + 0.3 * hard[2]]

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


def is_game_boundary(prev_ply, cur_ply):
    """Detect a game boundary: gamePly does not increase by exactly 1."""
    if prev_ply is None:
        return False
    return cur_ply != prev_ply + 1


def emit_buffered_game(game_records, board, planes_buf, policy_buf,
                       wdl_buf, mlh_buf, eval_coef):
    """
    Emit one game's records, computing remaining_plies based on the
    LAST recorded ply in the game. Returns (n_emitted, n_skipped_records).
    """
    if not game_records:
        return 0, 0
    final_ply = int(game_records[-1]['gamePly'])

    n_emitted = 0
    n_skipped = 0
    for rec in game_records:
        cur_ply = int(rec['gamePly'])
        remaining = max(0, final_ply - cur_ply)

        try:
            enc = encode_one_psv(board, rec, eval_coef)
        except Exception:
            n_skipped += 1
            continue
        if enc is None:
            n_skipped += 1
            continue

        planes, policy_idx, wdl = enc
        planes_buf.append(planes)
        policy_buf.append(policy_idx)
        wdl_buf.append(wdl)
        mlh_buf.append(remaining)
        n_emitted += 1

    return n_emitted, n_skipped


def process_psv_file(args):
    """Process a single PSV file → one or more shards."""
    psv_path, shard_id_base, output_dir, shard_size, eval_coef = args

    # Memory-map: never loads the full file at once.
    file_size = os.path.getsize(psv_path)
    if file_size % PSV_DTYPE.itemsize != 0:
        return (psv_path, 0, 0, 0, 0, 0,
                f"file size {file_size} not divisible by {PSV_DTYPE.itemsize}")
    n_records = file_size // PSV_DTYPE.itemsize
    records = np.memmap(psv_path, dtype=PSV_DTYPE, mode='r',
                        shape=(n_records,))

    board = cshogi.Board()

    planes_buf, policy_buf, wdl_buf, mlh_buf = [], [], [], []
    shard_id = shard_id_base
    n_games = 0
    n_emitted = 0
    n_skipped = 0
    n_boundaries_negative = 0    # gamePly went down (game restart)
    n_boundaries_positive = 0    # gamePly jumped forward
    err_msg = None

    game_records = []
    prev_ply = None

    try:
        for idx in range(n_records):
            rec = records[idx]
            cur_ply = int(rec['gamePly'])

            if is_game_boundary(prev_ply, cur_ply):
                # Track boundary type for diagnostics
                if cur_ply < prev_ply:
                    n_boundaries_negative += 1
                else:
                    n_boundaries_positive += 1

                # Flush the previous game
                e, s = emit_buffered_game(game_records, board,
                                          planes_buf, policy_buf,
                                          wdl_buf, mlh_buf, eval_coef)
                n_emitted += e
                n_skipped += s
                n_games += 1
                game_records = []

                # Flush full shards
                while len(planes_buf) >= shard_size:
                    flush_shard(shard_id, output_dir,
                                planes_buf[:shard_size],
                                policy_buf[:shard_size],
                                wdl_buf[:shard_size],
                                mlh_buf[:shard_size])
                    planes_buf = planes_buf[shard_size:]
                    policy_buf = policy_buf[shard_size:]
                    wdl_buf    = wdl_buf[shard_size:]
                    mlh_buf    = mlh_buf[shard_size:]
                    shard_id += 1

            game_records.append(rec)
            prev_ply = cur_ply

        # Flush the very last game
        e, s = emit_buffered_game(game_records, board,
                                  planes_buf, policy_buf,
                                  wdl_buf, mlh_buf, eval_coef)
        n_emitted += e
        n_skipped += s
        if game_records:
            n_games += 1

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"

    # Flush whatever's left
    while len(planes_buf) >= shard_size:
        flush_shard(shard_id, output_dir,
                    planes_buf[:shard_size],
                    policy_buf[:shard_size],
                    wdl_buf[:shard_size],
                    mlh_buf[:shard_size])
        planes_buf = planes_buf[shard_size:]
        policy_buf = policy_buf[shard_size:]
        wdl_buf    = wdl_buf[shard_size:]
        mlh_buf    = mlh_buf[shard_size:]
        shard_id += 1
    if planes_buf:
        flush_shard(shard_id, output_dir, planes_buf,
                    policy_buf, wdl_buf, mlh_buf)
        shard_id += 1

    return (psv_path, n_records, n_games, n_emitted, n_skipped,
            n_boundaries_positive + n_boundaries_negative, err_msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--psv-dir", required=True, help="Directory of .bin PSV files")
    p.add_argument("--psv-glob", default="*.bin",
                   help="Glob pattern within psv-dir (default: *.bin)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-size", type=int, default=500_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-coef", type=float, default=600.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import glob
    psv_files = sorted(glob.glob(os.path.join(args.psv_dir, args.psv_glob)))
    if not psv_files:
        print(f"No PSV files found in {args.psv_dir} (glob={args.psv_glob})",
              file=sys.stderr)
        return

    print(f"Found {len(psv_files)} PSV files, workers={args.workers}, "
          f"shard-size={args.shard_size:,}")

    tasks = []
    for idx, path in enumerate(psv_files):
        shard_id_base = idx * PER_FILE_SHARD_BUDGET
        tasks.append((path, shard_id_base, args.output_dir,
                      args.shard_size, args.eval_coef))

    t0 = time.time()
    total_records = 0
    total_games = 0
    total_emitted = 0
    total_skipped = 0
    total_boundaries = 0
    n_failed_files = 0

    with Pool(args.workers) as pool:
        for psv_path, n_recs, n_games, n_emit, n_skip, n_bound, err in \
                pool.imap_unordered(process_psv_file, tasks):
            total_records += n_recs
            total_games += n_games
            total_emitted += n_emit
            total_skipped += n_skip
            total_boundaries += n_bound
            if err:
                n_failed_files += 1
                print(f"  FAILED {os.path.basename(psv_path)}: {err}",
                      file=sys.stderr)
            elapsed = time.time() - t0
            print(f"  done {os.path.basename(psv_path)}: "
                  f"{n_recs:,} records, {n_games:,} games, "
                  f"{n_emit:,} emitted ({n_skip} skipped), "
                  f"total {total_emitted:,} in {elapsed:.0f}s "
                  f"({total_emitted/max(elapsed,1):.0f}/s)")

    print(f"\nFinished:")
    print(f"  records read   : {total_records:,}")
    print(f"  games detected : {total_games:,}")
    print(f"  positions emitted: {total_emitted:,}")
    print(f"  skipped records: {total_skipped:,}")
    print(f"  game boundaries: {total_boundaries:,}")
    print(f"  failed files   : {n_failed_files}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
