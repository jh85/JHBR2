"""
Pre-compute PSV files into numpy shards using cshogi (battle-tested decoder).

Uses cshogi to read PSV → SFEN + move + score, then our Python encoder
to create planes + policy indices. Guaranteed correct — same pipeline
that created the working floodgate training data.

Usage:
  python precompute_cshogi.py \
    --psv-dir /path/to/psv_files/ \
    --output-dir /workspace/psv_precomputed/ \
    --shard-size 500000 \
    --workers 32 \
    --max-positions 150000000

Then train:
  python shogi_train.py --data /workspace/psv_precomputed/shard ...
"""

import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import cshogi

# Import our encoder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shogi_train import (FEATURE_ENCODING, POLICY_ENCODING,
                         sfen_to_planes, move_to_policy_index)


HCPE_MLH_DTYPE = np.dtype([
    ("hcp", cshogi.dtypeHcp),
    ("eval", cshogi.dtypeEval),
    ("bestMove16", cshogi.dtypeMove16),
    ("gameResult", "u1"),
    ("dummy", "u1"),
    ("mlh", "<i2"),
])


def decode_one_record(board, sfen_data, move_raw, score, game_result,
                      is_hcpe, flip_override=None, eval_coef=600.0):
    """Decode one record (PSV or HCPE) into planes + policy + wdl."""
    # Set position
    if is_hcpe:
        board.set_hcp(np.array(sfen_data, dtype=np.uint8))
    else:
        board.set_psfen(np.array(sfen_data, dtype=np.uint8))

    sfen = board.sfen()
    flip = sfen.split()[1] == 'w'

    # Encode planes
    planes = sfen_to_planes(sfen)

    # Decode move
    if move_raw == 0:
        return None

    if is_hcpe:
        move_usi = cshogi.move_to_usi(move_raw)
    else:
        move_usi = cshogi.move_to_usi(move_raw)

    # Convert to our policy index
    policy_idx = move_to_policy_index(move_usi, flip)
    if policy_idx < 0 or policy_idx >= 2187:
        return None

    # WDL target
    win_rate = 1.0 / (1.0 + math.exp(-score / eval_coef))

    # HCPE game_result: 0=draw, 1=black_win, 2=white_win
    # PSV game_result: 1=side-to-move won, -1=lost, 0=draw
    if is_hcpe:
        # Convert HCPE result to side-to-move perspective
        is_black = sfen.split()[1] == 'b'
        if game_result == 0:
            hard = [0.0, 1.0, 0.0]  # draw
        elif (game_result == 1 and is_black) or (game_result == 2 and not is_black):
            hard = [1.0, 0.0, 0.0]  # side-to-move won
        else:
            hard = [0.0, 0.0, 1.0]  # side-to-move lost
    else:
        if game_result == 1:
            hard = [1.0, 0.0, 0.0]
        elif game_result == 0:
            hard = [0.0, 1.0, 0.0]
        else:
            hard = [0.0, 0.0, 1.0]

    wdl = [0.7 * win_rate + 0.3 * hard[0],
           0.0 + 0.3 * hard[1],
           0.7 * (1.0 - win_rate) + 0.3 * hard[2]]

    return planes, policy_idx, wdl


def process_chunk(args):
    """Process a range of records from one file (PSV or HCPE)."""
    (data_path, start_offset, num_records, shard_id, output_dir, eval_coef,
     is_hcpe, has_mlh) = args

    if is_hcpe:
        dtype = HCPE_MLH_DTYPE if has_mlh else cshogi.HuffmanCodedPosAndEval
    else:
        dtype = cshogi.PackedSfenValue
    records = np.fromfile(data_path, dtype=dtype,
                          offset=start_offset, count=num_records)
    actual = len(records)
    if actual == 0:
        return shard_id, 0

    board = cshogi.Board()

    planes_list = []
    policy_list = []
    wdl_list = []
    mlh_list = []
    errors = 0

    for i in range(actual):
        try:
            r = records[i]

            if is_hcpe:
                sfen_data = r['hcp']
                move_raw = int(r['bestMove16'])
                score = int(r['eval'])
                game_result = int(r['gameResult'])
                mlh_target = int(r['mlh']) if has_mlh else -1
            else:
                sfen_data = r['sfen']
                move_raw = int(r['move'])
                score = int(r['score'])
                game_result = int(r['game_result'])
                mlh_target = -1

            result = decode_one_record(board, sfen_data, move_raw, score,
                                       game_result, is_hcpe, eval_coef=eval_coef)
            if result is None:
                errors += 1
                continue

            planes, policy_idx, wdl = result
            planes_list.append(planes)
            policy_list.append(policy_idx)
            wdl_list.append(wdl)
            mlh_list.append(mlh_target)

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Shard {shard_id}: error at record {i}: {e}",
                      file=sys.stderr)

    if len(planes_list) == 0:
        return shard_id, 0

    # Save shard
    planes_arr = np.array(planes_list, dtype=np.float16)
    policy_arr = np.array(policy_list, dtype=np.int32)
    wdl_arr = np.array(wdl_list, dtype=np.float16)
    mlh_arr = np.array(mlh_list, dtype=np.int16)

    out_path = os.path.join(output_dir, f"shard_{shard_id:06d}.npz")
    np.savez_compressed(out_path, planes=planes_arr, policy=policy_arr,
                        wdl=wdl_arr, mlh=mlh_arr,
                        policy_encoding=np.array(POLICY_ENCODING),
                        feature_encoding=np.array(FEATURE_ENCODING))

    return shard_id, len(planes_list)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute PSV to numpy shards using cshogi")
    parser.add_argument("--psv-dir", default=None, help="Directory with .bin PSV files")
    parser.add_argument("--psv-file", default=None, help="Single PSV file")
    parser.add_argument("--hcpe-dir", default=None, help="Directory with .hcpe files")
    parser.add_argument("--hcpe-file", default=None, help="Single HCPE file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=500000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--eval-coef", type=float, default=600.0)
    parser.add_argument("--hcpe-format", choices=("auto", "hcpe", "hcpe_mlh"),
                        default="auto",
                        help="HCPE record format. Use hcpe_mlh for pack-converted "
                             "files with moves-left targets.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find data files (PSV or HCPE)
    is_hcpe = False
    if args.hcpe_file:
        data_files = [args.hcpe_file]
        is_hcpe = True
    elif args.hcpe_dir:
        data_files = sorted([
            os.path.join(args.hcpe_dir, f)
            for f in os.listdir(args.hcpe_dir) if f.endswith('.hcpe')
        ])
        is_hcpe = True
    elif args.psv_file:
        data_files = [args.psv_file]
    elif args.psv_dir:
        data_files = sorted([
            os.path.join(args.psv_dir, f)
            for f in os.listdir(args.psv_dir) if f.endswith('.bin')
        ])
    else:
        print("Error: specify --psv-dir, --psv-file, --hcpe-dir, or --hcpe-file",
              file=sys.stderr)
        return

    fmt = "HCPE" if is_hcpe else "PSV"
    print(f"Found {len(data_files)} {fmt} file(s)")

    # Build task list
    tasks = []
    shard_id = 0
    total_positions = 0

    for data_path in data_files:
        file_size = os.path.getsize(data_path)
        has_mlh = False
        if is_hcpe:
            can_be_mlh = file_size % HCPE_MLH_DTYPE.itemsize == 0
            can_be_hcpe = file_size % cshogi.HuffmanCodedPosAndEval.itemsize == 0
            if args.hcpe_format == "hcpe_mlh":
                if not can_be_mlh:
                    print(f"Skipping {data_path}: size {file_size} is not valid HCPE_MLH",
                          file=sys.stderr)
                    continue
                record_size = HCPE_MLH_DTYPE.itemsize
                has_mlh = True
            elif args.hcpe_format == "hcpe":
                if not can_be_hcpe:
                    print(f"Skipping {data_path}: size {file_size} is not valid HCPE",
                          file=sys.stderr)
                    continue
                record_size = cshogi.HuffmanCodedPosAndEval.itemsize
            elif can_be_mlh and can_be_hcpe:
                print(f"Skipping {data_path}: size {file_size} is ambiguous between "
                      "HCPE and HCPE_MLH; pass --hcpe-format hcpe or hcpe_mlh",
                      file=sys.stderr)
                continue
            elif can_be_mlh:
                record_size = HCPE_MLH_DTYPE.itemsize
                has_mlh = True
            elif can_be_hcpe:
                record_size = cshogi.HuffmanCodedPosAndEval.itemsize
            else:
                print(f"Skipping {data_path}: size {file_size} is not a valid HCPE/HCPE_MLH file",
                      file=sys.stderr)
                continue
        else:
            record_size = cshogi.PackedSfenValue.itemsize
        file_records = file_size // record_size
        offset = 0
        while offset < file_records:
            chunk = min(args.shard_size, file_records - offset)
            if args.max_positions and total_positions + chunk > args.max_positions:
                chunk = args.max_positions - total_positions
            if chunk <= 0:
                break
            tasks.append((data_path, offset * record_size, chunk, shard_id,
                          args.output_dir, args.eval_coef, is_hcpe, has_mlh))
            offset += chunk
            total_positions += chunk
            shard_id += 1
            if args.max_positions and total_positions >= args.max_positions:
                break
        if args.max_positions and total_positions >= args.max_positions:
            break

    print(f"Total: {total_positions:,} positions → {len(tasks)} shards")
    print(f"Workers: {args.workers}")
    print()

    t0 = time.time()
    done = 0
    total_ok = 0

    with Pool(args.workers) as pool:
        for sid, count in pool.imap_unordered(process_chunk, tasks):
            done += 1
            total_ok += count
            if done % 10 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = total_ok / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / (done / elapsed) if done > 0 else 0
                print(f"  {done}/{len(tasks)} shards, {total_ok:,} positions "
                      f"({rate:.0f} pos/sec, ETA {eta/60:.0f}min)")

    elapsed = time.time() - t0
    print(f"\nDone! {total_ok:,} positions in {elapsed:.0f}s "
          f"({total_ok/elapsed:.0f} pos/sec)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
