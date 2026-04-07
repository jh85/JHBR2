"""
Pre-compute training data from SFEN text to binary numpy format.

Converts SFEN strings → (planes, policy_idx, wdl) numpy arrays.
This eliminates the expensive per-sample SFEN parsing during training.

Usage:
    python precompute.py \
      --input floodgate_R3000_all.sfen \
      --output /mnt2/shogi_data/training/floodgate_R3000_all \
      --shard-size 5000000

    This creates sharded files:
      floodgate_R3000_all_000.npz (5M positions)
      floodgate_R3000_all_001.npz (5M positions)
      ...

Training then uses:
    python shogi_train.py --data /mnt2/shogi_data/training/floodgate_R3000_all --epochs 10
"""

import argparse
import os
import sys
import time
import math

import numpy as np

from shogi_train import sfen_to_planes, move_to_policy_index, build_move_index


def precompute(input_path, output_prefix, shard_size=500_000):
    """Convert SFEN text file to sharded .npz files with pre-computed tensors."""

    move_info_to_idx = build_move_index()

    # Count lines first
    print(f"Counting lines in {input_path}...")
    with open(input_path) as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines: {total_lines:,}")

    num_shards = math.ceil(total_lines / shard_size)
    print(f"Shard size: {shard_size:,}, Shards: {num_shards}")
    print(f"Output: {output_prefix}_000.npz ... {output_prefix}_{num_shards-1:03d}.npz")

    # Process
    shard_idx = 0
    buf_planes = []
    buf_policy = []
    buf_wdl = []
    total_written = 0
    errors = 0
    t0 = time.time()

    with open(input_path) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                sample = _parse_line(line, move_info_to_idx)
                if sample is None:
                    errors += 1
                    continue

                planes, policy_idx, wdl = sample
                buf_planes.append(planes)
                buf_policy.append(policy_idx)
                buf_wdl.append(wdl)

            except Exception:
                errors += 1
                continue

            # Flush shard when buffer is full
            if len(buf_planes) >= shard_size:
                _save_shard(output_prefix, shard_idx, buf_planes, buf_policy, buf_wdl)
                total_written += len(buf_planes)
                elapsed = time.time() - t0
                speed = total_written / elapsed
                print(f"  Shard {shard_idx:03d}: {len(buf_planes):,} positions "
                      f"(total {total_written/1e6:.1f}M, {speed:.0f} pos/sec)")
                buf_planes.clear()
                buf_policy.clear()
                buf_wdl.clear()
                shard_idx += 1

            # Progress
            if (line_num + 1) % 1_000_000 == 0:
                elapsed = time.time() - t0
                speed = (line_num + 1) / elapsed
                pct = 100.0 * (line_num + 1) / total_lines
                print(f"  Processing: {(line_num+1)/1e6:.0f}M / {total_lines/1e6:.0f}M "
                      f"({pct:.1f}%) {speed:.0f} lines/sec")

    # Save remaining
    if buf_planes:
        _save_shard(output_prefix, shard_idx, buf_planes, buf_policy, buf_wdl)
        total_written += len(buf_planes)
        print(f"  Shard {shard_idx:03d}: {len(buf_planes):,} positions")

    elapsed = time.time() - t0
    print(f"\nDone: {total_written:,} positions in {shard_idx+1} shards "
          f"({elapsed:.1f}s, {total_written/elapsed:.0f} pos/sec)")
    print(f"Errors: {errors:,}")

    # Write metadata
    meta_path = output_prefix + "_meta.txt"
    with open(meta_path, 'w') as f:
        f.write(f"total_positions={total_written}\n")
        f.write(f"num_shards={shard_idx+1}\n")
        f.write(f"shard_size={shard_size}\n")
        f.write(f"planes_shape=48,9,9\n")
        f.write(f"source={input_path}\n")
    print(f"Metadata: {meta_path}")


def _parse_line(line, move_info_to_idx):
    """Parse one training line and return (planes, policy_idx, wdl)."""
    parts = line.split()
    if parts[0] != 'sfen':
        return None

    sfen_parts = []
    i = 1
    while i < len(parts) and parts[i] not in ('bestmove', 'result', 'score'):
        sfen_parts.append(parts[i])
        i += 1
    sfen = ' '.join(sfen_parts)

    bestmove = None
    result = None
    score = None
    while i < len(parts):
        if parts[i] == 'bestmove' and i + 1 < len(parts):
            bestmove = parts[i + 1]; i += 2
        elif parts[i] == 'score' and i + 1 < len(parts):
            score = int(parts[i + 1]); i += 2
        elif parts[i] == 'result' and i + 1 < len(parts):
            result = parts[i + 1]; i += 2
        else:
            i += 1

    if result is None:
        return None

    # Compute planes
    planes = sfen_to_planes(sfen)

    # Policy target
    if bestmove:
        flip = sfen.split()[1] == 'w'
        info = move_to_policy_index(bestmove, flip)
        policy_idx = move_info_to_idx(info)
    else:
        policy_idx = -1

    # WDL target
    if score is not None:
        win_rate = 1.0 / (1.0 + math.exp(-score / 600.0))
        if result == 'W':
            hard = [1.0, 0.0, 0.0]
        elif result == 'D':
            hard = [0.0, 1.0, 0.0]
        else:
            hard = [0.0, 0.0, 1.0]
        soft = [win_rate, 0.0, 1.0 - win_rate]
        wdl = [0.7 * s + 0.3 * h for s, h in zip(soft, hard)]
    elif result == 'W':
        wdl = [1.0, 0.0, 0.0]
    elif result == 'D':
        wdl = [0.0, 1.0, 0.0]
    else:
        wdl = [0.0, 0.0, 1.0]

    return planes, policy_idx, wdl


def _save_shard(prefix, idx, planes_list, policy_list, wdl_list):
    """Save one shard as compressed .npz."""
    path = f"{prefix}_{idx:03d}.npz"
    n = len(planes_list)

    # Pre-allocate arrays and fill (avoids np.array() on huge list of arrays)
    planes = np.empty((n, 48, 9, 9), dtype=np.float16)
    for i in range(n):
        planes[i] = planes_list[i]

    np.savez_compressed(
        path,
        planes=planes,
        policy=np.array(policy_list, dtype=np.int32),
        wdl=np.array(wdl_list, dtype=np.float32),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute training data")
    parser.add_argument("--input", required=True, help="Input SFEN text file")
    parser.add_argument("--output", required=True, help="Output prefix (without .npz)")
    parser.add_argument("--shard-size", type=int, default=500_000,
                        help="Positions per shard file (default 500K)")
    args = parser.parse_args()
    precompute(args.input, args.output, args.shard_size)
