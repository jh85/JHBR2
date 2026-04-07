"""
Pre-compute training data from SFEN text to binary numpy format.

Converts SFEN strings → (planes, policy_idx, wdl) numpy arrays.
This eliminates the expensive per-sample SFEN parsing during training.

Supports multiprocessing for fast conversion on multi-core CPUs.

Usage:
    python precompute.py \
      --input floodgate_R3000_all.sfen \
      --output /mnt/shogi_data/training/floodgate_R3000_all \
      --workers 32

    This creates sharded files:
      floodgate_R3000_all_000.npz (500K positions)
      floodgate_R3000_all_001.npz (500K positions)
      ...

Training then uses:
    python shogi_train.py --data /mnt/shogi_data/training/floodgate_R3000_all --epochs 10
"""

import argparse
import os
import sys
import time
import math
from multiprocessing import Pool, cpu_count

import numpy as np

from shogi_train import sfen_to_planes, move_to_policy_index, build_move_index


# Global move index (initialized per worker)
_move_info_to_idx = None


def _init_worker():
    """Initialize the move index lookup in each worker process."""
    global _move_info_to_idx
    _move_info_to_idx = build_move_index()


def _process_line(line):
    """Parse one training line and return (planes, policy_idx, wdl) or None."""
    global _move_info_to_idx
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    try:
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
            policy_idx = _move_info_to_idx(info)
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

        return (planes, policy_idx, wdl)

    except Exception:
        return None


def _save_shard(prefix, idx, planes_list, policy_list, wdl_list):
    """Save one shard as compressed .npz."""
    path = f"{prefix}_{idx:03d}.npz"
    n = len(planes_list)

    # Pre-allocate array and fill
    planes = np.empty((n, 48, 9, 9), dtype=np.float16)
    for i in range(n):
        planes[i] = planes_list[i]

    np.savez_compressed(
        path,
        planes=planes,
        policy=np.array(policy_list, dtype=np.int32),
        wdl=np.array(wdl_list, dtype=np.float32),
    )


def precompute(input_path, output_prefix, shard_size=500_000, num_workers=None):
    """Convert SFEN text file to sharded .npz files using multiple CPU cores."""

    if num_workers is None:
        num_workers = min(cpu_count(), 32)

    # Count lines
    print(f"Counting lines in {input_path}...")
    with open(input_path) as f:
        total_lines = sum(1 for _ in f)
    num_shards = math.ceil(total_lines / shard_size)
    print(f"Total lines: {total_lines:,}")
    print(f"Shard size: {shard_size:,}, Expected shards: ~{num_shards}")
    print(f"Workers: {num_workers}")

    shard_idx = 0
    buf_planes = []
    buf_policy = []
    buf_wdl = []
    total_written = 0
    errors = 0
    t0 = time.time()

    # Process in chunks using multiprocessing
    CHUNK_SIZE = 10_000  # Lines per chunk sent to workers

    pool = Pool(num_workers, initializer=_init_worker)

    try:
        with open(input_path) as f:
            chunk = []
            line_num = 0

            for line in f:
                chunk.append(line)
                line_num += 1

                if len(chunk) >= CHUNK_SIZE:
                    # Process chunk in parallel
                    results = pool.map(_process_line, chunk, chunksize=500)

                    for sample in results:
                        if sample is None:
                            errors += 1
                            continue
                        planes, policy_idx, wdl = sample
                        buf_planes.append(planes)
                        buf_policy.append(policy_idx)
                        buf_wdl.append(wdl)

                        # Flush shard
                        if len(buf_planes) >= shard_size:
                            _save_shard(output_prefix, shard_idx,
                                        buf_planes, buf_policy, buf_wdl)
                            total_written += len(buf_planes)
                            elapsed = time.time() - t0
                            speed = total_written / elapsed
                            print(f"  Shard {shard_idx:03d}: {len(buf_planes):,} pos "
                                  f"(total {total_written/1e6:.1f}M, "
                                  f"{speed:.0f} pos/sec)")
                            buf_planes.clear()
                            buf_policy.clear()
                            buf_wdl.clear()
                            shard_idx += 1

                    chunk.clear()

                    # Progress
                    if line_num % 1_000_000 < CHUNK_SIZE:
                        elapsed = time.time() - t0
                        speed = total_written / max(elapsed, 1)
                        pct = 100.0 * line_num / total_lines
                        print(f"  Progress: {line_num/1e6:.0f}M / {total_lines/1e6:.0f}M "
                              f"({pct:.1f}%) {speed:.0f} pos/sec")

            # Process remaining lines
            if chunk:
                results = pool.map(_process_line, chunk, chunksize=500)
                for sample in results:
                    if sample is None:
                        errors += 1
                        continue
                    planes, policy_idx, wdl = sample
                    buf_planes.append(planes)
                    buf_policy.append(policy_idx)
                    buf_wdl.append(wdl)

    finally:
        pool.close()
        pool.join()

    # Save remaining
    if buf_planes:
        _save_shard(output_prefix, shard_idx, buf_planes, buf_policy, buf_wdl)
        total_written += len(buf_planes)
        print(f"  Shard {shard_idx:03d}: {len(buf_planes):,} positions")
        shard_idx += 1

    elapsed = time.time() - t0
    print(f"\nDone: {total_written:,} positions in {shard_idx} shards "
          f"({elapsed:.1f}s, {total_written/elapsed:.0f} pos/sec)")
    print(f"Errors/skipped: {errors:,}")

    # Write metadata
    meta_path = output_prefix + "_meta.txt"
    with open(meta_path, 'w') as f:
        f.write(f"total_positions={total_written}\n")
        f.write(f"num_shards={shard_idx}\n")
        f.write(f"shard_size={shard_size}\n")
        f.write(f"planes_shape=48,9,9\n")
        f.write(f"source={input_path}\n")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute training data")
    parser.add_argument("--input", required=True, help="Input SFEN text file")
    parser.add_argument("--output", required=True, help="Output prefix (without .npz)")
    parser.add_argument("--shard-size", type=int, default=500_000,
                        help="Positions per shard file (default 500K)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of CPU workers (default: min(cpu_count, 32))")
    args = parser.parse_args()
    precompute(args.input, args.output, args.shard_size, args.workers)
