"""
Pre-compute PSV files into numpy shards for fast training.

Converts PSV binary → numpy arrays (planes, policy, wdl).
Training then reads pre-computed arrays directly — no decoding overhead.

Usage:
  # Build C decoder first:
  gcc -O3 -shared -fPIC -o psv_decode_c.so psv_decode_c.c -lm

  # Convert (uses all CPU cores):
  python precompute_psv.py \
    --psv-dir /workspace/shogi_hao_depth9/ \
    --output-dir /workspace/psv_precomputed/ \
    --shard-size 500000 \
    --workers 64

  # Then train with existing ShardedDataset:
  python shogi_train.py \
    --data /workspace/psv_precomputed/shard \
    --epochs 1 --batch 3200 ...
"""

import argparse
import ctypes
import os
import sys
import time
from multiprocessing import Pool

import numpy as np


def get_c_decoder():
    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "psv_decode_c.so")
    if not os.path.exists(so_path):
        c_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "psv_decode_c.c")
        import subprocess
        subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-o", so_path, c_path, "-lm"], check=True)
    lib = ctypes.CDLL(so_path)
    lib.decode_psv_batch.restype = ctypes.c_int
    lib.decode_psv_batch.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
    ]
    return lib


def process_chunk(args):
    """Decode a chunk of PSV records and save as .npz shard."""
    psv_path, start_record, num_records, shard_id, output_dir = args

    lib = get_c_decoder()
    RECORD_SIZE = 40

    # Read raw records
    with open(psv_path, 'rb') as f:
        f.seek(start_record * RECORD_SIZE)
        raw = f.read(num_records * RECORD_SIZE)

    actual_records = len(raw) // RECORD_SIZE
    if actual_records == 0:
        return shard_id, 0

    records = np.frombuffer(raw, dtype=np.uint8).reshape(actual_records, RECORD_SIZE).copy()

    # Allocate output arrays
    planes = np.zeros((actual_records, 48 * 81), dtype=np.float32)
    policy = np.zeros(actual_records, dtype=np.int32)
    wdl = np.zeros((actual_records, 3), dtype=np.float32)

    # Batch decode
    ok = lib.decode_psv_batch(
        records.ctypes.data_as(ctypes.c_void_p),
        actual_records,
        planes.ctypes.data_as(ctypes.c_void_p),
        policy.ctypes.data_as(ctypes.c_void_p),
        wdl.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_float(600.0))

    # Reshape planes and convert to float16 to save disk space
    planes = planes.reshape(actual_records, 48, 9, 9).astype(np.float16)

    # Save shard
    out_path = os.path.join(output_dir, f"shard_{shard_id:06d}.npz")
    np.savez_compressed(out_path,
                        planes=planes,
                        policy=policy,
                        wdl=wdl.astype(np.float16))

    return shard_id, actual_records


def main():
    parser = argparse.ArgumentParser(description="Pre-compute PSV to numpy shards")
    parser.add_argument("--psv-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=500000,
                        help="Positions per output shard")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-positions", type=int, default=None,
                        help="Max total positions (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find PSV files
    psv_files = sorted([
        os.path.join(args.psv_dir, f)
        for f in os.listdir(args.psv_dir) if f.endswith('.bin')
    ])
    print(f"Found {len(psv_files)} PSV files")

    # Build task list
    tasks = []
    shard_id = 0
    total_positions = 0
    for psv_path in psv_files:
        file_size = os.path.getsize(psv_path)
        file_records = file_size // 40
        offset = 0
        while offset < file_records:
            chunk = min(args.shard_size, file_records - offset)
            if args.max_positions and total_positions + chunk > args.max_positions:
                chunk = args.max_positions - total_positions
            if chunk <= 0:
                break
            tasks.append((psv_path, offset, chunk, shard_id, args.output_dir))
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
    with Pool(args.workers) as pool:
        for sid, count in pool.imap_unordered(process_chunk, tasks):
            done += 1
            if done % 10 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(tasks)} shards ({rate:.1f} shards/sec, ETA {eta/60:.0f}min)")

    elapsed = time.time() - t0
    print(f"\nDone! {len(tasks)} shards in {elapsed:.0f}s")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
