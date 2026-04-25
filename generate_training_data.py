"""
Generate training data from PSV files using YaneuraOu search (USI).

For each position in the PSV dataset:
1. Decode packed sfen → SFEN string
2. Send to YaneuraOu via USI: "position sfen ... \n go depth N"
3. Get bestmove + evaluation score (full search with TT, pruning, etc.)
4. Output: sfen + move + score + game_result (TSV)

Each worker keeps a persistent YaneuraOu process (initialized once).
USI overhead per position is ~2μs — negligible vs search time (~2-50ms).

Usage:
  python generate_training_data.py \
    --psv-dir /path/to/psv/ \
    --engine /path/to/YaneuraOu-by-gcc \
    --eval-dir /path/to/eval/ \
    --output train_distilled.tsv \
    --num-positions 500000000 \
    --workers 64 \
    --depth 3

Speed estimates (per worker):
  depth 3: ~500 pos/sec  → 64 workers → 32K/sec → 500M in ~4 hours
  depth 5: ~50 pos/sec   → 64 workers → 3.2K/sec → 100M in ~9 hours
"""

import argparse
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from psv_loader import decode_packed_sfen, board_to_sfen


# =====================================================================
# YaneuraOu USI engine wrapper
# =====================================================================

def start_engine(engine_path, eval_dir=None, hash_mb=64):
    """Start a persistent YaneuraOu USI engine subprocess."""
    proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    # USI handshake
    proc.stdin.write("usi\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline().strip()
        if line == "usiok":
            break

    # Options: single thread per engine, small hash
    proc.stdin.write("setoption name Threads value 1\n")
    proc.stdin.write(f"setoption name USI_Hash value {hash_mb}\n")
    if eval_dir:
        proc.stdin.write(f"setoption name EvalDir value {eval_dir}\n")
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline().strip()
        if line == "readyok":
            break

    return proc


def evaluate_position(proc, sfen, depth):
    """Send position to engine, run search, return (bestmove, score)."""
    proc.stdin.write(f"position sfen {sfen}\n")
    proc.stdin.write(f"go depth {depth}\n")
    proc.stdin.flush()

    score = 0
    bestmove = None
    while True:
        line = proc.stdout.readline().strip()
        if not line:
            continue
        if line.startswith("info") and " score " in line:
            parts = line.split()
            try:
                if "cp" in parts:
                    idx = parts.index("cp")
                    score = int(parts[idx + 1])
                elif "mate" in parts:
                    idx = parts.index("mate")
                    mate_val = int(parts[idx + 1])
                    score = 30000 if mate_val > 0 else -30000
            except (ValueError, IndexError):
                pass
        elif line.startswith("bestmove"):
            parts = line.split()
            bestmove = parts[1] if len(parts) > 1 else None
            break

    return bestmove, score


def stop_engine(proc):
    """Gracefully stop the engine."""
    try:
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


# =====================================================================
# Worker: process a chunk of PSV records
# =====================================================================

def process_chunk(args):
    """Process a range of PSV records with one YaneuraOu instance."""
    (psv_file, start_offset, num_records, engine_path,
     eval_dir, depth, hash_mb, worker_id) = args

    RECORD_SIZE = 40

    # Start persistent engine (initialized once per worker)
    proc = start_engine(engine_path, eval_dir, hash_mb)

    results = []
    processed = 0
    errors = 0
    skipped = 0
    t0 = time.time()

    with open(psv_file, "rb") as f:
        f.seek(start_offset * RECORD_SIZE)
        for _ in range(num_records):
            rec = f.read(RECORD_SIZE)
            if len(rec) < RECORD_SIZE:
                break

            sfen_bytes = rec[:32]
            dl_score = struct.unpack("<h", rec[32:34])[0]
            game_ply = struct.unpack("<H", rec[36:38])[0]
            game_result = struct.unpack("<b", rec[38:39])[0]

            # Skip extreme positions (mate scores from DL Suisho)
            if abs(dl_score) >= 30000:
                skipped += 1
                processed += 1
                continue

            try:
                board, hands, turn = decode_packed_sfen(sfen_bytes)
                sfen = board_to_sfen(board, hands, turn, game_ply)

                bestmove, eval_score = evaluate_position(proc, sfen, depth)

                if bestmove and bestmove not in ("resign", "win", "none"):
                    results.append(
                        f"{sfen}\t{bestmove}\t{eval_score}\t{dl_score}\t{game_result}"
                    )
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  Worker {worker_id}: error at record {processed}: {e}",
                          file=sys.stderr)

            processed += 1
            if processed % 50000 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  Worker {worker_id}: {processed:,}/{num_records:,} "
                      f"({rate:.0f} pos/sec, {len(results):,} valid, "
                      f"{skipped:,} skipped, {errors} errors)",
                      file=sys.stderr)

    stop_engine(proc)

    elapsed = time.time() - t0
    rate = processed / elapsed if elapsed > 0 else 0
    print(f"  Worker {worker_id} done: {len(results):,} results in {elapsed:.0f}s "
          f"({rate:.0f} pos/sec)", file=sys.stderr)

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from PSV using YaneuraOu search")
    parser.add_argument("--psv-dir", default=None,
                        help="Directory with .bin PSV files")
    parser.add_argument("--psv-file", default=None,
                        help="Single PSV file (overrides --psv-dir)")
    parser.add_argument("--engine", required=True,
                        help="Path to YaneuraOu executable")
    parser.add_argument("--eval-dir", default=None,
                        help="Path to eval directory (contains nn.bin)")
    parser.add_argument("--output", default="train_distilled.tsv",
                        help="Output TSV file")
    parser.add_argument("--num-positions", type=int, default=1000000,
                        help="Total positions to process")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel YaneuraOu instances")
    parser.add_argument("--depth", type=int, default=3,
                        help="Search depth (3=fast+decent, 5=slower+better)")
    parser.add_argument("--hash", type=int, default=64,
                        help="Hash table size per engine (MB)")
    parser.add_argument("--shard", type=int, default=None,
                        help="Process only this shard index (0-based)")
    args = parser.parse_args()

    # Find PSV files
    if args.psv_file:
        psv_files = [args.psv_file]
    elif args.psv_dir:
        psv_files = sorted([
            os.path.join(args.psv_dir, f)
            for f in os.listdir(args.psv_dir)
            if f.endswith(".bin")
        ])
    else:
        print("Error: specify --psv-dir or --psv-file", file=sys.stderr)
        return

    if not psv_files:
        print("No PSV files found!", file=sys.stderr)
        return

    # Select shard if specified
    if args.shard is not None:
        if args.shard >= len(psv_files):
            print(f"Shard {args.shard} out of range (0-{len(psv_files)-1})",
                  file=sys.stderr)
            return
        psv_files = [psv_files[args.shard]]
        print(f"Processing shard {args.shard}: {psv_files[0]}", file=sys.stderr)

    print(f"Found {len(psv_files)} PSV file(s)", file=sys.stderr)
    print(f"Engine:     {args.engine}", file=sys.stderr)
    print(f"Eval dir:   {args.eval_dir}", file=sys.stderr)
    print(f"Depth:      {args.depth}", file=sys.stderr)
    print(f"Workers:    {args.workers}", file=sys.stderr)
    print(f"Hash/eng:   {args.hash} MB", file=sys.stderr)
    print(f"Positions:  {args.num_positions:,}", file=sys.stderr)
    print(f"Output:     {args.output}", file=sys.stderr)
    print(file=sys.stderr)

    # Calculate total records across selected files
    total_available = 0
    file_records = []
    for f in psv_files:
        fsize = os.path.getsize(f)
        n = fsize // 40
        file_records.append((f, n))
        total_available += n

    to_process = min(args.num_positions, total_available)
    print(f"Available: {total_available:,} records, processing {to_process:,}",
          file=sys.stderr)

    # Distribute work: split across files and workers
    # For simplicity, use the first file that has enough records
    # TODO: span multiple files if needed
    psv_file, file_total = file_records[0]
    to_process = min(to_process, file_total)
    per_worker = to_process // args.workers

    chunks = []
    for w in range(args.workers):
        start = w * per_worker
        count = per_worker if w < args.workers - 1 else (to_process - start)
        chunks.append((
            psv_file, start, count, args.engine,
            args.eval_dir, args.depth, args.hash, w
        ))

    # Process in parallel
    t0 = time.time()
    all_results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
                print(f"  Collected {len(all_results):,} results total",
                      file=sys.stderr)
            except Exception as e:
                print(f"  Worker failed: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    rate = len(all_results) / elapsed if elapsed > 0 else 0

    print(f"\nTotal: {len(all_results):,} positions in {elapsed:.1f}s "
          f"({rate:.0f} pos/sec)", file=sys.stderr)

    # Write output
    print(f"Writing to {args.output}...", file=sys.stderr)
    with open(args.output, "w") as f:
        # Header
        f.write("sfen\tbestmove\tnnue_score\tdl_score\tgame_result\n")
        for line in all_results:
            f.write(line + "\n")

    print(f"Done! {len(all_results):,} records written to {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
