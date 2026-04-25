"""
Generate training data from PSV files using YaneuraOu evaluation.

For each position in the PSV dataset:
1. Decode packed sfen → SFEN string
2. Send to YaneuraOu via USI: "position sfen ...\ngo depth 1"
3. Get bestmove + evaluation score
4. Output: position + move + score + game_result

Usage:
  python generate_training_data.py \
    --psv-dir /mnt2/shogi_data/Knowledge_distilled_dataset_by_DLSuisho15b/ \
    --engine /home/ei/Downloads/YaneuraOu/source/YaneuraOu-by-gcc \
    --eval-dir /path/to/eval_dir/ \
    --output train_distilled.bin \
    --num-positions 100000000 \
    --workers 8 \
    --depth 1
"""

import argparse
import os
import struct
import subprocess
import sys
import time
from multiprocessing import Pool, Queue
from pathlib import Path

from psv_loader import read_psv_file, decode_packed_sfen, board_to_sfen


def start_engine(engine_path, eval_dir=None, threads=1):
    """Start a YaneuraOu USI engine subprocess."""
    proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    # Initialize USI
    proc.stdin.write("usi\n")
    proc.stdin.flush()
    # Read until usiok
    while True:
        line = proc.stdout.readline().strip()
        if line == "usiok":
            break

    # Set options
    proc.stdin.write(f"setoption name Threads value {threads}\n")
    if eval_dir:
        proc.stdin.write(f"setoption name EvalDir value {eval_dir}\n")
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline().strip()
        if line == "readyok":
            break

    return proc


def evaluate_position(proc, sfen, depth=1):
    """Send position to engine, get bestmove and score."""
    proc.stdin.write(f"position sfen {sfen}\n")
    proc.stdin.write(f"go depth {depth}\n")
    proc.stdin.flush()

    score = 0
    bestmove = None
    while True:
        line = proc.stdout.readline().strip()
        if line.startswith("info") and "score cp" in line:
            parts = line.split()
            try:
                idx = parts.index("cp")
                score = int(parts[idx + 1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("info") and "score mate" in line:
            parts = line.split()
            try:
                idx = parts.index("mate")
                mate_val = int(parts[idx + 1])
                score = 30000 if mate_val > 0 else -30000
            except (ValueError, IndexError):
                pass
        elif line.startswith("bestmove"):
            bestmove = line.split()[1]
            break

    return bestmove, score


def process_chunk(args):
    """Process a chunk of PSV records with one engine instance."""
    psv_file, start_offset, num_records, engine_path, eval_dir, depth, worker_id = args
    RECORD_SIZE = 40

    proc = start_engine(engine_path, eval_dir)

    results = []
    processed = 0
    errors = 0
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

            try:
                board, hands, turn = decode_packed_sfen(sfen_bytes)
                sfen = board_to_sfen(board, hands, turn, game_ply)

                bestmove, eval_score = evaluate_position(proc, sfen, depth)
                if bestmove and bestmove != "resign" and bestmove != "win":
                    results.append({
                        "sfen": sfen,
                        "move": bestmove,
                        "score": eval_score,
                        "dl_score": dl_score,
                        "game_result": game_result,
                    })
            except Exception as e:
                errors += 1
                if errors < 5:
                    print(f"  Worker {worker_id}: error at record {processed}: {e}")

            processed += 1
            if processed % 10000 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                print(f"  Worker {worker_id}: {processed}/{num_records} "
                      f"({rate:.0f} pos/sec, {len(results)} valid)")

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait()

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate training data from PSV")
    parser.add_argument("--psv-dir", required=True, help="Directory with .bin PSV files")
    parser.add_argument("--psv-file", default=None, help="Single PSV file (overrides --psv-dir)")
    parser.add_argument("--engine", required=True, help="Path to YaneuraOu executable")
    parser.add_argument("--eval-dir", default=None, help="Path to eval directory")
    parser.add_argument("--output", default="train_distilled.sfen", help="Output file")
    parser.add_argument("--num-positions", type=int, default=1000000, help="Positions to process")
    parser.add_argument("--workers", type=int, default=8, help="Parallel engine instances")
    parser.add_argument("--depth", type=int, default=1, help="Search depth for YaneuraOu")
    args = parser.parse_args()

    # Find PSV files
    if args.psv_file:
        psv_files = [args.psv_file]
    else:
        psv_files = sorted([
            os.path.join(args.psv_dir, f)
            for f in os.listdir(args.psv_dir)
            if f.endswith(".bin")
        ])

    if not psv_files:
        print("No PSV files found!")
        return

    print(f"Found {len(psv_files)} PSV files")
    print(f"Processing {args.num_positions:,} positions with {args.workers} workers, depth={args.depth}")

    # Use the first file, split into chunks for workers
    psv_file = psv_files[0]
    file_size = os.path.getsize(psv_file)
    total_records = file_size // 40
    num_to_process = min(args.num_positions, total_records)
    per_worker = num_to_process // args.workers

    chunks = []
    for w in range(args.workers):
        start = w * per_worker
        count = per_worker if w < args.workers - 1 else (num_to_process - start)
        chunks.append((psv_file, start, count, args.engine, args.eval_dir,
                        args.depth, w))

    t0 = time.time()

    # Process in parallel
    all_results = []
    with Pool(args.workers) as pool:
        for worker_results in pool.imap_unordered(process_chunk, chunks):
            all_results.extend(worker_results)
            print(f"  Collected {len(all_results):,} results so far...")

    elapsed = time.time() - t0
    print(f"\nDone! {len(all_results):,} positions in {elapsed:.1f}s "
          f"({len(all_results)/elapsed:.0f} pos/sec)")

    # Write output as SFEN with moves (our training format)
    with open(args.output, "w") as f:
        for r in all_results:
            # Format: sfen<tab>move<tab>score<tab>result
            f.write(f"{r['sfen']}\t{r['move']}\t{r['score']}\t{r['game_result']}\n")

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
