"""
Convert YaneuraOu PSV (PackedSfenValue) files to training format.

Uses cshogi to decode packed SFEN positions.

PSV layout (40 bytes per record):
  PackedSfen:  32 bytes
  score:        2 bytes (s16, evaluation in centipawn-like units)
  move16:       2 bytes (may be 0 if not available)
  gamePly:      2 bytes
  game_result:  1 byte (+1=win, -1=loss, 0=draw from side-to-move)
  padding:      1 byte

Usage:
    # Value-only data (DLSuisho15b — no moves):
    python convert_psv.py --input data.bin --output train_value.sfen --limit 50000000

    # Full data (with moves):
    python convert_psv.py --input data.bin --output train.sfen --limit 50000000 --require-move
"""

import argparse
import struct
import os
import sys
import time
import random

import numpy as np
import cshogi


PSV_SIZE = 40
PIECE_CHARS = ".PLNSBRGK"


def decode_move16(move16_raw, board):
    """Decode YaneuraOu's Move16 from PSV. Returns USI string or None."""
    if move16_raw == 0:
        return None

    to_sq = move16_raw & 0x7F
    from_val = (move16_raw >> 7) & 0x7F
    is_drop = bool(move16_raw & (1 << 14))
    is_promo = bool(move16_raw & (1 << 15))

    if is_drop:
        pt = from_val
        if pt < 1 or pt > 7:
            return None
        to_file = to_sq // 9
        to_rank = to_sq % 9
        return f"{PIECE_CHARS[pt]}*{to_file + 1}{chr(ord('a') + to_rank)}"
    else:
        if from_val >= 81 or to_sq >= 81:
            return None
        from_file = from_val // 9
        from_rank = from_val % 9
        to_file = to_sq // 9
        to_rank = to_sq % 9
        usi = f"{from_file + 1}{chr(ord('a') + from_rank)}{to_file + 1}{chr(ord('a') + to_rank)}"
        if is_promo:
            usi += "+"
        return usi


def convert_psv(input_path, output_path, limit=None, sample_rate=1.0,
                eval_coef=600, require_move=False, skip_extreme=True):
    """
    Convert PSV binary to text training format.

    Args:
        input_path:   PSV binary file
        output_path:  output text file
        limit:        max positions to write
        sample_rate:  random sampling rate (0-1)
        eval_coef:    coefficient for score→WDL conversion
        require_move: skip records with move16==0
        skip_extreme: skip positions with |score| > 10000 (likely won/lost)
    """
    file_size = os.path.getsize(input_path)
    total_records = file_size // PSV_SIZE
    target = min(limit or total_records, total_records)

    print(f"Input:  {input_path}")
    print(f"Size:   {file_size / 1e9:.2f} GB ({total_records:,} records)")
    print(f"Target: {target:,} positions (sample_rate={sample_rate})")
    if require_move:
        print("Mode:   require best move (skip records without moves)")
    else:
        print("Mode:   value-only (positions + score + result)")

    board = cshogi.Board()
    written = 0
    skipped_no_move = 0
    skipped_extreme = 0
    errors = 0
    t0 = time.time()

    with open(input_path, "rb") as fin, open(output_path, "w") as fout:
        while written < target:
            data = fin.read(PSV_SIZE)
            if len(data) < PSV_SIZE:
                break

            # Random sampling
            if sample_rate < 1.0 and random.random() > sample_rate:
                continue

            # Parse fields
            packed_sfen = np.frombuffer(data[:32], dtype=np.uint8).copy()
            score = struct.unpack("<h", data[32:34])[0]
            move16_raw = struct.unpack("<H", data[34:36])[0]
            game_ply = struct.unpack("<H", data[36:38])[0]
            game_result = struct.unpack("<b", data[38:39])[0]

            # Skip extreme evaluations (clearly won/lost, not useful for training)
            if skip_extreme and abs(score) > 10000:
                skipped_extreme += 1
                continue

            # Decode position
            try:
                board.set_psfen(packed_sfen)
                sfen = board.sfen()
            except Exception:
                errors += 1
                continue

            # Decode move
            move_usi = decode_move16(move16_raw, board)

            if require_move and move_usi is None:
                skipped_no_move += 1
                continue

            # Game result
            result = "W" if game_result > 0 else ("L" if game_result < 0 else "D")

            # Write output
            line = f"sfen {sfen}"
            if move_usi:
                line += f" bestmove {move_usi}"
            line += f" score {score} result {result}\n"
            fout.write(line)
            written += 1

            if written % 1_000_000 == 0:
                elapsed = time.time() - t0
                rate = written / elapsed
                pct = 100.0 * written / target
                print(f"  {written/1e6:.0f}M written ({pct:.1f}%) "
                      f"{rate:.0f} pos/sec  "
                      f"errors={errors} skipped_extreme={skipped_extreme}")

    elapsed = time.time() - t0
    print(f"\nDone: {written:,} positions written in {elapsed:.1f}s")
    print(f"  Errors: {errors:,}")
    print(f"  Skipped (no move): {skipped_no_move:,}")
    print(f"  Skipped (extreme score): {skipped_extreme:,}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PSV to training format")
    parser.add_argument("--input", required=True, help="Input PSV binary file")
    parser.add_argument("--output", required=True, help="Output text file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--eval-coef", type=float, default=600)
    parser.add_argument("--require-move", action="store_true",
                        help="Skip records without best move")
    parser.add_argument("--no-skip-extreme", action="store_true",
                        help="Don't skip extreme evaluations")
    args = parser.parse_args()

    convert_psv(
        args.input, args.output,
        limit=args.limit,
        sample_rate=args.sample_rate,
        eval_coef=args.eval_coef,
        require_move=args.require_move,
        skip_extreme=not args.no_skip_extreme,
    )
