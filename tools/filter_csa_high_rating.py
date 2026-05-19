#!/usr/bin/env python3
"""Copy high-quality floodgate CSA games by rating/result.

Selected games are:
  * black rating > threshold and black lost
  * white rating > threshold and white lost
  * black rating > threshold and the game was drawn

Only normal floodgate endings are selected by default: toryo and sennichite.

The output directory preserves the relative source directory layout.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RATE_RE = re.compile(r"^'(black|white)_rate:(.*):([-+]?\d+(?:\.\d+)?)\s*$")


@dataclass
class CsaInfo:
    path: str
    black_name: str | None = None
    white_name: str | None = None
    black_rate: float | None = None
    white_rate: float | None = None
    moves: int = 0
    result: str | None = None  # "black_win", "white_win", or "draw"
    summary_reason: str | None = None
    parse_error: str | None = None


@dataclass
class WorkerArgs:
    path: str
    input_root: str
    output_root: str
    threshold: float
    allowed_endings: frozenset[str]
    min_moves: int
    dry_run: bool
    overwrite: bool


def iter_csa_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".csa"):
                yield Path(dirpath) / name


def parse_summary(line: str) -> tuple[str | None, str | None]:
    """Return (reason, result) from a floodgate summary comment."""
    if not line.startswith("'summary:"):
        return None, None

    body = line[len("'summary:"):].strip()
    parts = body.split(":", 2)
    if len(parts) != 3:
        return None, None

    reason, black_part, white_part = parts
    reason = reason.strip().lower()
    try:
        _, black_result = black_part.rsplit(" ", 1)
        _, white_result = white_part.rsplit(" ", 1)
    except ValueError:
        return reason, None

    black_result = black_result.lower()
    white_result = white_result.lower()
    if black_result == "win" and white_result == "lose":
        return reason, "black_win"
    if black_result == "lose" and white_result == "win":
        return reason, "white_win"
    if black_result == "draw" and white_result == "draw":
        return reason, "draw"
    return reason, None


def parse_csa(path: Path) -> CsaInfo:
    info = CsaInfo(path=str(path))
    last_move_side: str | None = None
    terminal: str | None = None

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if line.startswith("N+"):
                    info.black_name = line[2:].strip()
                elif line.startswith("N-"):
                    info.white_name = line[2:].strip()
                elif line.startswith("+") and len(line) >= 7:
                    last_move_side = "black"
                    info.moves += 1
                elif line.startswith("-") and len(line) >= 7:
                    last_move_side = "white"
                    info.moves += 1
                elif line.startswith("%"):
                    terminal = line.strip()

                m = RATE_RE.match(line)
                if m:
                    side, _, value = m.groups()
                    if side == "black":
                        info.black_rate = float(value)
                    else:
                        info.white_rate = float(value)
                    continue

                reason, result = parse_summary(line)
                if reason is not None:
                    info.summary_reason = reason
                    info.result = result
    except OSError as exc:
        info.parse_error = str(exc)
        return info

    if info.result is None:
        info.summary_reason, info.result = infer_result_from_terminal(
            terminal, last_move_side)
    return info


def infer_result_from_terminal(terminal: str | None,
                               last_move_side: str | None
                               ) -> tuple[str | None, str | None]:
    """Best-effort fallback for CSA files missing a floodgate summary."""
    if terminal == "%SENNICHITE":
        return "sennichite", "draw"
    if terminal == "%TORYO" and last_move_side is not None:
        # These terminal records are made by the side to move. If the previous
        # actual move was black, white is to move and lost; vice versa.
        result = "black_win" if last_move_side == "black" else "white_win"
        return "toryo", result
    return None, None


def match_reason(info: CsaInfo, threshold: float,
                 allowed_endings: frozenset[str],
                 min_moves: int) -> str | None:
    if info.summary_reason not in allowed_endings:
        return None
    if info.moves < min_moves:
        return None

    black_high = info.black_rate is not None and info.black_rate > threshold
    white_high = info.white_rate is not None and info.white_rate > threshold

    if info.result == "white_win" and black_high:
        return "black_high_rating_lost"
    if info.result == "black_win" and white_high:
        return "white_high_rating_lost"
    if info.result == "draw" and black_high:
        return "black_high_rating_draw"
    return None


def selected_destination(src: Path, input_root: Path, output_root: Path) -> Path:
    rel = src.relative_to(input_root)
    return output_root / rel


def process_one(args: WorkerArgs) -> dict[str, object]:
    path = Path(args.path)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    info = parse_csa(path)
    reason = None if info.parse_error else match_reason(
        info, args.threshold, args.allowed_endings, args.min_moves)
    copied = False
    dest = None

    if reason is not None:
        dest_path = selected_destination(path, input_root, output_root)
        dest = str(dest_path)
        if not args.dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite or not dest_path.exists():
                shutil.copy2(path, dest_path)
            copied = True

    return {
        "path": str(path),
        "selected": reason is not None,
        "reason": reason or "",
        "dest": dest or "",
        "copied": copied,
        "black_name": info.black_name or "",
        "white_name": info.white_name or "",
        "black_rate": "" if info.black_rate is None else info.black_rate,
        "white_rate": "" if info.white_rate is None else info.white_rate,
        "moves": info.moves,
        "result": info.result or "",
        "summary_reason": info.summary_reason or "",
        "parse_error": info.parse_error or "",
    }


def write_manifest_row(writer: csv.DictWriter, row: dict[str, object]) -> None:
    writer.writerow({
        "source": row["path"],
        "destination": row["dest"],
        "reason": row["reason"],
        "black_name": row["black_name"],
        "white_name": row["white_name"],
        "black_rate": row["black_rate"],
        "white_rate": row["white_rate"],
        "moves": row["moves"],
        "result": row["result"],
        "summary_reason": row["summary_reason"],
    })


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy CSA games where a >threshold player lost, or "
                    "a >threshold black player drew.")
    parser.add_argument("--input", default="/mnt/shogi_data/floodgate",
                        help="Input directory scanned recursively")
    parser.add_argument("--output", required=True,
                        help="Output directory for selected CSA files")
    parser.add_argument("--threshold", type=float, default=3500.0,
                        help="Rating threshold; selection uses strict > threshold")
    parser.add_argument("--allowed-endings", default="toryo,sennichite",
                        help="Comma-separated floodgate summary endings to keep "
                             "(default: toryo,sennichite)")
    parser.add_argument("--min-moves", type=int, default=20,
                        help="Minimum number of played moves in the CSA file "
                             "(default: 20)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count and write manifest without copying files")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files in the output directory")
    parser.add_argument("--manifest", default=None,
                        help="CSV manifest path. Default: <output>/selected_manifest.csv")
    parser.add_argument("--progress-interval", type=int, default=10000,
                        help="Print progress every N files")
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    if not input_root.is_dir():
        print(f"Input directory does not exist: {input_root}", file=sys.stderr)
        return 2

    allowed_endings = frozenset(
        item.strip().lower()
        for item in args.allowed_endings.split(",")
        if item.strip()
    )
    if not allowed_endings:
        print("--allowed-endings must not be empty", file=sys.stderr)
        return 2
    if args.min_moves < 0:
        print("--min-moves must be non-negative", file=sys.stderr)
        return 2

    manifest_path = (Path(args.manifest).resolve() if args.manifest
                     else output_root / "selected_manifest.csv")
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "source", "destination", "reason", "black_name", "white_name",
        "black_rate", "white_rate", "moves", "result", "summary_reason",
    ]

    total = 0
    selected = 0
    errors = 0

    worker_args = (
        WorkerArgs(str(path), str(input_root), str(output_root),
                   args.threshold, allowed_endings, args.min_moves,
                   args.dry_run, args.overwrite)
        for path in iter_csa_files(input_root)
    )

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fields)
        writer.writeheader()

        if args.workers <= 1:
            results = map(process_one, worker_args)
        else:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            results = executor.map(process_one, worker_args, chunksize=256)

        try:
            for row in results:
                total += 1
                if row["parse_error"]:
                    errors += 1
                if row["selected"]:
                    selected += 1
                    write_manifest_row(writer, row)

                if args.progress_interval > 0 and total % args.progress_interval == 0:
                    print(f"scanned={total:,} selected={selected:,} errors={errors:,}",
                          flush=True)
        finally:
            if args.workers > 1:
                executor.shutdown(wait=True, cancel_futures=True)

    action = "would copy" if args.dry_run else "copied"
    print(f"Done: scanned={total:,} selected={selected:,} {action}={selected:,} "
          f"errors={errors:,}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
