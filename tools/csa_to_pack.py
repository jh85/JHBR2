#!/usr/bin/env python3
"""Convert floodgate CSA game records to YaneuraOu .pack data."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import cshogi


REPO_ROOT = Path(__file__).resolve().parents[1]
YANE_COMMON = REPO_ROOT.parent / "YaneuraOu-ScriptCollection" / "CommonLib"
sys.path.insert(0, str(YANE_COMMON))
from YaneShogiLib import GameDataEncoder  # noqa: E402


MOVE_RE = re.compile(r"^[+-]\d{4}[A-Z]{2}")
SUMMARY_PREFIX = "'summary:"
EVAL_PREFIX = "'**"


@dataclass
class MoveRecord:
    csa: str
    eval_cp: int | None = None


@dataclass
class ParsedGame:
    path: str
    moves: list[MoveRecord]
    result_abs: int | None
    ending: str | None


@dataclass
class ConvertResult:
    path: str
    ok: bool
    game_bytes: bytes = b""
    moves: int = 0
    ending: str = ""
    result_abs: int = -1
    missing_evals: int = 0
    error: str = ""


def iter_csa_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".csa"):
                yield Path(dirpath) / name


def parse_summary(line: str) -> tuple[str | None, int | None]:
    if not line.startswith(SUMMARY_PREFIX):
        return None, None

    body = line[len(SUMMARY_PREFIX):].strip()
    parts = body.split(":", 2)
    if len(parts) != 3:
        return None, None

    ending = parts[0].strip().lower()
    try:
        _, black_result = parts[1].rsplit(" ", 1)
        _, white_result = parts[2].rsplit(" ", 1)
    except ValueError:
        return ending, None

    black_result = black_result.lower()
    white_result = white_result.lower()
    if black_result == "win" and white_result == "lose":
        return ending, 1
    if black_result == "lose" and white_result == "win":
        return ending, 2
    if black_result == "draw" and white_result == "draw":
        return ending, 0
    return ending, None


def parse_eval(line: str) -> int | None:
    if not line.startswith(EVAL_PREFIX):
        return None
    parts = line[len(EVAL_PREFIX):].strip().split()
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def infer_result_from_terminal(terminal: str | None,
                               last_move_side: str | None
                               ) -> tuple[str | None, int | None]:
    if terminal == "%SENNICHITE":
        return "sennichite", 0
    if terminal == "%TORYO" and last_move_side is not None:
        return "toryo", 1 if last_move_side == "black" else 2
    return None, None


def parse_csa(path: Path) -> ParsedGame:
    moves: list[MoveRecord] = []
    result_abs: int | None = None
    ending: str | None = None
    terminal: str | None = None
    last_move_side: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if MOVE_RE.match(line):
                moves.append(MoveRecord(csa=line[1:7]))
                last_move_side = "black" if line[0] == "+" else "white"
                continue

            if line.startswith("%"):
                terminal = line.strip()
                continue

            eval_cp = parse_eval(line)
            if eval_cp is not None and moves and moves[-1].eval_cp is None:
                moves[-1].eval_cp = eval_cp
                continue

            parsed_ending, parsed_result = parse_summary(line)
            if parsed_ending is not None:
                ending = parsed_ending
                result_abs = parsed_result

    if result_abs is None:
        ending, result_abs = infer_result_from_terminal(terminal, last_move_side)

    return ParsedGame(str(path), moves, result_abs, ending)


def ending_code(ending: str | None) -> int:
    if ending == "sennichite":
        return 1
    if ending == "toryo":
        return 0
    return 0


def convert_one(task: tuple[str, int, frozenset[str] | None]) -> ConvertResult:
    path_text, min_moves, allowed_endings = task
    path = Path(path_text)
    try:
        game = parse_csa(path)
        if game.result_abs is None:
            return ConvertResult(path_text, False, error="missing result")
        if allowed_endings is not None and game.ending not in allowed_endings:
            return ConvertResult(path_text, False,
                                 ending=game.ending or "",
                                 result_abs=game.result_abs,
                                 moves=len(game.moves),
                                 error=f"ending {game.ending!r} not allowed")
        if len(game.moves) < min_moves:
            return ConvertResult(path_text, False,
                                 ending=game.ending or "",
                                 result_abs=game.result_abs,
                                 moves=len(game.moves),
                                 error=f"too few moves: {len(game.moves)}")

        encoder = GameDataEncoder()
        encoder.set_startsfen("startpos")
        board = encoder.board
        missing_evals = 0

        for idx, rec in enumerate(game.moves):
            move = board.move_from_csa(rec.csa)
            if move == 0:
                return ConvertResult(
                    path_text, False,
                    ending=game.ending or "",
                    result_abs=game.result_abs,
                    moves=idx,
                    error=f"illegal CSA move at ply {idx + 1}: {rec.csa}",
                )

            move16 = move & 0xffff
            eval_cp = rec.eval_cp
            if eval_cp is None:
                eval_cp = 0
                missing_evals += 1

            encoder.write_uint16(move16)
            encoder.write_eval(eval_cp)
            board.push_move16(move16)

        encoder.write_game_result(game.result_abs)
        encoder.write_uint8(ending_code(game.ending))
        return ConvertResult(
            path_text,
            True,
            game_bytes=bytes(encoder.get_bytes()),
            moves=len(game.moves),
            ending=game.ending or "",
            result_abs=game.result_abs,
            missing_evals=missing_evals,
        )
    except Exception as exc:
        return ConvertResult(path_text, False, error=f"{type(exc).__name__}: {exc}")


def parse_allowed_endings(text: str) -> frozenset[str] | None:
    if text.strip().lower() in ("", "any", "all"):
        return None
    endings = frozenset(
        item.strip().lower()
        for item in text.split(",")
        if item.strip()
    )
    if not endings:
        return None
    return endings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CSA files recursively into one YaneuraOu .pack file.")
    parser.add_argument("--input", required=True,
                        help="Input directory containing CSA files")
    parser.add_argument("--output", required=True,
                        help="Output .pack path")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-moves", type=int, default=20)
    parser.add_argument("--allowed-endings", default="toryo,sennichite",
                        help="Comma-separated endings to include, or 'any'")
    parser.add_argument("--manifest", default=None,
                        help="CSV conversion manifest. Default: <output>.manifest.csv")
    parser.add_argument("--progress-interval", type=int, default=5000)
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_root.is_dir():
        print(f"Input directory does not exist: {input_root}", file=sys.stderr)
        return 2
    if args.min_moves < 0:
        print("--min-moves must be non-negative", file=sys.stderr)
        return 2

    allowed_endings = parse_allowed_endings(args.allowed_endings)
    manifest_path = (Path(args.manifest).resolve() if args.manifest
                     else output_path.with_suffix(output_path.suffix + ".manifest.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    csa_files = [str(path) for path in iter_csa_files(input_root)]
    print(f"Found {len(csa_files):,} CSA files")
    print(f"Output: {output_path}")
    print(f"Manifest: {manifest_path}")
    print(f"min_moves={args.min_moves}, allowed_endings="
          f"{'any' if allowed_endings is None else ','.join(sorted(allowed_endings))}")

    tasks = ((path, args.min_moves, allowed_endings) for path in csa_files)
    t0 = time.time()
    scanned = 0
    written_games = 0
    skipped_games = 0
    written_moves = 0
    missing_evals = 0

    fields = [
        "source", "ok", "moves", "ending", "result_abs",
        "missing_evals", "error",
    ]

    with output_path.open("wb", buffering=1024 * 1024) as pack_out, \
            manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fields)
        writer.writeheader()

        if args.workers <= 1:
            results = map(convert_one, tasks)
            pool = None
        else:
            pool = Pool(args.workers)
            results = pool.imap_unordered(convert_one, tasks, chunksize=128)

        try:
            for result in results:
                scanned += 1
                if result.ok:
                    pack_out.write(result.game_bytes)
                    written_games += 1
                    written_moves += result.moves
                    missing_evals += result.missing_evals
                else:
                    skipped_games += 1

                writer.writerow({
                    "source": result.path,
                    "ok": int(result.ok),
                    "moves": result.moves,
                    "ending": result.ending,
                    "result_abs": result.result_abs,
                    "missing_evals": result.missing_evals,
                    "error": result.error,
                })

                if args.progress_interval > 0 and scanned % args.progress_interval == 0:
                    elapsed = max(time.time() - t0, 1.0)
                    print(f"scanned={scanned:,} written_games={written_games:,} "
                          f"written_moves={written_moves:,} skipped={skipped_games:,} "
                          f"({written_moves / elapsed:.0f} moves/sec)",
                          flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    elapsed = max(time.time() - t0, 1.0)
    print(f"Done: scanned={scanned:,} written_games={written_games:,} "
          f"written_moves={written_moves:,} skipped={skipped_games:,} "
          f"missing_evals={missing_evals:,} time={elapsed:.1f}s "
          f"rate={written_moves / elapsed:.0f} moves/sec")
    print(f"Pack: {output_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
