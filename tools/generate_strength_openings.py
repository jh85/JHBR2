#!/usr/bin/env python3
"""Generate a deterministic, balanced opening suite from a YBB V1 book.

The output is a text file understood by tools/strength_test.py.  Every useful
line has the form:

    startpos moves 7g7f 3c3d ...

The generator follows reasonably good book moves, rather than always choosing
the single best move, to obtain a varied suite.  It keeps only final positions
whose best stored book evaluation is near equality.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import mmap
import random
import struct
import sys
from pathlib import Path

try:
    import cshogi
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised by wrapper
    raise SystemExit(
        "cshogi and numpy are required. Run this through "
        "tools/run_strength_test.sh or install cshogi."
    ) from exc


YBB_MAGIC = b"YANE-BINBOOK-V1\0"
HEADER_SIZE = 32
INDEX_SIZE = 44
FLAG_MOVE_DEPTH = 1


class YbbBook:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self.data) < HEADER_SIZE or self.data[:16] != YBB_MAGIC:
            raise ValueError(f"{path}: not a YBB V1 file")
        self.record_count, self.flags = struct.unpack_from("<QQ", self.data, 16)
        if self.flags & ~FLAG_MOVE_DEPTH:
            raise ValueError(f"{path}: unsupported YBB flags {self.flags:#x}")
        self.move_size = 6 if self.flags & FLAG_MOVE_DEPTH else 4
        self.moves_base = HEADER_SIZE + self.record_count * INDEX_SIZE
        if self.moves_base > len(self.data):
            raise ValueError(f"{path}: truncated YBB index")
        self._packed = np.empty(32, dtype=np.uint8)

    def close(self) -> None:
        self.data.close()
        self.file.close()

    def _key(self, board: "cshogi.Board") -> bytes:
        board.to_psfen(self._packed)
        return self._packed.tobytes()

    def lookup(self, board: "cshogi.Board") -> list[tuple[int, int, int]]:
        key = self._key(board)
        low, high = 0, self.record_count
        while low < high:
            middle = (low + high) // 2
            offset = HEADER_SIZE + middle * INDEX_SIZE
            candidate = self.data[offset : offset + 32]
            if candidate < key:
                low = middle + 1
            else:
                high = middle
        if low >= self.record_count:
            return []
        offset = HEADER_SIZE + low * INDEX_SIZE
        if self.data[offset : offset + 32] != key:
            return []
        moves_offset, _ply, move_count = struct.unpack_from(
            "<QHH", self.data, offset + 32
        )
        result: list[tuple[int, int, int]] = []
        for index in range(move_count):
            move_offset = self.moves_base + moves_offset + index * self.move_size
            if move_offset + self.move_size > len(self.data):
                raise ValueError(f"{self.path}: truncated YBB move area")
            if self.move_size == 6:
                move16, evaluation, depth = struct.unpack_from(
                    "<HhH", self.data, move_offset
                )
            else:
                move16, evaluation = struct.unpack_from(
                    "<Hh", self.data, move_offset
                )
                depth = 0
            result.append((move16, evaluation, depth))
        return result


def choose_move(
    book: YbbBook,
    board: "cshogi.Board",
    records: list[tuple[int, int, int]],
    rng: random.Random,
    eval_margin: int,
    temperature: float,
) -> int | None:
    legal: list[tuple[int, int]] = []
    for move16, evaluation, _depth in records:
        move = board.move_from_move16(move16)
        if not board.is_legal(move):
            continue
        # Distributed books can intentionally contain leaf moves whose child
        # position is outside the book.  Do not select one before the desired
        # opening position, because it cannot yield a book-evaluated endpoint.
        board.push(move)
        child_exists = bool(book.lookup(board))
        board.pop()
        if child_exists:
            legal.append((move, evaluation))
    if not legal:
        return None
    best = max(evaluation for _, evaluation in legal)
    candidates = [
        (move, evaluation)
        for move, evaluation in legal
        if evaluation >= best - eval_margin
    ]
    if temperature <= 0:
        best_moves = [move for move, evaluation in candidates if evaluation == best]
        return rng.choice(best_moves)
    weights = [
        math.exp((evaluation - best) / temperature)
        for _, evaluation in candidates
    ]
    return rng.choices([move for move, _ in candidates], weights=weights, k=1)[0]


def generate_one(
    book: YbbBook,
    rng: random.Random,
    target_ply: int,
    move_eval_margin: int,
    move_temperature: float,
    position_eval_limit: int,
) -> tuple[list[str], int] | None:
    board = cshogi.Board()
    moves: list[str] = []
    for _ in range(target_ply):
        records = book.lookup(board)
        if not records:
            return None
        move = choose_move(
            book, board, records, rng, move_eval_margin, move_temperature
        )
        if move is None:
            return None
        moves.append(cshogi.move_to_usi(move))
        board.push(move)
        if board.is_game_over() or board.is_draw() != cshogi.NOT_REPETITION:
            return None

    final_records = book.lookup(board)
    legal_evals = []
    for move16, evaluation, _depth in final_records:
        move = board.move_from_move16(move16)
        if board.is_legal(move):
            legal_evals.append(evaluation)
    if not legal_evals:
        return None
    best_eval = max(legal_evals)
    if abs(best_eval) > position_eval_limit:
        return None
    return moves, best_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book", type=Path, help="source YBB V1 book")
    parser.add_argument("output", type=Path, help="output opening-suite text file")
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=12)
    parser.add_argument(
        "--move-eval-margin",
        type=int,
        default=100,
        help="consider book moves this many cp below the best (default: 100)",
    )
    parser.add_argument(
        "--move-temperature",
        type=float,
        default=80.0,
        help="softmax temperature in cp; 0 chooses only best moves (default: 80)",
    )
    parser.add_argument(
        "--position-eval-limit",
        type=int,
        default=250,
        help="reject final positions with |best book eval| above this (default: 250)",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="attempt limit; default is max(10000, count*200)",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.min_ply < 0 or args.max_ply < args.min_ply:
        raise SystemExit("invalid --min-ply/--max-ply")
    if args.move_eval_margin < 0 or args.position_eval_limit < 0:
        raise SystemExit("evaluation limits must be non-negative")
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; pass --force to replace it")
    if not args.book.is_file():
        raise SystemExit(f"{args.book} is not a readable file")

    rng = random.Random(args.seed)
    attempts_limit = args.max_attempts or max(10_000, args.count * 200)
    book = YbbBook(args.book)
    selected: list[tuple[list[str], int]] = []
    seen: set[tuple[str, ...]] = set()
    try:
        for attempt in range(1, attempts_limit + 1):
            target_ply = rng.randint(args.min_ply, args.max_ply)
            generated = generate_one(
                book,
                rng,
                target_ply,
                args.move_eval_margin,
                args.move_temperature,
                args.position_eval_limit,
            )
            if generated is None:
                continue
            moves, evaluation = generated
            key = tuple(moves)
            if key in seen:
                continue
            seen.add(key)
            selected.append((moves, evaluation))
            if len(selected) % 50 == 0 or len(selected) == args.count:
                print(
                    f"generated {len(selected)}/{args.count} "
                    f"after {attempt} attempts",
                    file=sys.stderr,
                )
            if len(selected) >= args.count:
                break
    finally:
        book.close()

    if len(selected) < args.count:
        raise SystemExit(
            f"could generate only {len(selected)}/{args.count} unique openings "
            f"in {attempts_limit} attempts; relax the filters"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with args.book.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    source_hash = digest.hexdigest()
    with args.output.open("w", encoding="utf-8") as output:
        output.write("# JHBR2 paired strength-test opening suite\n")
        output.write(f"# source={args.book.resolve()}\n")
        output.write(f"# source_sha256={source_hash}\n")
        output.write(
            f"# seed={args.seed} count={args.count} "
            f"ply={args.min_ply}..{args.max_ply} "
            f"move_eval_margin={args.move_eval_margin} "
            f"move_temperature={args.move_temperature} "
            f"position_eval_limit={args.position_eval_limit}\n"
        )
        for moves, evaluation in selected:
            output.write(
                "startpos moves "
                + " ".join(moves)
                + f" # book_eval={evaluation}\n"
            )
    print(f"wrote {len(selected)} openings to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
