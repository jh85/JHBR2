#!/usr/bin/env python3
"""
Load YaneuraOu PackedSfenValue / PSV .bin files into ClickHouse.

PSV records are fixed 40-byte rows:
  - packed_sfen: 32 bytes
  - score:       int16
  - move:        uint16
  - gamePly:     uint16
  - game_result: int8, from side-to-move's perspective
  - padding:     uint8

The ClickHouse table matches tools/pack_to_clickhouse.py so .pack and PSV
sources can be loaded into one table for duplicate-position analysis.

Example:
  python3 tools/psv_to_clickhouse.py \
      --psv-dir /mnt/shogi_data \
      --recursive \
      --dataset tanuki_2024_07_30 \
      --database shogi \
      --table pack_positions \
      --create-table
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


BLACK = 0
WHITE = 1

NO_PIECE = 0
PAWN = 1
LANCE = 2
KNIGHT = 3
SILVER = 4
BISHOP = 5
ROOK = 6
GOLD = 7
KING = 8
PRO_PAWN = 9
PRO_LANCE = 10
PRO_KNIGHT = 11
PRO_SILVER = 12
HORSE = 13
DRAGON = 14

RECORD_SIZE = 40
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

HUFFMAN_PIECES = (
    (0b0,      1, NO_PIECE),
    (0b01,     2, PAWN),
    (0b0011,   4, LANCE),
    (0b1011,   4, KNIGHT),
    (0b0111,   4, SILVER),
    (0b011111, 6, BISHOP),
    (0b111111, 6, ROOK),
    (0b01111,  5, GOLD),
)

PIECE_TO_SFEN = {
    PAWN: "P", LANCE: "L", KNIGHT: "N", SILVER: "S",
    BISHOP: "B", ROOK: "R", GOLD: "G", KING: "K",
    PRO_PAWN: "+P", PRO_LANCE: "+L", PRO_KNIGHT: "+N",
    PRO_SILVER: "+S", HORSE: "+B", DRAGON: "+R",
}

HAND_PIECES = {
    PAWN: "P", LANCE: "L", KNIGHT: "N", SILVER: "S",
    BISHOP: "B", ROOK: "R", GOLD: "G",
}

DROP_PIECES = {
    PAWN: "P", LANCE: "L", KNIGHT: "N", SILVER: "S",
    BISHOP: "B", ROOK: "R", GOLD: "G",
}

INSERT_COLUMNS = (
    "dataset",
    "source_file",
    "game_index",
    "ply",
    "game_ply",
    "position_key",
    "side_to_move",
    "move_usi",
    "move16",
    "eval",
    "result_abs",
    "moves_left",
)


@dataclass(frozen=True)
class PsvRecord:
    packed_sfen: bytes
    score: int
    move16: int
    game_ply: int
    game_result: int


class BitStream:
    def __init__(self, data: bytes):
        self.data = data
        self.cursor = 0

    def read_one_bit(self) -> int:
        if self.cursor >= 256:
            return 0
        byte_idx = self.cursor >> 3
        bit_idx = self.cursor & 7
        self.cursor += 1
        return (self.data[byte_idx] >> bit_idx) & 1

    def read_n_bits(self, n: int) -> int:
        value = 0
        for i in range(n):
            value |= self.read_one_bit() << i
        return value


def decode_piece(stream: BitStream) -> int:
    code = 0
    bits = 0
    while bits < 6:
        code |= stream.read_one_bit() << bits
        bits += 1
        for hcode, hbits, piece in HUFFMAN_PIECES:
            if hbits == bits and hcode == code:
                return piece
    return NO_PIECE


def decode_hand_piece(stream: BitStream):
    # Hand pieces use shifted Huffman codes: code >> 1, bits - 1.
    code = 0
    bits = 0
    while bits < 6 and stream.cursor < 256:
        code |= stream.read_one_bit() << bits
        bits += 1
        for hcode, hbits, piece in HUFFMAN_PIECES[1:]:
            if (hcode >> 1) == code and (hbits - 1) == bits:
                return piece
    return None


def decode_packed_sfen(sfen_bytes: bytes):
    stream = BitStream(sfen_bytes)
    turn = stream.read_one_bit()
    black_king_sq = stream.read_n_bits(7)
    white_king_sq = stream.read_n_bits(7)

    board = [0] * 81
    if black_king_sq < 81:
        board[black_king_sq] = KING
    if white_king_sq < 81:
        board[white_king_sq] = 16 + KING

    for sq in range(81):
        if sq == black_king_sq or sq == white_king_sq:
            continue
        piece = decode_piece(stream)
        if piece == NO_PIECE:
            continue
        promoted = False
        if piece != GOLD:
            promoted = stream.read_one_bit() == 1
        color = stream.read_one_bit()
        if promoted:
            piece += 8
        board[sq] = piece + (16 if color == WHITE else 0)

    hands = [{}, {}]
    while stream.cursor < 256:
        piece = decode_hand_piece(stream)
        if piece is None or stream.cursor >= 256:
            break
        promoted = False
        if piece != GOLD:
            promoted = stream.read_one_bit() == 1
        if stream.cursor >= 256:
            break
        color = stream.read_one_bit()
        if promoted:
            continue
        hand_piece = HAND_PIECES.get(piece)
        if hand_piece:
            hands[color][hand_piece] = hands[color].get(hand_piece, 0) + 1

    return board, hands, turn


def board_to_position_key(board, hands, turn: int) -> str:
    ranks = []
    for rank in range(9):
        empty = 0
        rank_text = []
        # YaneuraOu square index is file-major with file 1 first.
        # SFEN writes each rank from file 9 down to file 1.
        for ya_file in range(8, -1, -1):
            sq = ya_file * 9 + rank
            piece_val = board[sq]
            if piece_val == 0:
                empty += 1
                continue
            if empty:
                rank_text.append(str(empty))
                empty = 0
            color = WHITE if piece_val >= 16 else BLACK
            piece_type = piece_val & 15
            text = PIECE_TO_SFEN.get(piece_type, "?")
            if color == WHITE:
                text = "+" + text[1:].lower() if text.startswith("+") else text.lower()
            rank_text.append(text)
        if empty:
            rank_text.append(str(empty))
        ranks.append("".join(rank_text))

    hand_text = []
    for color in (BLACK, WHITE):
        for piece in ("R", "B", "G", "S", "N", "L", "P"):
            count = hands[color].get(piece, 0)
            if not count:
                continue
            ch = piece if color == BLACK else piece.lower()
            hand_text.append((str(count) if count > 1 else "") + ch)

    return "{} {} {}".format(
        "/".join(ranks),
        "b" if turn == BLACK else "w",
        "".join(hand_text) if hand_text else "-",
    )


def square_to_usi(sq: int) -> str:
    file_num = sq // 9 + 1
    rank = chr(ord("a") + (sq % 9))
    return f"{file_num}{rank}"


def move16_to_usi(move16: int) -> str:
    if move16 == 0:
        return ""
    to_sq = move16 & 0x7f
    from_raw = (move16 >> 7) & 0x7f
    is_drop = (move16 >> 14) & 1
    promote = (move16 >> 15) & 1
    if to_sq >= 81:
        return ""
    if is_drop:
        piece = DROP_PIECES.get(from_raw)
        return f"{piece}*{square_to_usi(to_sq)}" if piece else ""
    if from_raw >= 81:
        return ""
    return f"{square_to_usi(from_raw)}{square_to_usi(to_sq)}" + ("+" if promote else "")


def result_abs_from_psv(game_result: int, side_to_move: int) -> int:
    if game_result == 0:
        return 0
    stm_won = game_result > 0
    if side_to_move == BLACK:
        return 1 if stm_won else 2
    return 2 if stm_won else 1


def require_identifier(name: str) -> str:
    if not IDENT_RE.match(name):
        raise SystemExit(f"Invalid ClickHouse identifier: {name!r}")
    return name


def tsv_escape(value) -> str:
    if value is None:
        return r"\N"
    s = str(value)
    return (
        s.replace("\\", "\\\\")
         .replace("\t", "\\t")
         .replace("\n", "\\n")
         .replace("\r", "\\r")
         .replace("\0", "\\0")
    )


def clickhouse_auth(args):
    user = args.user
    password = args.password
    if args.password_file:
        password = Path(args.password_file).read_text().strip()
    if password is not None and user is None:
        user = "default"
    if user is None:
        return None
    return user, password or ""


def ch_post(url: str, query: str, data: bytes | None = None, timeout: int = 300,
            auth=None):
    endpoint = url.rstrip("/") + "/?query=" + urllib.parse.quote(query)
    req = urllib.request.Request(endpoint, data=data or b"", method="POST")
    if auth is not None:
        user, password = auth
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        hint = ""
        if exc.code == 401:
            hint = " Pass --user/--password or set CLICKHOUSE_USER and CLICKHOUSE_PASSWORD."
        raise RuntimeError(f"ClickHouse HTTP {exc.code}: {body}{hint}") from exc


def create_table(url: str, database: str, table: str, auth=None):
    database = require_identifier(database)
    table = require_identifier(table)
    ch_post(url, f"CREATE DATABASE IF NOT EXISTS {database}", auth=auth)
    ch_post(url, f"""
CREATE TABLE IF NOT EXISTS {database}.{table}
(
    dataset LowCardinality(String),
    source_file String,
    game_index UInt32,
    ply UInt16,
    game_ply UInt16,
    position_key String,
    pos_hash UInt64 MATERIALIZED cityHash64(position_key),
    side_to_move UInt8,
    move_usi LowCardinality(String),
    move16 UInt16,
    eval Int16,
    result_abs UInt8,
    moves_left UInt16,
    inserted_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY dataset
ORDER BY (pos_hash, position_key, dataset, source_file, game_index, ply)
SETTINGS index_granularity = 8192
""", auth=auth)


def row_to_tsv(row) -> str:
    return "\t".join(tsv_escape(x) for x in row) + "\n"


def flush_rows(url: str, database: str, table: str, rows: list[str],
               dry_run: bool, auth=None) -> int:
    if not rows:
        return 0
    if not dry_run:
        columns = ", ".join(INSERT_COLUMNS)
        query = f"INSERT INTO {database}.{table} ({columns}) FORMAT TabSeparated"
        ch_post(url, query, "".join(rows).encode("utf-8"), timeout=1800,
                auth=auth)
    n = len(rows)
    rows.clear()
    return n


def parse_record(buf: bytes) -> PsvRecord:
    score, move16, game_ply = struct.unpack_from("<hHH", buf, 32)
    game_result = struct.unpack_from("<b", buf, 38)[0]
    return PsvRecord(buf[:32], score, move16, game_ply, game_result)


def iter_psv_files(psv_dir: Path, glob_pattern: str, recursive: bool):
    pattern = f"**/{glob_pattern}" if recursive else glob_pattern
    yield from sorted(p for p in psv_dir.glob(pattern) if p.is_file())


def is_game_boundary(prev_ply: int | None, cur_ply: int) -> bool:
    return prev_ply is not None and cur_ply != prev_ply + 1


def rows_for_game(records: list[PsvRecord], dataset: str, source_file: str,
                  game_index: int):
    if not records:
        return
    final_ply = records[-1].game_ply
    for ply, rec in enumerate(records):
        board, hands, turn = decode_packed_sfen(rec.packed_sfen)
        position_key = board_to_position_key(board, hands, turn)
        result_abs = result_abs_from_psv(rec.game_result, turn)
        moves_left = max(0, final_ply - rec.game_ply)
        yield (
            dataset,
            source_file,
            game_index,
            ply,
            rec.game_ply,
            position_key,
            turn,
            move16_to_usi(rec.move16),
            rec.move16,
            rec.score,
            result_abs,
            moves_left,
        )


def load_file(path: Path, args, rows: list[str], total_inserted: int):
    size = path.stat().st_size
    if size % RECORD_SIZE != 0:
        raise ValueError(f"file size {size} is not divisible by {RECORD_SIZE}")

    source_file = str(path)
    game_records: list[PsvRecord] = []
    game_index = 0
    prev_ply = None
    file_rows = 0
    sample_rows_printed = 0
    max_records = args.limit_records if args.limit_records > 0 else None

    def emit_game():
        nonlocal game_records, game_index, file_rows
        nonlocal total_inserted, sample_rows_printed
        if not game_records:
            return
        for row in rows_for_game(game_records, args.dataset, source_file, game_index):
            text = row_to_tsv(row)
            if args.dry_run and sample_rows_printed < args.print_rows:
                print(text, end="")
                sample_rows_printed += 1
            rows.append(text)
            file_rows += 1
            if len(rows) >= args.batch_rows:
                total_inserted += flush_rows(
                    args.url, args.database, args.table, rows, args.dry_run,
                    auth=args.clickhouse_auth)
        game_index += 1
        game_records = []

    with path.open("rb") as f:
        rec_idx = 0
        while True:
            if max_records is not None and rec_idx >= max_records:
                break
            buf = f.read(RECORD_SIZE)
            if not buf:
                break
            if len(buf) != RECORD_SIZE:
                raise ValueError(f"short trailing record: {len(buf)} bytes")
            rec = parse_record(buf)
            if is_game_boundary(prev_ply, rec.game_ply):
                emit_game()
            game_records.append(rec)
            prev_ply = rec.game_ply
            rec_idx += 1

    emit_game()
    return file_rows, game_index, total_inserted


def load_psv_files(args):
    if args.psv_file:
        files = [Path(p) for p in args.psv_file]
    else:
        psv_dir = Path(args.psv_dir)
        if not psv_dir.is_dir():
            raise SystemExit(f"Not a directory: {psv_dir}")
        files = list(iter_psv_files(psv_dir, args.psv_glob, args.recursive))
    if args.limit_files:
        files = files[:args.limit_files]
    if not files:
        raise SystemExit("No PSV files found")

    args.clickhouse_auth = clickhouse_auth(args)

    if args.create_table and not args.dry_run:
        create_table(args.url, args.database, args.table,
                     auth=args.clickhouse_auth)

    rows: list[str] = []
    total_inserted = 0
    total_seen = 0
    total_games = 0
    bad_files = 0
    t0 = time.time()

    for file_idx, path in enumerate(files, 1):
        try:
            file_rows, file_games, total_inserted = load_file(
                path, args, rows, total_inserted)
            total_seen += file_rows
            total_games += file_games
        except Exception as exc:
            bad_files += 1
            print(f"skip file {path}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue

        elapsed = max(time.time() - t0, 1e-6)
        pending = len(rows)
        print(f"{file_idx}/{len(files)} {path.name}: "
              f"{file_rows:,} rows, {file_games:,} games, "
              f"total {total_seen:,} rows "
              f"({(total_inserted + pending) / elapsed:,.0f} rows/s)",
              flush=True)

    total_inserted += flush_rows(
        args.url, args.database, args.table, rows, args.dry_run,
        auth=args.clickhouse_auth)
    elapsed = max(time.time() - t0, 1e-6)
    action = "decoded" if args.dry_run else "inserted"
    print(f"done: {action} {total_inserted:,} rows, "
          f"{total_games:,} games, {bad_files:,} bad files, "
          f"{elapsed:.1f}s, {total_inserted / elapsed:,.0f} rows/s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--psv-dir", default=".")
    p.add_argument("--psv-file", action="append",
                   help="Load one explicit PSV file. May be repeated.")
    p.add_argument("--psv-glob", default="*.bin")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--dataset", default="psv")
    p.add_argument("--url", default="http://localhost:8123")
    p.add_argument("--user", default=os.environ.get("CLICKHOUSE_USER"),
                   help="ClickHouse user. Defaults to CLICKHOUSE_USER.")
    p.add_argument("--password", default=os.environ.get("CLICKHOUSE_PASSWORD"),
                   help="ClickHouse password. Defaults to CLICKHOUSE_PASSWORD.")
    p.add_argument("--password-file",
                   help="Read ClickHouse password from this file.")
    p.add_argument("--database", default="shogi")
    p.add_argument("--table", default="pack_positions")
    p.add_argument("--create-table", action="store_true")
    p.add_argument("--batch-rows", type=int, default=100_000)
    p.add_argument("--limit-files", type=int, default=0)
    p.add_argument("--limit-records", type=int, default=0,
                   help="For smoke tests: read only this many records per file")
    p.add_argument("--dry-run", action="store_true",
                   help="Decode and batch rows, but do not write to ClickHouse")
    p.add_argument("--print-rows", type=int, default=0,
                   help="With --dry-run, print the first N TabSeparated rows")
    args = p.parse_args()

    require_identifier(args.database)
    require_identifier(args.table)
    load_psv_files(args)


if __name__ == "__main__":
    main()
