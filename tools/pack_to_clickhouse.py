#!/usr/bin/env python3
"""
Load YaneuraOu .pack kifu files into ClickHouse as per-position rows.

This is intended for duplicate-position analysis. It stores one row before
each played move:
  - position_key: SFEN without the move number (board, side-to-move, hands)
  - result_abs: 0 draw, 1 black win, 2 white win
  - moves_left: remaining plies to the recorded game end

Example:
  python3 tools/pack_to_clickhouse.py \
      --pack-dir /mnt/shogi_data/training_yane \
      --database shogi \
      --table pack_positions \
      --dataset training_yane \
      --create-table \
      --recursive
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import cshogi
except ImportError as exc:
    raise SystemExit(
        "Missing Python module: cshogi. Install it in the Python environment "
        "used for this script, then rerun."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for base in (ROOT, ROOT.parent):
    for subdir in ("YaneuraOu-ScriptCollection/CommonLib",
                   "YaneuraOu-ScriptCollection/GenSfen"):
        path = base / subdir
        if path.is_dir():
            sys.path.insert(0, str(path))

try:
    from YaneShogiLib import GameDataDecoder
except ImportError as exc:
    raise SystemExit(
        "Could not import YaneShogiLib.GameDataDecoder. Run this from the "
        "JHBR2 checkout with YaneuraOu-ScriptCollection next to it."
    ) from exc


IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

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


def iter_pack_files(pack_dir: Path, recursive: bool):
    pattern = "**/*.pack" if recursive else "*.pack"
    yield from sorted(pack_dir.glob(pattern))


def decode_pack_games(pack_path: Path):
    data = bytearray(pack_path.read_bytes())
    decoder = GameDataDecoder(data)
    while not decoder.eof():
        sfen = decoder.get_sfen()
        moves = []
        result_abs = 0
        while True:
            move = decoder.read_uint16()
            sq1 = move & 0x7f
            sq2 = (move >> 7) & 0x7f
            if sq1 == sq2:
                result_abs = sq1
                decoder.read_uint8()
                break
            eval16 = decoder.read_int16()
            moves.append((move, eval16))
        yield sfen, moves, result_abs


def position_key_from_sfen(sfen: str) -> str:
    # Drop only the move number. Keep board, side-to-move, and hands.
    return " ".join(sfen.split()[:3])


def rows_for_game(dataset: str, source_file: str, game_index: int,
                  start_sfen: str, moves, result_abs: int):
    board = cshogi.Board()
    board.set_sfen(start_sfen)
    n_moves = len(moves)
    for ply, (move16, eval16) in enumerate(moves):
        sfen = board.sfen()
        parts = sfen.split()
        side_to_move = 0 if parts[1] == "b" else 1
        moves_left = n_moves - ply - 1
        try:
            move_usi = cshogi.move_to_usi(move16)
        except Exception:
            move_usi = ""

        yield (
            dataset,
            source_file,
            game_index,
            ply,
            int(getattr(board, "move_number", ply + 1)),
            position_key_from_sfen(sfen),
            side_to_move,
            move_usi,
            move16 & 0xffff,
            int(eval16),
            int(result_abs),
            moves_left,
        )

        board.push_move16(move16)


def row_to_tsv(row) -> str:
    return "\t".join(tsv_escape(x) for x in row) + "\n"


def flush_rows(url: str, database: str, table: str, rows: list[str],
               auth=None) -> int:
    if not rows:
        return 0
    columns = ", ".join(INSERT_COLUMNS)
    query = f"INSERT INTO {database}.{table} ({columns}) FORMAT TabSeparated"
    payload = "".join(rows).encode("utf-8")
    ch_post(url, query, payload, timeout=1800, auth=auth)
    n = len(rows)
    rows.clear()
    return n


def load_pack_files(args):
    pack_dir = Path(args.pack_dir)
    if not pack_dir.is_dir():
        raise SystemExit(f"Not a directory: {pack_dir}")

    files = list(iter_pack_files(pack_dir, args.recursive))
    if args.limit_files:
        files = files[:args.limit_files]
    if not files:
        raise SystemExit(f"No .pack files found under {pack_dir}")

    auth = clickhouse_auth(args)

    if args.create_table:
        create_table(args.url, args.database, args.table, auth=auth)

    rows: list[str] = []
    total_rows = 0
    total_games = 0
    bad_games = 0
    t0 = time.time()

    for file_index, pack_path in enumerate(files, 1):
        file_rows = 0
        source_file = str(pack_path)
        try:
            game_iter = decode_pack_games(pack_path)
            for game_index, (sfen, moves, result_abs) in enumerate(game_iter):
                total_games += 1
                try:
                    for row in rows_for_game(args.dataset, source_file,
                                             game_index, sfen, moves,
                                             result_abs):
                        rows.append(row_to_tsv(row))
                        file_rows += 1
                        if len(rows) >= args.batch_rows:
                            total_rows += flush_rows(
                                args.url, args.database, args.table, rows,
                                auth=auth)
                except Exception as exc:
                    bad_games += 1
                    if bad_games <= 20:
                        print(f"skip game {source_file}:{game_index}: "
                              f"{type(exc).__name__}: {exc}",
                              file=sys.stderr)
        except Exception as exc:
            print(f"skip file {source_file}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue

        elapsed = max(time.time() - t0, 1e-6)
        print(f"{file_index}/{len(files)} {pack_path.name}: "
              f"{file_rows:,} rows, total {total_rows + len(rows):,}, "
              f"{(total_rows + len(rows)) / elapsed:,.0f} rows/s",
              flush=True)

    total_rows += flush_rows(args.url, args.database, args.table, rows,
                             auth=auth)
    elapsed = max(time.time() - t0, 1e-6)
    print(f"done: {total_games:,} games, {bad_games:,} bad games, "
          f"{total_rows:,} rows in {elapsed:.1f}s "
          f"({total_rows / elapsed:,.0f} rows/s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack-dir", required=True)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--dataset", default="pack")
    p.add_argument("--url", default="http://localhost:8123",
                   help="ClickHouse HTTP URL")
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
    p.add_argument("--limit-files", type=int, default=0,
                   help="For smoke tests: load only the first N pack files")
    args = p.parse_args()

    require_identifier(args.database)
    require_identifier(args.table)
    load_pack_files(args)


if __name__ == "__main__":
    main()
