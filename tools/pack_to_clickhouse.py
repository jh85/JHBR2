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

For compatibility with PSV --fast-raw tables, use --fast-raw with a fresh
table. It stores cshogi/YaneuraOu PackedSfen bytes as position_key.
"""

from __future__ import annotations

import argparse
import base64
import copy
import multiprocessing as mp
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
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing Python module: cshogi or numpy. Install them in the Python "
        "environment used for this script, then rerun."
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

FAST_RAW_INSERT_COLUMNS = (
    "dataset",
    "source_file",
    "game_index",
    "ply",
    "game_ply",
    "position_key",
    "side_to_move",
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


def create_table(url: str, database: str, table: str, auth=None,
                 fast_raw: bool = False):
    database = require_identifier(database)
    table = require_identifier(table)
    ch_post(url, f"CREATE DATABASE IF NOT EXISTS {database}", auth=auth)
    if fast_raw:
        ch_post(url, f"""
CREATE TABLE IF NOT EXISTS {database}.{table}
(
    dataset String,
    source_file String,
    game_index UInt32,
    ply UInt16,
    game_ply UInt16,
    position_key String,
    pos_hash UInt64 MATERIALIZED cityHash64(position_key),
    side_to_move UInt8,
    move_usi String DEFAULT '',
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
        return

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


def collect_pack_files(args):
    pack_files = getattr(args, "pack_files", None)
    if pack_files is not None:
        files = [Path(p) for p in pack_files]
    else:
        pack_dir = Path(args.pack_dir)
        if not pack_dir.is_dir():
            raise SystemExit(f"Not a directory: {pack_dir}")
        files = list(iter_pack_files(pack_dir, args.recursive))
    if args.limit_files:
        files = files[:args.limit_files]
    return files


def apply_file_shard(files, args):
    if args.file_shard_count < 1:
        raise SystemExit("--file-shard-count must be >= 1")
    if not 0 <= args.file_shard_index < args.file_shard_count:
        raise SystemExit("--file-shard-index must satisfy 0 <= index < count")
    if args.file_shard_count > 1:
        files = [
            path for idx, path in enumerate(files)
            if idx % args.file_shard_count == args.file_shard_index
        ]
    return files


def source_file_value(path: Path, mode: str) -> str:
    if mode == "empty":
        return ""
    if mode == "basename":
        return path.name
    return str(path)


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


def board_to_packed_sfen_bytes(board: cshogi.Board, psfen_buf) -> bytes:
    board.to_psfen(psfen_buf)
    return psfen_buf.tobytes()


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


def fast_raw_rows_for_game(dataset: str, source_file: str, game_index: int,
                           start_sfen: str, moves, result_abs: int,
                           psfen_buf):
    board = cshogi.Board()
    board.set_sfen(start_sfen)
    n_moves = len(moves)
    for ply, (move16, eval16) in enumerate(moves):
        side_to_move = int(board.turn)
        moves_left = n_moves - ply - 1
        yield (
            dataset,
            source_file,
            game_index,
            ply,
            int(getattr(board, "move_number", ply + 1)),
            board_to_packed_sfen_bytes(board, psfen_buf),
            side_to_move,
            move16 & 0xffff,
            int(eval16),
            int(result_abs),
            moves_left,
        )

        board.push_move16(move16)


def row_to_tsv(row) -> str:
    return "\t".join(tsv_escape(x) for x in row) + "\n"


def write_varuint(out: bytearray, value: int):
    while value >= 0x80:
        out.append((value & 0x7f) | 0x80)
        value >>= 7
    out.append(value)


def write_rb_string(out: bytearray, value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    write_varuint(out, len(value))
    out.extend(value)


def row_to_rowbinary(row) -> bytes:
    out = bytearray()
    write_rb_string(out, row[0])
    write_rb_string(out, row[1])
    out.extend(int(row[2]).to_bytes(4, "little", signed=False))
    out.extend(int(row[3]).to_bytes(2, "little", signed=False))
    out.extend(int(row[4]).to_bytes(2, "little", signed=False))
    write_rb_string(out, row[5])
    out.append(int(row[6]))
    out.extend(int(row[7]).to_bytes(2, "little", signed=False))
    out.extend(int(row[8]).to_bytes(2, "little", signed=True))
    out.append(int(row[9]))
    out.extend(int(row[10]).to_bytes(2, "little", signed=False))
    return bytes(out)


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


def flush_fast_raw_rows(url: str, database: str, table: str,
                        rows: list[bytes], auth=None) -> int:
    if not rows:
        return 0
    columns = ", ".join(FAST_RAW_INSERT_COLUMNS)
    query = f"INSERT INTO {database}.{table} ({columns}) FORMAT RowBinary"
    payload = b"".join(rows)
    ch_post(url, query, payload, timeout=1800, auth=auth)
    n = len(rows)
    rows.clear()
    return n


def load_pack_files(args):
    prefix = getattr(args, "worker_prefix", "")
    files = apply_file_shard(collect_pack_files(args), args)
    if not files:
        raise SystemExit(f"No .pack files found under {args.pack_dir}")

    auth = clickhouse_auth(args)

    if args.create_table:
        create_table(args.url, args.database, args.table, auth=auth,
                     fast_raw=args.fast_raw)

    rows: list[str] = []
    raw_rows: list[bytes] = []
    psfen_buf = np.empty(1, dtype=cshogi.PackedSfen)
    total_rows = 0
    total_games = 0
    bad_games = 0
    t0 = time.time()

    for file_index, pack_path in enumerate(files, 1):
        file_rows = 0
        source_file = source_file_value(pack_path, args.source_file_field)
        try:
            game_iter = decode_pack_games(pack_path)
            for game_index, (sfen, moves, result_abs) in enumerate(game_iter):
                total_games += 1
                try:
                    if args.fast_raw:
                        row_iter = fast_raw_rows_for_game(
                            args.dataset, source_file, game_index, sfen,
                            moves, result_abs, psfen_buf)
                        for row in row_iter:
                            raw_rows.append(row_to_rowbinary(row))
                            file_rows += 1
                            if len(raw_rows) >= args.batch_rows:
                                total_rows += flush_fast_raw_rows(
                                    args.url, args.database, args.table,
                                    raw_rows, auth=auth)
                    else:
                        row_iter = rows_for_game(
                            args.dataset, source_file, game_index, sfen,
                            moves, result_abs)
                        for row in row_iter:
                            rows.append(row_to_tsv(row))
                            file_rows += 1
                            if len(rows) >= args.batch_rows:
                                total_rows += flush_rows(
                                    args.url, args.database, args.table, rows,
                                    auth=auth)
                except Exception as exc:
                    bad_games += 1
                    if bad_games <= 20:
                        print(f"{prefix}skip game {source_file}:{game_index}: "
                              f"{type(exc).__name__}: {exc}",
                              file=sys.stderr)
        except Exception as exc:
            print(f"{prefix}skip file {source_file}: "
                  f"{type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue

        elapsed = max(time.time() - t0, 1e-6)
        pending = len(raw_rows) if args.fast_raw else len(rows)
        print(f"{prefix}{file_index}/{len(files)} {pack_path.name}: "
              f"{file_rows:,} rows, total {total_rows + pending:,}, "
              f"{(total_rows + pending) / elapsed:,.0f} rows/s",
              flush=True)

    if args.fast_raw:
        total_rows += flush_fast_raw_rows(args.url, args.database, args.table,
                                          raw_rows, auth=auth)
    else:
        total_rows += flush_rows(args.url, args.database, args.table, rows,
                                 auth=auth)
    elapsed = max(time.time() - t0, 1e-6)
    print(f"{prefix}done: {total_games:,} games, {bad_games:,} bad games, "
          f"{total_rows:,} rows in {elapsed:.1f}s "
          f"({total_rows / elapsed:,.0f} rows/s)")


def run_threaded(args):
    if args.file_shard_count != 1 or args.file_shard_index != 0:
        raise SystemExit("--threads cannot be combined with manual "
                         "--file-shard-count/--file-shard-index")

    files = collect_pack_files(args)
    if not files:
        raise SystemExit(f"No .pack files found under {args.pack_dir}")

    n_workers = min(args.threads, len(files))
    auth = clickhouse_auth(args)
    if args.create_table:
        create_table(args.url, args.database, args.table, auth=auth,
                     fast_raw=args.fast_raw)

    chunks = [files[i::n_workers] for i in range(n_workers)]
    print(f"starting {n_workers} worker processes for {len(files):,} files",
          flush=True)

    processes = []
    for worker_idx, chunk in enumerate(chunks):
        worker_args = copy.copy(args)
        worker_args.pack_files = [str(path) for path in chunk]
        worker_args.limit_files = 0
        worker_args.file_shard_count = 1
        worker_args.file_shard_index = 0
        worker_args.create_table = False
        worker_args.threads = 1
        worker_args.worker_prefix = f"[worker {worker_idx}/{n_workers}] "
        proc = mp.Process(target=load_pack_files, args=(worker_args,),
                          name=f"pack-loader-{worker_idx}")
        proc.start()
        processes.append(proc)

    failed = []
    try:
        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                failed.append((proc.name, proc.exitcode))
    except KeyboardInterrupt:
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.join()
        raise

    if failed:
        details = ", ".join(f"{name}={code}" for name, code in failed)
        raise SystemExit(f"{len(failed)} worker process(es) failed: {details}")


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
    p.add_argument("--create-table", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Create database/table if missing (default: true). "
                        "Use --no-create-table to require an existing table.")
    p.add_argument("--batch-rows", type=int, default=100_000)
    p.add_argument("--fast-raw", action="store_true",
                   help="Fast/compatible mode: store raw 32-byte PackedSfen "
                        "as position_key and insert RowBinary. Use the same "
                        "fresh table as PSV --fast-raw.")
    p.add_argument("--source-file-field",
                   choices=["path", "basename", "empty"], default="path",
                   help="Value inserted into source_file. Use empty for less "
                        "IO when source provenance is not needed.")
    p.add_argument("--file-shard-count", type=int, default=1,
                   help="Split sorted file list into N shards for manual "
                        "parallel loading")
    p.add_argument("--file-shard-index", type=int, default=0,
                   help="Load only files whose sorted index modulo shard "
                        "count equals this index")
    p.add_argument("--threads", type=int, default=1,
                   help="Run N parallel worker processes. This is a "
                        "convenience wrapper around file sharding.")
    p.add_argument("--limit-files", type=int, default=0,
                   help="For smoke tests: load only the first N pack files")
    args = p.parse_args()

    require_identifier(args.database)
    require_identifier(args.table)
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if args.threads > 1:
        run_threaded(args)
    else:
        load_pack_files(args)


if __name__ == "__main__":
    main()
