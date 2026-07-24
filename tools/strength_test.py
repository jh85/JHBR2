#!/usr/bin/env python3
"""Reproducible, GPU-aware paired-game strength tests for USI engines."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import queue
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import IO, Any

try:
    import cshogi
except ImportError as exc:  # pragma: no cover - exercised by wrapper
    raise SystemExit(
        "cshogi is required. Run this through tools/run_strength_test.sh "
        "or install cshogi."
    ) from exc


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_options(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"option must be NAME=VALUE, got: {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"empty option name in: {item!r}")
        result[name] = value
    return result


def has_option(options: dict[str, str], name: str) -> bool:
    wanted = name.casefold()
    return any(option.casefold() == wanted for option in options)


def detect_gpu_devices(explicit: str | None) -> list[str]:
    if explicit is not None:
        value = explicit.strip()
        if value.casefold() in {"", "none", "cpu"}:
            return []
        return [part.strip() for part in value.split(",") if part.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() not in {"", "-1"}:
        return [part.strip() for part in visible.split(",") if part.strip()]
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gpu_description() -> str:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return ""
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


@dataclasses.dataclass(frozen=True)
class Opening:
    source_line: str
    base: str
    initial_moves: tuple[str, ...]

    def position_command(self, played_moves: list[str]) -> str:
        moves = [*self.initial_moves, *played_moves]
        return "position " + self.base + ((" moves " + " ".join(moves)) if moves else "")

    def make_board(self) -> "cshogi.Board":
        board = cshogi.Board()
        if self.base != "startpos":
            board.set_sfen(self.base.removeprefix("sfen "))
        for move_text in self.initial_moves:
            move = board.move_from_usi(move_text)
            if move == cshogi.NONE or not board.is_legal(move):
                raise ValueError(
                    f"illegal opening move {move_text!r} in {self.source_line!r}"
                )
            board.push(move)
        return board


def parse_opening(line: str) -> Opening:
    content = line.split("#", 1)[0].strip()
    tokens = content.split()
    if not tokens:
        raise ValueError("empty opening")
    if tokens[0] == "startpos":
        if len(tokens) == 1:
            return Opening(line.strip(), "startpos", ())
        if tokens[1] != "moves":
            raise ValueError(f"expected 'moves' after startpos: {line!r}")
        return Opening(line.strip(), "startpos", tuple(tokens[2:]))
    if tokens[0] == "sfen":
        if "moves" in tokens:
            split = tokens.index("moves")
            sfen_tokens = tokens[1:split]
            moves = tuple(tokens[split + 1 :])
        else:
            sfen_tokens = tokens[1:]
            moves = ()
        if len(sfen_tokens) != 4:
            raise ValueError(f"SFEN must contain four fields: {line!r}")
        return Opening(line.strip(), "sfen " + " ".join(sfen_tokens), moves)
    raise ValueError(f"opening must begin with startpos or sfen: {line!r}")


def load_openings(path: Path) -> list[Opening]:
    openings: list[Opening] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            opening = parse_opening(line)
            opening.make_board()
            openings.append(opening)
        except Exception as exc:
            raise SystemExit(f"{path}:{line_number}: {exc}") from exc
    if not openings:
        raise SystemExit(f"{path}: no openings found")
    return openings


_artifact_cache: dict[tuple[str, int, int], dict[str, Any]] = {}


def artifact_metadata(
    command: str, cwd: Path | None, options: dict[str, str]
) -> list[dict[str, Any]]:
    base = cwd.resolve() if cwd else Path.cwd()
    candidates: list[Path] = []
    argv = shlex.split(command)
    for index, token in enumerate(argv):
        if index == 0 and "/" not in token:
            located = shutil.which(token)
            if located:
                candidates.append(Path(located))
            continue
        candidate = Path(token).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.is_file():
            candidates.append(candidate)
    for value in options.values():
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.is_file():
            candidates.append(candidate)

    result: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        stat = resolved.stat()
        cache_key = (str(resolved), stat.st_size, stat.st_mtime_ns)
        metadata = _artifact_cache.get(cache_key)
        if metadata is None:
            metadata = {
                "path": str(resolved),
                "size": stat.st_size,
                "sha256": sha256_file(resolved),
            }
            _artifact_cache[cache_key] = metadata
        result.append(metadata)
    return result


class UsiError(RuntimeError):
    pass


class UsiEngine:
    def __init__(
        self,
        label: str,
        command: str,
        cwd: Path | None,
        options: dict[str, str],
        environment: dict[str, str],
        log_path: Path,
        startup_timeout: float,
        protocol_log: bool,
    ):
        self.label = label
        self.command = command
        self.argv = shlex.split(command)
        if not self.argv:
            raise UsiError(f"{label}: empty engine command")
        self.cwd = cwd
        self.options = options
        self.environment = environment
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.protocol_log = protocol_log
        self.process: subprocess.Popen[str] | None = None
        self.output: queue.Queue[str | None] = queue.Queue()
        self.log_file: IO[str] | None = None
        self.reader_threads: list[threading.Thread] = []
        self.name = label

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("a", encoding="utf-8", buffering=1)
        self.log_file.write(f"\n[{utc_now()}] start {self.label}: {self.command}\n")
        self.process = subprocess.Popen(
            self.argv,
            cwd=self.cwd,
            env=self.environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_thread = threading.Thread(
            target=self._read_stdout, args=(self.process,), daemon=True
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr, args=(self.process,), daemon=True
        )
        self.reader_threads = [stdout_thread, stderr_thread]
        for thread in self.reader_threads:
            thread.start()
        self.send("usi")
        deadline = time.monotonic() + self.startup_timeout
        while True:
            line = self.read_line(deadline)
            if line.startswith("id name "):
                self.name = line.removeprefix("id name ").strip()
            if line == "usiok":
                break
        for name, value in self.options.items():
            self.send(f"setoption name {name} value {value}")
        self.send("isready")
        self.wait_for("readyok", self.startup_timeout)

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                stripped = line.rstrip("\r\n")
                if self.protocol_log and self.log_file:
                    self.log_file.write(f"< {stripped}\n")
                self.output.put(stripped)
        finally:
            self.output.put(None)

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            if self.log_file:
                self.log_file.write(f"! {line}")

    def send(self, command: str) -> None:
        if (
            self.process is None
            or self.process.poll() is not None
            or self.process.stdin is None
        ):
            raise UsiError(f"{self.label}: engine is not running")
        if self.protocol_log and self.log_file:
            self.log_file.write(f"> {command}\n")
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise UsiError(f"{self.label}: failed to send {command!r}") from exc

    def read_line(self, deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise UsiError(f"{self.label}: timed out waiting for output")
        try:
            line = self.output.get(timeout=remaining)
        except queue.Empty as exc:
            raise UsiError(f"{self.label}: timed out waiting for output") from exc
        if line is None:
            code = self.process.poll() if self.process else None
            raise UsiError(f"{self.label}: engine exited (code {code})")
        return line

    def wait_for(self, expected: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while self.read_line(deadline) != expected:
            pass

    def new_game(self) -> None:
        self.send("usinewgame")

    def search(
        self, position: str, go_command: str, timeout: float
    ) -> tuple[str, str | None, int]:
        self.send(position)
        self.send(go_command)
        deadline = time.monotonic() + timeout
        last_info: str | None = None
        started = time.monotonic()
        while True:
            line = self.read_line(deadline)
            if line.startswith("info "):
                last_info = line
            elif line.startswith("bestmove "):
                fields = line.split()
                if len(fields) < 2:
                    raise UsiError(f"{self.label}: malformed bestmove: {line!r}")
                elapsed_ms = round((time.monotonic() - started) * 1000)
                return fields[1], last_info, elapsed_ms

    def gameover(self, result: str) -> None:
        self.send(f"gameover {result}")

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            try:
                if process.stdin:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
                process.wait(timeout=3)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        for thread in self.reader_threads:
            thread.join(timeout=1)
        self.reader_threads = []
        if self.log_file:
            self.log_file.write(f"[{utc_now()}] stop {self.label}\n")
            self.log_file.close()
            self.log_file = None


def player_result(winner: int | None, color: int) -> str:
    if winner is None:
        return "draw"
    return "win" if winner == color else "lose"


def repetition_winner(board: "cshogi.Board", result: int) -> int | None:
    if result == cshogi.REPETITION_DRAW:
        return None
    if result in (cshogi.REPETITION_WIN, cshogi.REPETITION_SUPERIOR):
        return board.turn
    if result in (cshogi.REPETITION_LOSE, cshogi.REPETITION_INFERIOR):
        return board.turn ^ 1
    return None


def play_game(
    pair_index: int,
    game_in_pair: int,
    opening_index: int,
    opening: Opening,
    black_label: str,
    white_label: str,
    engines: dict[str, UsiEngine],
    nodes: int | None,
    main_time_ms: int,
    byoyomi_ms: int,
    increment_ms: int,
    move_timeout: float,
    clock_grace_ms: int,
    max_plies: int,
) -> dict[str, Any]:
    board = opening.make_board()
    played_moves: list[str] = []
    moves: list[dict[str, Any]] = []
    clocks = [main_time_ms, main_time_ms]
    for engine in engines.values():
        engine.new_game()

    winner: int | None = None
    reason = "max_plies"
    for _ply in range(max_plies):
        color = board.turn
        label = black_label if color == cshogi.BLACK else white_label
        engine = engines[label]
        if nodes is not None:
            go_command = f"go nodes {nodes}"
            search_timeout = move_timeout
        else:
            go_command = (
                f"go btime {clocks[cshogi.BLACK]} "
                f"wtime {clocks[cshogi.WHITE]} "
                f"byoyomi {byoyomi_ms} "
                f"binc {increment_ms} winc {increment_ms}"
            )
            legal_think_ms = clocks[color] + byoyomi_ms + clock_grace_ms
            search_timeout = max(move_timeout, legal_think_ms / 1000.0 + 5.0)
        bestmove, last_info, elapsed_ms = engine.search(
            opening.position_command(played_moves), go_command, search_timeout
        )
        move_record: dict[str, Any] = {
            "ply": len(opening.initial_moves) + len(played_moves) + 1,
            "engine": label,
            "color": "black" if color == cshogi.BLACK else "white",
            "bestmove": bestmove,
            "elapsed_ms": elapsed_ms,
        }
        if last_info:
            move_record["last_info"] = last_info
        moves.append(move_record)

        if nodes is None:
            available_ms = clocks[color] + byoyomi_ms + clock_grace_ms
            if elapsed_ms > available_ms:
                winner = color ^ 1
                reason = "time_forfeit"
                break
            clocks[color] = max(0, clocks[color] - elapsed_ms) + increment_ms

        if bestmove == "resign":
            winner = color ^ 1
            reason = "resign"
            break
        if bestmove == "win":
            if board.is_nyugyoku():
                winner = color
                reason = "declaration"
            else:
                winner = color ^ 1
                reason = "invalid_declaration"
            break
        move = board.move_from_usi(bestmove)
        if move == cshogi.NONE or not board.is_legal(move):
            winner = color ^ 1
            reason = "illegal_move"
            break
        board.push(move)
        played_moves.append(bestmove)
        repetition = board.is_draw()
        if repetition != cshogi.NOT_REPETITION:
            winner = repetition_winner(board, repetition)
            reason = {
                cshogi.REPETITION_DRAW: "repetition_draw",
                cshogi.REPETITION_WIN: "repetition_win",
                cshogi.REPETITION_LOSE: "repetition_lose",
                cshogi.REPETITION_SUPERIOR: "repetition_superior",
                cshogi.REPETITION_INFERIOR: "repetition_inferior",
            }.get(repetition, "repetition")
            break
        if board.is_game_over():
            winner = board.turn ^ 1
            reason = "checkmate"
            break

    engines[black_label].gameover(player_result(winner, cshogi.BLACK))
    engines[white_label].gameover(player_result(winner, cshogi.WHITE))
    if winner is None:
        winner_label = None
    else:
        winner_label = black_label if winner == cshogi.BLACK else white_label
    return {
        "pair_index": pair_index,
        "game_in_pair": game_in_pair,
        "opening_index": opening_index,
        "black": black_label,
        "white": white_label,
        "winner": winner_label,
        "reason": reason,
        "initial_ply": len(opening.initial_moves),
        "played_plies": len(played_moves),
        "final_clock_ms": {
            "black": clocks[cshogi.BLACK],
            "white": clocks[cshogi.WHITE],
        }
        if nodes is None
        else None,
        "moves": moves,
    }


def score_for_a(game: dict[str, Any]) -> float:
    winner = game["winner"]
    return 0.5 if winner is None else (1.0 if winner == "A" else 0.0)


def elo_from_score(score: float) -> float | str:
    if score <= 0:
        return "-inf"
    if score >= 1:
        return "+inf"
    return 400.0 * math.log10(score / (1.0 - score))


def summarize(records: list[dict[str, Any]], requested_pairs: int) -> dict[str, Any]:
    successful = [record for record in records if "games" in record]
    failures = [record for record in records if "error" in record]
    wins = losses = draws = 0
    pair_scores: list[float] = []
    pentanomial: Counter[int] = Counter()
    reasons: Counter[str] = Counter()
    for record in successful:
        points = 0.0
        for game in record["games"]:
            score = score_for_a(game)
            points += score
            reasons[game["reason"]] += 1
            if score == 1:
                wins += 1
            elif score == 0:
                losses += 1
            else:
                draws += 1
        pair_scores.append(points / 2.0)
        pentanomial[round(points * 2)] += 1

    n = len(pair_scores)
    if n:
        score = statistics.fmean(pair_scores)
    else:
        score = 0.5
    if n:
        # A Jeffreys-smoothed Dirichlet posterior over the five possible
        # paired scores avoids the zero-width interval produced by the usual
        # sample-variance formula when the first few pairs all split 1-1.
        score_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        alphas = [pentanomial[index] + 0.5 for index in range(5)]
        alpha_total = sum(alphas)
        posterior_mean = sum(
            alpha * level for alpha, level in zip(alphas, score_levels)
        ) / alpha_total
        posterior_second_moment = sum(
            alpha * level * level
            for alpha, level in zip(alphas, score_levels)
        ) / alpha_total
        posterior_variance = (
            posterior_second_moment - posterior_mean * posterior_mean
        ) / (alpha_total + 1.0)
        standard_error = math.sqrt(max(0.0, posterior_variance))
        low = max(0.0, posterior_mean - 1.96 * standard_error)
        high = min(1.0, posterior_mean + 1.96 * standard_error)
        if standard_error > 0:
            los = statistics.NormalDist().cdf(
                (posterior_mean - 0.5) / standard_error
            )
        else:
            los = (
                1.0
                if posterior_mean > 0.5
                else (0.0 if posterior_mean < 0.5 else 0.5)
            )
    else:
        standard_error = None
        low, high, los = 0.0, 1.0, 0.5
    return {
        "requested_pairs": requested_pairs,
        "completed_pairs": n,
        "failed_pairs": len(failures),
        "games": wins + losses + draws,
        "a_wins": wins,
        "a_losses": losses,
        "draws": draws,
        "score": score,
        "elo": elo_from_score(score),
        "elo_95ci": [elo_from_score(low), elo_from_score(high)],
        "score_standard_error": standard_error,
        "los": los,
        "pentanomial_a_points_0_to_2": [
            pentanomial[index] for index in range(5)
        ],
        "termination_reasons": dict(sorted(reasons.items())),
        "updated_utc": utc_now(),
    }


def format_elo(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:+.1f}"


def print_summary(summary: dict[str, Any]) -> None:
    low, high = summary["elo_95ci"]
    print(
        f"pairs {summary['completed_pairs']}/{summary['requested_pairs']} "
        f"(failed {summary['failed_pairs']}), "
        f"A {summary['a_wins']}-{summary['a_losses']}-{summary['draws']}, "
        f"score {summary['score'] * 100:.2f}%, "
        f"Elo {format_elo(summary['elo'])} "
        f"[{format_elo(low)}, {format_elo(high)}], "
        f"LOS {summary['los'] * 100:.1f}%",
        flush=True,
    )


class MatchRun:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.openings = load_openings(args.openings)
        self.devices = detect_gpu_devices(args.gpu_devices)
        if args.gpus_per_worker <= 0:
            raise SystemExit("--gpus-per-worker must be positive")
        self.gpu_groups = [
            self.devices[index : index + args.gpus_per_worker]
            for index in range(0, len(self.devices), args.gpus_per_worker)
            if len(self.devices[index : index + args.gpus_per_worker])
            == args.gpus_per_worker
        ]
        if args.workers:
            worker_count = args.workers
        else:
            worker_count = len(self.gpu_groups) if self.gpu_groups else 1
        if self.gpu_groups and worker_count > len(self.gpu_groups):
            raise SystemExit(
                f"{worker_count} workers requested but only "
                f"{len(self.gpu_groups)} complete GPU groups are available"
            )
        self.worker_count = worker_count
        self.options_a = parse_options(args.option_a)
        self.options_b = parse_options(args.option_b)
        if self.gpu_groups:
            for options in (self.options_a, self.options_b):
                if not has_option(options, "NumGPUs"):
                    options["NumGPUs"] = str(args.gpus_per_worker)
        self.run_dir = args.output.resolve()
        self.results_path = self.run_dir / "pairs.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.summary_path = self.run_dir / "summary.json"
        self.write_lock = threading.Lock()
        self.records: list[dict[str, Any]] = []
        self.completed: set[int] = set()
        self.work: queue.Queue[tuple[int, int, Opening]] = queue.Queue()
        self.config = self._make_config()

    def _make_config(self) -> dict[str, Any]:
        stable = {
            "schema": 1,
            "engine_a": self.args.engine_a,
            "engine_b": self.args.engine_b,
            "cwd_a": str(self.args.cwd_a.resolve()) if self.args.cwd_a else None,
            "cwd_b": str(self.args.cwd_b.resolve()) if self.args.cwd_b else None,
            "options_a": self.options_a,
            "options_b": self.options_b,
            "artifacts_a": artifact_metadata(
                self.args.engine_a, self.args.cwd_a, self.options_a
            ),
            "artifacts_b": artifact_metadata(
                self.args.engine_b, self.args.cwd_b, self.options_b
            ),
            "openings": str(self.args.openings.resolve()),
            "openings_sha256": sha256_file(self.args.openings),
            "pairs": self.args.pairs,
            "seed": self.args.seed,
            "nodes": self.args.nodes,
            "main_time_ms": self.args.main_time_ms,
            "byoyomi_ms": self.args.byoyomi_ms,
            "increment_ms": self.args.increment_ms,
            "clock_grace_ms": self.args.clock_grace_ms,
            "move_timeout": self.args.move_timeout,
            "max_plies": self.args.max_plies,
            "gpus_per_worker": self.args.gpus_per_worker,
        }
        fingerprint = hashlib.sha256(
            json.dumps(stable, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            **stable,
            "fingerprint": fingerprint,
            "created_utc": utc_now(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "cshogi": importlib.metadata.version("cshogi"),
            "workers": self.worker_count,
            "gpu_devices": self.devices,
            "gpu_description": gpu_description(),
            "git_revision": git_revision(),
        }

    def prepare(self) -> None:
        if self.run_dir.exists() and not self.args.resume:
            raise SystemExit(
                f"{self.run_dir} already exists; choose another --output "
                "or pass --resume"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.args.resume:
            if not self.config_path.is_file():
                raise SystemExit(f"{self.run_dir}: cannot resume without config.json")
            old_config = json.loads(self.config_path.read_text(encoding="utf-8"))
            if old_config.get("fingerprint") != self.config["fingerprint"]:
                raise SystemExit(
                    f"{self.run_dir}: configuration differs from the saved run"
                )
            if self.results_path.exists():
                for line in self.results_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self.records.append(record)
                    self.completed.add(record["pair_index"])
        else:
            self.config_path.write_text(
                json.dumps(self.config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        order = list(range(len(self.openings)))
        rng = random.Random(self.args.seed)
        rng.shuffle(order)
        for pair_index in range(self.args.pairs):
            if pair_index in self.completed:
                continue
            opening_index = order[pair_index % len(order)]
            self.work.put(
                (pair_index, opening_index, self.openings[opening_index])
            )
        self._write_summary()

    def _write_summary(self) -> None:
        summary = summarize(self.records, self.args.pairs)
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.summary_path)
        print_summary(summary)

    def save_record(self, record: dict[str, Any]) -> None:
        with self.write_lock:
            with self.results_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
            self.records.append(record)
            self.completed.add(record["pair_index"])
            self._write_summary()

    def run(self) -> int:
        self.prepare()
        threads = [
            threading.Thread(target=self.worker, args=(index,), daemon=True)
            for index in range(self.worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        final = summarize(self.records, self.args.pairs)
        return 1 if final["failed_pairs"] else 0

    def worker(self, worker_index: int) -> None:
        if self.gpu_groups:
            group = self.gpu_groups[worker_index]
        else:
            group = []
        environment = os.environ.copy()
        if group:
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(group)
        elif self.args.gpu_devices is not None:
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
        engines: dict[str, UsiEngine] = {}
        try:
            while True:
                try:
                    pair_index, opening_index, opening = self.work.get_nowait()
                except queue.Empty:
                    return
                record: dict[str, Any] | None = None
                last_error: Exception | None = None
                for attempt in range(self.args.retries + 1):
                    try:
                        if not engines:
                            engines = self.start_engines(
                                worker_index, environment
                            )
                        games = [
                            play_game(
                                pair_index,
                                0,
                                opening_index,
                                opening,
                                "A",
                                "B",
                                engines,
                                self.args.nodes,
                                self.args.main_time_ms,
                                self.args.byoyomi_ms,
                                self.args.increment_ms,
                                self.args.move_timeout,
                                self.args.clock_grace_ms,
                                self.args.max_plies,
                            ),
                            play_game(
                                pair_index,
                                1,
                                opening_index,
                                opening,
                                "B",
                                "A",
                                engines,
                                self.args.nodes,
                                self.args.main_time_ms,
                                self.args.byoyomi_ms,
                                self.args.increment_ms,
                                self.args.move_timeout,
                                self.args.clock_grace_ms,
                                self.args.max_plies,
                            ),
                        ]
                        record = {
                            "pair_index": pair_index,
                            "opening_index": opening_index,
                            "opening": opening.source_line,
                            "worker": worker_index,
                            "gpu_devices": group,
                            "attempt": attempt,
                            "games": games,
                            "completed_utc": utc_now(),
                        }
                        break
                    except Exception as exc:
                        last_error = exc
                        for engine in engines.values():
                            engine.close()
                        engines = {}
                if record is None:
                    record = {
                        "pair_index": pair_index,
                        "opening_index": opening_index,
                        "opening": opening.source_line,
                        "worker": worker_index,
                        "gpu_devices": group,
                        "error": repr(last_error),
                        "completed_utc": utc_now(),
                    }
                self.save_record(record)
                self.work.task_done()
        finally:
            for engine in engines.values():
                engine.close()

    def start_engines(
        self, worker_index: int, environment: dict[str, str]
    ) -> dict[str, UsiEngine]:
        engines = {
            "A": UsiEngine(
                "A",
                self.args.engine_a,
                self.args.cwd_a,
                self.options_a,
                environment,
                self.run_dir / "logs" / f"worker-{worker_index}-A.log",
                self.args.startup_timeout,
                self.args.protocol_log,
            ),
            "B": UsiEngine(
                "B",
                self.args.engine_b,
                self.args.cwd_b,
                self.options_b,
                environment,
                self.run_dir / "logs" / f"worker-{worker_index}-B.log",
                self.args.startup_timeout,
                self.args.protocol_log,
            ),
        }
        try:
            engines["A"].start()
            engines["B"].start()
        except Exception:
            for engine in engines.values():
                engine.close()
            raise
        return engines


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def default_output() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("strength-runs") / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-a", required=True, help="baseline command")
    parser.add_argument("--engine-b", required=True, help="candidate command")
    parser.add_argument("--cwd-a", type=Path)
    parser.add_argument("--cwd-b", type=Path)
    parser.add_argument(
        "--option-a", action="append", default=[], metavar="NAME=VALUE"
    )
    parser.add_argument(
        "--option-b", action="append", default=[], metavar="NAME=VALUE"
    )
    parser.add_argument("--openings", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--main-time-ms", type=int)
    parser.add_argument("--byoyomi-ms", type=int)
    parser.add_argument("--increment-ms", type=int, default=0)
    parser.add_argument(
        "--clock-grace-ms",
        type=int,
        default=1000,
        help="wall-clock allowance before a time forfeit (default: 1000)",
    )
    parser.add_argument(
        "--gpu-devices",
        help="comma-separated physical devices; default auto, 'none' for CPU",
    )
    parser.add_argument("--gpus-per-worker", type=int, default=1)
    parser.add_argument(
        "--workers", type=int, help="parallel pairs; default one per GPU group"
    )
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--move-timeout", type=float, default=300.0)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--protocol-log",
        action="store_true",
        help="include all USI stdin/stdout in worker logs (can be large)",
    )
    args = parser.parse_args()
    if args.pairs <= 0:
        parser.error("--pairs must be positive")
    if args.nodes is not None and args.nodes <= 0:
        parser.error("--nodes must be positive")
    for name in (
        "main_time_ms",
        "byoyomi_ms",
        "increment_ms",
        "clock_grace_ms",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    clock_requested = (
        args.main_time_ms is not None
        or args.byoyomi_ms is not None
        or args.increment_ms > 0
    )
    if args.nodes is not None and clock_requested:
        parser.error("--nodes cannot be combined with clock options")
    if args.nodes is None and not clock_requested:
        args.nodes = 100_000
    if args.nodes is None:
        args.main_time_ms = args.main_time_ms or 0
        args.byoyomi_ms = args.byoyomi_ms or 0
        if (
            args.main_time_ms == 0
            and args.byoyomi_ms == 0
            and args.increment_ms == 0
        ):
            parser.error("time control must allocate a positive amount of time")
    else:
        args.main_time_ms = 0
        args.byoyomi_ms = 0
    return args


def main() -> int:
    return MatchRun(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
