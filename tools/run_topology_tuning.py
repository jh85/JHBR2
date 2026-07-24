#!/usr/bin/env python3
"""Unattended GPU-topology screening for JHBR2.

The driver performs three stages:

1. Sweep WorkersPerGpu at the incumbent MinibatchSize.
2. Sweep MinibatchSize at the best-throughput worker count, then benchmark a
   small interaction grid made from the best worker and batch values.
3. Run a color-reversed paired strength match between the incumbent and the
   highest-throughput non-incumbent configuration.

NPS is used only to choose a candidate for the paired match. The final
``summary.json`` is the evidence for strength; this script does not
automatically install the candidate as a new default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


NPS_MEDIAN_RE = re.compile(r"^\s*median\s*:\s*([0-9,]+)\s*$")
NPS_MEAN_RE = re.compile(r"^\s*mean\s*:\s*([0-9,]+)\s*$")
CONTROLLED_OPTIONS = {
    "onnxmodel",
    "workerspergpu",
    "threads",
    "minibatchsize",
    "numgpus",
    "nncachesize",
    "leafmatemode",
    "leafmatedepth",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_positive_csv(value: str, option: str) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for field in value.split(","):
        try:
            number = int(field.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{option} must be a comma-separated integer list"
            ) from exc
        if number <= 0:
            raise argparse.ArgumentTypeError(
                f"{option} values must be positive"
            )
        if number not in seen:
            result.append(number)
            seen.add(number)
    if not result:
        raise argparse.ArgumentTypeError(f"{option} cannot be empty")
    return result


def parse_engine_options(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--engine-option must be NAME=VALUE: {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"--engine-option has an empty name: {item!r}")
        if name.casefold() in CONTROLLED_OPTIONS:
            raise SystemExit(
                f"{name} is controlled by the topology driver and cannot be "
                "passed through --engine-option"
            )
        result[name] = value
    return result


def detect_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        fields = [
            field.strip()
            for field in visible.split(",")
            if field.strip() and field.strip() != "-1"
        ]
        if fields:
            return len(fields)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
        )
        devices = [line for line in result.stdout.splitlines() if line.strip()]
        if result.returncode == 0 and devices:
            return len(devices)
    return 1


def parse_benchmark_log(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    in_nps = False
    mean: int | None = None
    median: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() == "NPS:":
            in_nps = True
            continue
        if not in_nps:
            continue
        mean_match = NPS_MEAN_RE.match(line)
        if mean_match:
            mean = int(mean_match.group(1).replace(",", ""))
            continue
        median_match = NPS_MEDIAN_RE.match(line)
        if median_match:
            median = int(median_match.group(1).replace(",", ""))
            break
    if mean is None or median is None:
        return None
    return {"mean_nps": mean, "median_nps": median}


def run_tee(command: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + shlex.join(command), flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("# " + shlex.join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return process.wait()


def ranking(
    results: dict[tuple[int, int], dict[str, Any]],
    keys: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for key in keys:
        result = results.get(key)
        if result and result.get("status") == "ok":
            unique[key] = result
    return sorted(
        unique.values(),
        key=lambda item: (
            -int(item["median_nps"]),
            -int(item["mean_nps"]),
            int(item["workers_per_gpu"]),
            int(item["minibatch_size"]),
        ),
    )


class TopologyTuning:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo = Path(__file__).resolve().parent.parent
        self.benchmark_script = self.repo / "tools" / "benchmark.py"
        self.strength_wrapper = self.repo / "tools" / "run_strength_test.sh"
        self.engine = args.engine.expanduser().resolve()
        self.model = args.model.expanduser().resolve()
        self.openings = args.openings.expanduser().resolve()
        self.output = args.output.expanduser().resolve()
        self.benchmark_dir = self.output / "benchmarks"
        self.results: dict[tuple[int, int], dict[str, Any]] = {}
        self.extra_options = parse_engine_options(args.engine_option)

        for path, label in (
            (self.engine, "engine"),
            (self.model, "model"),
            (self.openings, "openings"),
            (self.benchmark_script, "benchmark script"),
            (self.strength_wrapper, "strength-test wrapper"),
        ):
            if not path.is_file():
                raise SystemExit(f"{label} does not exist: {path}")

        self.worker_values = list(args.worker_values)
        self.batch_values = list(args.minibatch_values)
        if args.baseline_workers not in self.worker_values:
            self.worker_values.append(args.baseline_workers)
        if args.baseline_minibatch not in self.batch_values:
            self.batch_values.append(args.baseline_minibatch)

        self.config = {
            "schema": 1,
            "created_utc": utc_now(),
            "engine": str(self.engine),
            "engine_sha256": sha256_file(self.engine),
            "model": str(self.model),
            "model_sha256": sha256_file(self.model),
            "openings": str(self.openings),
            "openings_sha256": sha256_file(self.openings),
            "gpus": args.gpus,
            "gpu_devices": args.gpu_devices,
            "worker_values": self.worker_values,
            "minibatch_values": self.batch_values,
            "baseline_workers": args.baseline_workers,
            "baseline_minibatch": args.baseline_minibatch,
            "nn_cache_size": args.nn_cache_size,
            "engine_options": self.extra_options,
            "benchmark_byoyomi_ms": args.benchmark_byoyomi_ms,
            "benchmark_positions": args.benchmark_positions,
            "interaction_top_k": args.interaction_top_k,
            "leaf_mate_mode": args.leaf_mate_mode,
            "leaf_mate_depth": args.leaf_mate_depth,
            "strength_pairs": args.strength_pairs,
            "strength_byoyomi_ms": args.strength_byoyomi_ms,
            "strength_gpus_per_worker": args.strength_gpus_per_worker,
            "strength_max_plies": args.strength_max_plies,
            "strength_seed": args.strength_seed,
            "skip_strength": args.skip_strength,
        }

    def prepare(self) -> None:
        config_path = self.output / "config.json"
        if self.output.exists() and not self.args.resume:
            raise SystemExit(
                f"{self.output} already exists; use --resume or another --output"
            )
        self.output.mkdir(parents=True, exist_ok=True)
        if self.args.resume:
            if not config_path.is_file():
                raise SystemExit(f"cannot resume without {config_path}")
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            comparable = dict(previous)
            comparable["created_utc"] = self.config["created_utc"]
            if comparable != self.config:
                raise SystemExit(
                    "the requested topology run differs from the saved config"
                )
        else:
            atomic_write_json(config_path, self.config)

    def benchmark(self, workers: int, batch: int) -> dict[str, Any] | None:
        key = (workers, batch)
        if key in self.results:
            return self.results[key]
        log_path = self.benchmark_dir / f"w{workers}-b{batch}.log"
        parsed = parse_benchmark_log(log_path) if self.args.resume else None
        if parsed is not None:
            print(
                f"Reusing benchmark w={workers} b={batch}: "
                f"median NPS {parsed['median_nps']:,}",
                flush=True,
            )
            result = {
                "status": "ok",
                "workers_per_gpu": workers,
                "minibatch_size": batch,
                "log": str(log_path),
                **parsed,
            }
            self.results[key] = result
            return result

        options = {"NNCacheSize": str(self.args.nn_cache_size)}
        options.update(self.extra_options)
        benchmark_options = ",".join(
            f"{name}:{value}" for name, value in options.items()
        )
        command = [
            sys.executable,
            str(self.benchmark_script),
            str(self.engine),
            str(self.model),
            "--workers-per-gpu",
            str(workers),
            "--gpus",
            str(self.args.gpus),
            "--minibatch",
            str(batch),
            "--byoyomi",
            str(self.args.benchmark_byoyomi_ms),
            "--limit",
            str(self.args.benchmark_positions),
            "--leaf-mate-mode",
            self.args.leaf_mate_mode,
            "--leaf-mate-depth",
            str(self.args.leaf_mate_depth),
            "--options",
            benchmark_options,
        ]
        return_code = run_tee(command, log_path, self.repo)
        parsed = parse_benchmark_log(log_path)
        if return_code != 0 or parsed is None:
            print(
                f"WARNING: benchmark failed for workers={workers}, batch={batch}; "
                f"see {log_path}",
                file=sys.stderr,
                flush=True,
            )
            result = {
                "status": "failed",
                "return_code": return_code,
                "workers_per_gpu": workers,
                "minibatch_size": batch,
                "log": str(log_path),
            }
            self.results[key] = result
            self.write_benchmark_summary()
            return None

        result = {
            "status": "ok",
            "workers_per_gpu": workers,
            "minibatch_size": batch,
            "log": str(log_path),
            **parsed,
        }
        self.results[key] = result
        self.write_benchmark_summary()
        return result

    def write_benchmark_summary(self) -> None:
        ordered = sorted(
            self.results.values(),
            key=lambda item: (
                int(item["workers_per_gpu"]),
                int(item["minibatch_size"]),
            ),
        )
        atomic_write_json(
            self.output / "benchmark_summary.json",
            {"updated_utc": utc_now(), "results": ordered},
        )

    def run_strength(
        self, candidate: dict[str, Any]
    ) -> dict[str, Any] | None:
        if self.args.skip_strength:
            print("Skipping paired strength match (--skip-strength).", flush=True)
            return None

        candidate_workers = int(candidate["workers_per_gpu"])
        candidate_batch = int(candidate["minibatch_size"])
        strength_dir = (
            self.output
            / "strength"
            / f"baseline-vs-w{candidate_workers}-b{candidate_batch}"
        )
        summary_path = strength_dir / "summary.json"
        if self.args.resume and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("completed_pairs", 0) >= self.args.strength_pairs:
                print(f"Reusing completed strength match: {strength_dir}", flush=True)
                return summary

        # LeafMateDepth must precede LeafMateMode: setting mode=off last keeps
        # leaf mate search disabled instead of re-enabling it via the depth.
        common_options = {
            "NNCacheSize": str(self.args.nn_cache_size),
            "LeafMateDepth": str(self.args.leaf_mate_depth),
            "LeafMateMode": self.args.leaf_mate_mode,
        }
        common_options.update(self.extra_options)
        command = [
            str(self.strength_wrapper),
            "match",
            "--engine-a",
            str(self.engine),
            "--engine-b",
            str(self.engine),
            "--openings",
            str(self.openings),
            "--pairs",
            str(self.args.strength_pairs),
            "--byoyomi-ms",
            str(self.args.strength_byoyomi_ms),
            "--max-plies",
            str(self.args.strength_max_plies),
            "--seed",
            str(self.args.strength_seed),
            "--option-a",
            f"OnnxModel={self.model}",
            "--option-b",
            f"OnnxModel={self.model}",
            "--option-a",
            f"WorkersPerGpu={self.args.baseline_workers}",
            "--option-a",
            f"MinibatchSize={self.args.baseline_minibatch}",
            "--option-b",
            f"WorkersPerGpu={candidate_workers}",
            "--option-b",
            f"MinibatchSize={candidate_batch}",
            "--gpus-per-worker",
            str(self.args.strength_gpus_per_worker),
            "--output",
            str(strength_dir),
        ]
        if self.args.gpu_devices:
            command.extend(["--gpu-devices", self.args.gpu_devices])
        for name, value in common_options.items():
            command.extend(["--option-a", f"{name}={value}"])
            command.extend(["--option-b", f"{name}={value}"])
        if strength_dir.exists():
            command.append("--resume")

        return_code = run_tee(
            command, self.output / "strength-driver.log", self.repo
        )
        if return_code != 0 or not summary_path.is_file():
            raise SystemExit(
                f"paired strength match failed; see "
                f"{self.output / 'strength-driver.log'}"
            )
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def run(self) -> int:
        self.prepare()
        baseline = (
            self.args.baseline_workers,
            self.args.baseline_minibatch,
        )

        print("\nStage 1: WorkersPerGpu sweep", flush=True)
        stage1_keys: list[tuple[int, int]] = []
        for workers in self.worker_values:
            key = (workers, self.args.baseline_minibatch)
            stage1_keys.append(key)
            self.benchmark(*key)
        stage1 = ranking(self.results, stage1_keys)
        if not stage1:
            raise SystemExit("all worker benchmarks failed")
        best_workers = int(stage1[0]["workers_per_gpu"])
        print(
            f"Stage 1 leader: workers={best_workers}, "
            f"median NPS={stage1[0]['median_nps']:,}",
            flush=True,
        )

        print("\nStage 2: MinibatchSize sweep", flush=True)
        stage2_keys: list[tuple[int, int]] = []
        for batch in self.batch_values:
            key = (best_workers, batch)
            stage2_keys.append(key)
            self.benchmark(*key)
        stage2 = ranking(self.results, stage2_keys)
        if not stage2:
            raise SystemExit("all minibatch benchmarks failed")
        print(
            f"Stage 2 leader: workers={stage2[0]['workers_per_gpu']}, "
            f"batch={stage2[0]['minibatch_size']}, "
            f"median NPS={stage2[0]['median_nps']:,}",
            flush=True,
        )

        print("\nStage 2b: top-value interaction grid", flush=True)
        top_k = self.args.interaction_top_k
        top_workers = [
            int(item["workers_per_gpu"]) for item in stage1[:top_k]
        ]
        top_batches = [
            int(item["minibatch_size"]) for item in stage2[:top_k]
        ]
        if self.args.baseline_workers not in top_workers:
            top_workers.append(self.args.baseline_workers)
        if self.args.baseline_minibatch not in top_batches:
            top_batches.append(self.args.baseline_minibatch)
        interaction_keys: list[tuple[int, int]] = []
        for workers in top_workers:
            for batch in top_batches:
                key = (workers, batch)
                interaction_keys.append(key)
                self.benchmark(*key)
        interactions = ranking(self.results, interaction_keys)
        non_incumbents = [
            item
            for item in interactions
            if (
                int(item["workers_per_gpu"]),
                int(item["minibatch_size"]),
            )
            != baseline
        ]
        if not non_incumbents:
            raise SystemExit("no successful non-incumbent configuration to test")
        candidate = non_incumbents[0]
        print(
            f"Throughput candidate: workers={candidate['workers_per_gpu']}, "
            f"batch={candidate['minibatch_size']}, "
            f"median NPS={candidate['median_nps']:,}",
            flush=True,
        )

        print("\nStage 3: paired strength match", flush=True)
        strength = self.run_strength(candidate)
        final = {
            "completed_utc": utc_now(),
            "baseline": {
                "workers_per_gpu": self.args.baseline_workers,
                "minibatch_size": self.args.baseline_minibatch,
            },
            "throughput_candidate": candidate,
            "stage1_ranking": stage1,
            "stage2_ranking": stage2,
            "interaction_ranking": interactions,
            "strength_summary": strength,
            "interpretation": (
                "NPS selected the candidate only. Use the paired score and "
                "confidence interval before changing production defaults."
            ),
        }
        atomic_write_json(self.output / "result.json", final)

        if strength is not None:
            score_a = float(strength["score"])
            low, high = strength["elo_95ci"]
            print(
                "\nCompleted. A is the configured baseline and B is the "
                "throughput candidate.",
                flush=True,
            )
            print(
                f"Baseline score={score_a * 100:.2f}%, "
                f"candidate score={(1.0 - score_a) * 100:.2f}%, "
                f"baseline Elo 95% CI=[{low}, {high}]",
                flush=True,
            )
        print(f"Results: {self.output / 'result.json'}", flush=True)
        return 0


def default_output() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("topology-runs") / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--openings", required=True, type=Path)
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="NumGPUs for throughput benchmarks (default: auto-detect)",
    )
    parser.add_argument(
        "--gpu-devices",
        help="comma-separated devices for strength tests (default: auto)",
    )
    parser.add_argument(
        "--worker-values",
        default="1,2,4,8,12,16,24,32",
        help="comma-separated WorkersPerGpu sweep",
    )
    parser.add_argument(
        "--minibatch-values",
        default="32,64,96,128,192,256",
        help="comma-separated MinibatchSize sweep",
    )
    parser.add_argument("--baseline-workers", type=int, default=16)
    parser.add_argument("--baseline-minibatch", type=int, default=256)
    parser.add_argument("--nn-cache-size", type=int, default=1_000_000)
    parser.add_argument(
        "--engine-option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="common option for every benchmark and both match engines",
    )
    parser.add_argument("--benchmark-byoyomi-ms", type=int, default=2000)
    parser.add_argument("--benchmark-positions", type=int, default=20)
    parser.add_argument(
        "--interaction-top-k",
        type=int,
        default=3,
        help="top worker and batch values used in the interaction grid",
    )
    parser.add_argument(
        "--leaf-mate-mode", choices=("off", "shallow"), default="shallow"
    )
    parser.add_argument("--leaf-mate-depth", type=int, default=5)
    parser.add_argument("--strength-pairs", type=int, default=80)
    parser.add_argument("--strength-byoyomi-ms", type=int, default=1000)
    parser.add_argument("--strength-gpus-per-worker", type=int, default=1)
    parser.add_argument("--strength-max-plies", type=int, default=512)
    parser.add_argument("--strength-seed", type=int, default=20260724)
    parser.add_argument("--skip-strength", action="store_true")
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        args.worker_values = parse_positive_csv(
            args.worker_values, "--worker-values"
        )
        args.minibatch_values = parse_positive_csv(
            args.minibatch_values, "--minibatch-values"
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.gpus is None:
        args.gpus = detect_gpu_count()
    for name in (
        "gpus",
        "baseline_workers",
        "baseline_minibatch",
        "benchmark_byoyomi_ms",
        "benchmark_positions",
        "interaction_top_k",
        "strength_pairs",
        "strength_byoyomi_ms",
        "strength_gpus_per_worker",
        "strength_max_plies",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.nn_cache_size < 0:
        parser.error("--nn-cache-size must be non-negative")
    if args.baseline_workers > 64 or max(args.worker_values) > 64:
        parser.error("WorkersPerGpu values cannot exceed the engine limit of 64")
    if args.gpus > 8:
        parser.error("NumGPUs cannot exceed the engine limit of 8")
    return args


def main() -> int:
    return TopologyTuning(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
