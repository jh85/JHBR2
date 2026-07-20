#!/usr/bin/env python3

import argparse
import subprocess

import cshogi


def load_sfens(path):
    sfens = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sfens.append(line.split("\t", 1)[0])
    return sfens


def exhaustive_mate_in_one(board):
    for move in list(board.legal_moves):
        board.push(move)
        mate = board.is_check() and not any(board.legal_moves)
        board.pop()
        if mate:
            return move
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_binary")
    parser.add_argument("positions")
    args = parser.parse_args()

    sfens = load_sfens(args.positions)
    process = subprocess.run(
        [args.test_binary, "--verdicts", args.positions],
        check=True,
        text=True,
        capture_output=True,
    )
    fast_moves = process.stdout.splitlines()
    if len(fast_moves) != len(sfens):
        raise RuntimeError(
            f"verdict count mismatch: {len(fast_moves)} != {len(sfens)}"
        )

    checked = 0
    exact_mates = 0
    upstream_mates = 0
    upstream_false_negatives = 0
    failures = []

    for index, (sfen, fast_usi) in enumerate(zip(sfens, fast_moves), 1):
        board = cshogi.Board(sfen)
        if board.is_check():
            # mate_move_in_1ply has the same non-check precondition as
            # the upstream C++ routine. JHBR2's in-check fallback is
            # covered by the C++ oracle test.
            continue

        checked += 1
        exact = exhaustive_mate_in_one(board)
        upstream = board.mate_move_in_1ply()
        fast_has_mate = fast_usi not in ("0000", "INVALID")
        exact_has_mate = exact != 0
        upstream_has_mate = upstream != 0

        exact_mates += exact_has_mate
        upstream_mates += upstream_has_mate
        if exact_has_mate and not upstream_has_mate:
            upstream_false_negatives += 1

        if fast_has_mate != exact_has_mate:
            failures.append(
                (index, sfen, fast_usi,
                 cshogi.move_to_usi(exact) if exact else "0000")
            )
            continue

        if upstream_has_mate and not fast_has_mate:
            failures.append(
                (index, sfen, fast_usi, cshogi.move_to_usi(upstream))
            )
            continue

        if fast_has_mate:
            fast = board.move_from_usi(fast_usi)
            if fast not in board.legal_moves:
                failures.append((index, sfen, fast_usi, "illegal"))
                continue
            board.push(fast)
            valid = board.is_check() and not any(board.legal_moves)
            board.pop()
            if not valid:
                failures.append((index, sfen, fast_usi, "not mate"))

    print(f"Checked non-check positions: {checked}")
    print(f"Exact mates: {exact_mates}")
    print(f"Upstream cshogi mates: {upstream_mates}")
    print(f"Upstream long-range false negatives: {upstream_false_negatives}")
    print(f"Failures: {len(failures)}")
    for index, sfen, fast, expected in failures[:20]:
        print(f"FAIL [{index}] fast={fast} expected={expected}")
        print(f"  {sfen}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
