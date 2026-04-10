#!/usr/bin/env python3
"""
cshogi move generation benchmark — same positions as bench_movegen.cc

Measures:
1. Raw movegen speed: legal_moves generation per second
2. Perft: recursive move count
"""

import time
import cshogi


def perft(board, depth):
    if depth == 0:
        return 1
    moves = list(board.legal_moves)
    if depth == 1:
        return len(moves)
    nodes = 0
    for m in moves:
        board.push(m)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


def bench_raw_movegen(sfens, repeats):
    boards = [cshogi.Board(s) for s in sfens]

    # Warmup
    for b in boards:
        _ = list(b.legal_moves)

    total_calls = 0
    total_moves = 0

    t0 = time.perf_counter()
    for _ in range(repeats):
        for b in boards:
            moves = list(b.legal_moves)
            total_moves += len(moves)
            total_calls += 1
    t1 = time.perf_counter()

    secs = t1 - t0
    print("Raw movegen:")
    print(f"  {total_calls} calls in {secs:.3f} sec")
    print(f"  {total_calls / secs:.0f} calls/sec")
    print(f"  {total_moves / secs:.0f} moves/sec")
    print(f"  Avg {total_moves / total_calls:.1f} legal moves/position")


def bench_perft(sfen, label, depth):
    board = cshogi.Board(sfen)
    print(f"Perft {label} (depth {depth}):")
    print(f"  SFEN: {sfen}")

    t0 = time.perf_counter()
    nodes = perft(board, depth)
    t1 = time.perf_counter()

    secs = t1 - t0
    print(f"  Nodes: {nodes}")
    print(f"  Time:  {secs:.3f} sec")
    print(f"  Speed: {nodes / secs:.0f} nodes/sec")


def main():
    startpos = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    midgame1 = "ln1gk2nl/1r2g2b1/p1sppsppp/2p3p2/1p7/2P1P4/PP1PSP1PP/1BG4R1/LN2KG1NL b - 1"
    midgame2 = "l3k2nl/4g2b1/p1sppsppp/2p3p2/1p7/2P1P4/PP1PSP1PP/1BG4R1/LN2KG1NL b RNPrnp 1"
    endgame = "3g1k3/5+P3/4p1+Spp/p4N3/6p2/1P1P5/P3+b1P1P/2+r6/K1S3GNL w RBG2SN4Pl2p 1"

    all_positions = [startpos, midgame1, midgame2, endgame]

    print("============================")
    print("cshogi Move Generation Bench")
    print("============================\n")

    bench_raw_movegen(all_positions, 100000)

    print()
    bench_perft(startpos, "startpos", 4)
    print()
    bench_perft(startpos, "startpos", 5)
    print()
    bench_perft(midgame2, "midgame+drops", 4)


if __name__ == "__main__":
    main()
