#!/usr/bin/env python3
"""
Generate test positions with legal moves using cshogi as ground truth.

Output format (one test case per line):
  SFEN<TAB>move1 move2 move3 ...

Usage:
  python3 test/generate_positions.py > test/positions.txt
"""

import cshogi
import random
import sys

random.seed(42)


def legal_moves_usi(board):
    """Return sorted list of USI move strings for all legal moves."""
    return sorted(cshogi.move_to_usi(m) for m in board.legal_moves)


def emit(board, out, seen):
    """Emit one test case if this position hasn't been seen yet."""
    sfen = board.sfen()
    # Use board part + side + hand as dedup key (ignore move number)
    parts = sfen.split()
    key = " ".join(parts[:3])
    if key in seen:
        return
    seen.add(key)
    moves = legal_moves_usi(board)
    out.write(f"{sfen}\t{' '.join(moves)}\n")


def gen_startpos(out, seen):
    """Starting position."""
    board = cshogi.Board()
    emit(board, out, seen)


def gen_random_games(out, seen, num_games=200, max_ply=200):
    """Play random games, emitting every position encountered."""
    for _ in range(num_games):
        board = cshogi.Board()
        for _ in range(max_ply):
            emit(board, out, seen)
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
            if board.is_game_over():
                emit(board, out, seen)
                break


def gen_edge_case_positions(out, seen):
    """Hand-crafted SFENs targeting tricky rules."""
    edge_cases = [
        # Pawn on rank b for BLACK: must promote if moving forward
        "9/9/9/9/9/9/9/4P4/4K4 b - 1",
        # Pawn on rank a: already at last rank (shouldn't appear, but test)
        # Lance on rank b for BLACK: must promote
        "9/4L4/9/9/9/9/9/9/4K4 b - 1",
        # Knight on rank c for BLACK: must promote if jumping to rank a
        "9/9/4N4/9/9/9/9/9/4K4 b - 1",
        # Knight on rank b: can't exist without promotion, but test board setup
        # WHITE pawn on rank h: must promote if moving forward
        "4k4/9/9/9/9/9/9/4p4/9 w - 1",
        # Pieces in hand: pawn drop restrictions
        # BLACK has pawn in hand, column 5 has own pawn -> nifu
        "4k4/9/9/9/9/9/4P4/9/4K4 b P 1",
        # Pawn drop checkmate (uchifuzume): BLACK has pawn, can drop to give mate
        # but if it's immediate checkmate it's illegal
        "4k4/9/9/9/9/9/9/9/4K4 b P 1",
        # King in check — must escape
        "4k4/4r4/9/9/9/9/9/9/4K4 b - 1",
        # Double check — only king move works
        "4k4/9/9/9/9/9/9/3r1b3/4K4 b - 1",
        # Pinned piece — silver pinned by rook
        "4k4/9/9/9/4r4/9/4S4/9/4K4 b - 1",
        # Promotion zone: piece moving INTO zone can promote
        "4k4/9/9/4S4/9/9/9/9/4K4 b - 1",
        # Promotion zone: piece moving OUT of zone can promote
        "4k4/4S4/9/9/9/9/9/9/4K4 b - 1",
        # Promotion zone: piece moving WITHIN zone can promote
        "4k4/9/4S4/9/9/9/9/9/4K4 b - 1",
        # Horse (promoted bishop) with step moves
        "4k4/9/9/9/4+B4/9/9/9/4K4 b - 1",
        # Dragon (promoted rook) with step moves
        "4k4/9/9/9/4+R4/9/9/9/4K4 b - 1",
        # Full hand: all piece types in hand
        "4k4/9/9/9/9/9/9/9/4K4 b RBGSNLPrbgsnlp 1",
        # Knight drop restriction: can't drop on ranks a,b for BLACK
        "4k4/9/9/9/9/9/9/9/4K4 b N 1",
        # Lance drop restriction: can't drop on rank a for BLACK
        "4k4/9/9/9/9/9/9/9/4K4 b L 1",
        # WHITE knight drop restriction: can't drop on ranks h,i
        "4K4/9/9/9/9/9/9/9/4k4 w n 1",
        # Entering king territory
        "3K5/9/9/9/9/9/9/9/4k4 b RBG2S2N2L9P 1",
        # Complex midgame with many pieces
        "ln1gk2nl/1r2g2b1/p1sppsppp/2p3p2/1p7/2P1P4/PP1PSP1PP/1BG4R1/LN2KG1NL b - 1",
        # Position after some captures — pieces in hand
        "ln1gk2nl/1r4sb1/p1pppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b Pp 1",
        # WHITE to move with drops available
        "lnsgk2nl/1r4gs1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w Bb 1",
        # Tokin (promoted pawn) movement
        "4k4/9/9/9/4+P4/9/9/9/4K4 b - 1",
        # Promoted lance movement
        "4k4/9/9/9/4+L4/9/9/9/4K4 b - 1",
        # Promoted knight movement
        "4k4/9/9/9/4+N4/9/9/9/4K4 b - 1",
        # Promoted silver movement
        "4k4/9/9/9/4+S4/9/9/9/4K4 b - 1",
        # King surrounded — limited escape squares
        "3rkr3/3ggg3/9/9/9/9/9/9/4K4 w - 1",
        # Lots of pieces on board — stress test
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    ]

    for sfen in edge_cases:
        try:
            board = cshogi.Board(sfen)
            emit(board, out, seen)
        except Exception as e:
            print(f"# Warning: skipping invalid SFEN: {sfen} ({e})",
                  file=sys.stderr)


def gen_from_specific_games(out, seen):
    """Play games from specific openings to get diverse positions."""
    openings = [
        # Yagura opening moves
        ["7g7f", "8c8d", "6i7h", "3c3d", "6g6f", "7a6b"],
        # Ranging rook
        ["7g7f", "3c3d", "2g2f", "4c4d", "2f2e", "2b3c", "6i7h", "8b4b"],
        # Static rook
        ["2g2f", "8c8d", "2f2e", "8d8e", "7g7f", "3c3d"],
        # Bishop exchange
        ["7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "2b3c", "8h2b+"],
        # Fourth-file rook
        ["7g7f", "3c3d", "6g6f", "8c8d", "7i6h", "7a6b", "5i4h", "5a4b"],
    ]

    for opening_moves in openings:
        board = cshogi.Board()
        for usi in opening_moves:
            try:
                move = board.move_from_usi(usi)
                if move in board.legal_moves:
                    emit(board, out, seen)
                    board.push(move)
                else:
                    break
            except Exception:
                break
        # Continue with random play from this opening
        for _ in range(100):
            emit(board, out, seen)
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
            if board.is_game_over():
                emit(board, out, seen)
                break


def main():
    out = sys.stdout
    seen = set()

    print("# Generating startpos...", file=sys.stderr)
    gen_startpos(out, seen)

    print("# Generating edge case positions...", file=sys.stderr)
    gen_edge_case_positions(out, seen)

    print("# Generating positions from specific openings...", file=sys.stderr)
    gen_from_specific_games(out, seen)

    print("# Generating random game positions...", file=sys.stderr)
    gen_random_games(out, seen)

    print(f"# Total positions generated: {len(seen)}", file=sys.stderr)


if __name__ == "__main__":
    main()
