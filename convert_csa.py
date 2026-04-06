"""
Convert Floodgate CSA game records to training format.

Parses CSA files, replays each game, and outputs one training record per position:
    sfen <SFEN> bestmove <USI_move> result <W|D|L>

Supports filtering by player rating.

Usage:
    # Convert all games from 2024, minimum rating 3000:
    python convert_csa.py --input /mnt2/shogi_data/floodgate/2024 \
        --output train_floodgate_2024.sfen --min-rating 3000

    # Convert multiple years:
    python convert_csa.py --input /mnt2/shogi_data/floodgate/2023 \
                          --input /mnt2/shogi_data/floodgate/2024 \
        --output train_floodgate.sfen --min-rating 3000
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import cshogi


# CSA piece mapping for move parsing
# CSA move format: +7776FU (sente moves pawn from 77 to 76)
#                  -8384FU (gote moves pawn from 83 to 84)
# CSA drop format: +0055KA (sente drops bishop on 55)

CSA_PIECE_TO_USI = {
    'FU': 'P', 'KY': 'L', 'KE': 'N', 'GI': 'S',
    'KI': 'G', 'KA': 'B', 'HI': 'R', 'OU': 'K',
    'TO': '+P', 'NY': '+L', 'NK': '+N', 'NG': '+S',
    'UM': '+B', 'RY': '+R',
}


def csa_move_to_usi(csa_move):
    """
    Convert CSA move string to USI format.
    CSA: +7776FU or -0055KA (drop)
    USI: 7g7f or B*5e
    """
    if len(csa_move) < 7:
        return None

    sign = csa_move[0]  # '+' or '-'
    from_file = int(csa_move[1])
    from_rank = int(csa_move[2])
    to_file = int(csa_move[3])
    to_rank = int(csa_move[4])
    piece = csa_move[5:7]

    # Drop move (from = 00)
    if from_file == 0 and from_rank == 0:
        # Get base piece for drop
        usi_piece = CSA_PIECE_TO_USI.get(piece, '')
        if not usi_piece or usi_piece.startswith('+'):
            return None  # Can't drop promoted pieces
        to_usi = f"{to_file}{chr(ord('a') + to_rank - 1)}"
        return f"{usi_piece}*{to_usi}"

    # Normal move
    from_usi = f"{from_file}{chr(ord('a') + from_rank - 1)}"
    to_usi = f"{to_file}{chr(ord('a') + to_rank - 1)}"
    return f"{from_usi}{to_usi}"


def parse_csa_game(filepath):
    """
    Parse a CSA game file and return game info.

    Returns:
        dict with:
            'sente_name': str
            'gote_name': str
            'sente_rate': float or None
            'gote_rate': float or None
            'moves': list of CSA move strings
            'result': 'sente_win', 'gote_win', 'draw', or 'unknown'
    """
    game = {
        'sente_name': '',
        'gote_name': '',
        'sente_rate': None,
        'gote_rate': None,
        'moves': [],
        'result': 'unknown',
    }

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Player names
                if line.startswith('N+'):
                    game['sente_name'] = line[2:]
                elif line.startswith('N-'):
                    game['gote_name'] = line[2:]

                # Ratings: 'black_rate:name+hash:RATING or 'white_rate:name+hash:RATING
                elif line.startswith("'black_rate:") or line.startswith("'white_rate:"):
                    match = re.search(r':(\d+\.?\d*)\s*$', line)
                    if match:
                        rate = float(match.group(1))
                        if line.startswith("'black_rate:"):
                            game['sente_rate'] = rate
                        else:
                            game['gote_rate'] = rate

                # Moves
                elif line.startswith('+') or line.startswith('-'):
                    if len(line) >= 7 and line[1:3].isdigit():
                        game['moves'].append(line)

                # Game result
                elif line.startswith('%TORYO'):
                    # The side that resigned loses
                    pass  # Result determined by summary line
                elif line.startswith("'summary:"):
                    # 'summary:toryo:name1 win:name2 lose
                    # or: 'summary:sennichite:name1 draw:name2 draw
                    if 'win' in line:
                        # Find who won
                        parts = line.split(':')
                        for i, p in enumerate(parts):
                            if 'win' in p:
                                winner = p.split()[0]
                                if winner == game['sente_name']:
                                    game['result'] = 'sente_win'
                                else:
                                    game['result'] = 'gote_win'
                                break
                    elif 'draw' in line:
                        game['result'] = 'draw'

    except Exception:
        return None

    return game


def extract_ratings_from_filename(filepath):
    """Try to extract player names from filename for rating lookup."""
    # Filename: wdoor+floodgate-300-10F+player1+player2+timestamp.csa
    basename = os.path.basename(filepath)
    parts = basename.replace('.csa', '').split('+')
    # Typically: wdoor, floodgate-300-10F, player1, player2, timestamp
    if len(parts) >= 5:
        return parts[2], parts[3]
    return None, None


def parse_csa_game_cshogi(filepath):
    """
    Parse CSA game using cshogi for reliable move parsing and SFEN generation.
    Returns list of (sfen, usi_move, result_for_side_to_move) tuples.
    """
    try:
        # First parse the game metadata manually
        game = parse_csa_game(filepath)
        if game is None or game['result'] == 'unknown' or len(game['moves']) < 10:
            return None, game

        # Use cshogi to replay the game
        board = cshogi.Board()
        positions = []

        for csa_move_str in game['moves']:
            sfen = board.sfen()
            side = board.turn  # 0=BLACK(sente), 1=WHITE(gote)

            # Convert CSA move to USI
            usi_move = csa_move_to_usi(csa_move_str)
            if usi_move is None:
                break

            # Determine result from side-to-move's perspective
            if game['result'] == 'sente_win':
                result = 'W' if side == cshogi.BLACK else 'L'
            elif game['result'] == 'gote_win':
                result = 'L' if side == cshogi.BLACK else 'W'
            else:
                result = 'D'

            positions.append((sfen, usi_move, result))

            # Apply move using cshogi
            # cshogi expects USI move — but we need to handle promotion
            # The CSA move tells us the piece AFTER the move, so we can
            # detect promotion by checking if the piece changed type.
            try:
                # Try to find the legal move matching our USI string
                move_found = False
                for legal_move in board.legal_moves:
                    legal_usi = cshogi.move_to_usi(legal_move)
                    # CSA doesn't have '+' suffix for promotion — we need to
                    # check if this move should be a promotion by comparing
                    # the piece type in CSA vs the piece on the from square.
                    if legal_usi == usi_move or legal_usi == usi_move + '+':
                        # Check if CSA indicates promotion
                        piece_csa = csa_move_str[5:7]
                        from_sq_file = int(csa_move_str[1])
                        from_sq_rank = int(csa_move_str[2])

                        is_promoted_piece = piece_csa in ('TO', 'NY', 'NK', 'NG', 'UM', 'RY')

                        if is_promoted_piece and not usi_move.endswith('+'):
                            # The piece became promoted — check if it wasn't already
                            if from_sq_file != 0:  # not a drop
                                from_sq = (9 - from_sq_file) * 9 + (from_sq_rank - 1)
                                piece_on_from = board.piece(from_sq)
                                # If the piece on from-square is NOT promoted but CSA says promoted piece
                                # then this move is a promotion
                                if piece_on_from is not None:
                                    if legal_usi == usi_move + '+':
                                        usi_move = usi_move + '+'
                                        # Update the stored position
                                        positions[-1] = (sfen, usi_move, result)

                        board.push_usi(legal_usi if legal_usi.startswith(usi_move) else usi_move)
                        move_found = True
                        break

                if not move_found:
                    # Try direct push
                    board.push_usi(usi_move)
            except Exception:
                # If move application fails, try with promotion suffix
                try:
                    board.push_usi(usi_move + '+')
                    usi_move = usi_move + '+'
                    positions[-1] = (sfen, usi_move, result)
                except Exception:
                    break  # Can't continue this game

        return positions, game

    except Exception:
        return None, None


def convert_csa_directory(input_dirs, output_path, min_rating=0,
                          limit=None, skip_short=10):
    """
    Convert all CSA files in directories to training format.
    """
    # Collect all CSA files
    csa_files = []
    for input_dir in input_dirs:
        for f in Path(input_dir).glob("*.csa"):
            csa_files.append(str(f))

    print(f"Found {len(csa_files):,} CSA files")
    print(f"Min rating filter: {min_rating}")
    print(f"Output: {output_path}")

    t0 = time.time()
    games_processed = 0
    games_used = 0
    games_skipped_rating = 0
    games_skipped_short = 0
    games_skipped_error = 0
    positions_written = 0

    with open(output_path, 'w') as fout:
        for i, csa_file in enumerate(csa_files):
            if limit and positions_written >= limit:
                break

            # Quick rating check from game metadata
            game = parse_csa_game(csa_file)
            if game is None:
                games_skipped_error += 1
                continue

            # Rating filter: both players must be above threshold
            if min_rating > 0:
                sr = game.get('sente_rate')
                gr = game.get('gote_rate')
                if sr is None or gr is None:
                    games_skipped_rating += 1
                    continue
                if sr < min_rating or gr < min_rating:
                    games_skipped_rating += 1
                    continue

            # Skip short/invalid games
            if game['result'] == 'unknown' or len(game['moves']) < skip_short:
                games_skipped_short += 1
                continue

            # Parse and replay with cshogi
            positions, _ = parse_csa_game_cshogi(csa_file)
            if positions is None or len(positions) < skip_short:
                games_skipped_error += 1
                continue

            games_used += 1

            # Write positions
            for sfen, usi_move, result in positions:
                fout.write(f"sfen {sfen} bestmove {usi_move} result {result}\n")
                positions_written += 1

                if limit and positions_written >= limit:
                    break

            games_processed += 1

            if games_processed % 1000 == 0:
                elapsed = time.time() - t0
                rate = positions_written / max(elapsed, 1)
                print(f"  {games_processed:,} games → {positions_written/1e6:.1f}M positions "
                      f"({rate:.0f} pos/sec) "
                      f"[skip: rate={games_skipped_rating}, short={games_skipped_short}, "
                      f"err={games_skipped_error}]")

    elapsed = time.time() - t0
    print(f"\nDone: {positions_written:,} positions from {games_used:,} games "
          f"in {elapsed:.1f}s")
    print(f"  Games processed: {games_processed:,}")
    print(f"  Skipped (rating): {games_skipped_rating:,}")
    print(f"  Skipped (short/no result): {games_skipped_short:,}")
    print(f"  Skipped (parse error): {games_skipped_error:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Floodgate CSA to training format")
    parser.add_argument("--input", action="append", required=True,
                        help="Input directory with CSA files (can specify multiple)")
    parser.add_argument("--output", required=True, help="Output training file")
    parser.add_argument("--min-rating", type=float, default=0,
                        help="Minimum player rating (both players must meet)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max positions to output")
    parser.add_argument("--skip-short", type=int, default=10,
                        help="Skip games with fewer than N moves")
    args = parser.parse_args()

    convert_csa_directory(
        args.input, args.output,
        min_rating=args.min_rating,
        limit=args.limit,
        skip_short=args.skip_short,
    )
