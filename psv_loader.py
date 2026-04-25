"""
PSV (Packed Sfen Value) data loader for YaneuraOu format.

Decodes PackedSfenValue records (40 bytes each):
  - PackedSfen (32 bytes): Huffman-coded board position
  - score (int16): evaluation from DL Suisho
  - move (uint16): best move (may be 0 if not available)
  - gamePly (uint16): game ply
  - game_result (int8): -1=loss, 0=draw, 1=win
  - padding (uint8)

Huffman tables and decoding logic from:
  YaneuraOu/source/extra/sfen_packer.cpp
"""

import struct
import numpy as np
from typing import Optional

# =====================================================================
# Piece types (YaneuraOu convention)
# =====================================================================

NO_PIECE = 0
PAWN = 1
LANCE = 2
KNIGHT = 3
SILVER = 4
BISHOP = 5
ROOK = 6
GOLD = 7
KING = 8

# Promoted pieces
PRO_PAWN = PAWN + 8      # 9
PRO_LANCE = LANCE + 8    # 10
PRO_KNIGHT = KNIGHT + 8  # 11
PRO_SILVER = SILVER + 8  # 12
HORSE = BISHOP + 8       # 13
DRAGON = ROOK + 8        # 14

BLACK = 0  # Sente
WHITE = 1  # Gote

# =====================================================================
# Huffman table for board pieces
# =====================================================================
# (code, bits) — read LSB first
HUFFMAN_TABLE = {
    # code: (piece_type, bits)
}

# Build decode table: read bits one at a time, match against codes
# NO_PIECE: code=0b0, bits=1
# PAWN:     code=0b01, bits=2
# LANCE:    code=0b0011, bits=4
# KNIGHT:   code=0b1011, bits=4
# SILVER:   code=0b0111, bits=4
# BISHOP:   code=0b011111, bits=6
# ROOK:     code=0b111111, bits=6
# GOLD:     code=0b01111, bits=5

HUFFMAN_PIECES = [
    (0b0,      1, NO_PIECE),
    (0b01,     2, PAWN),
    (0b0011,   4, LANCE),
    (0b1011,   4, KNIGHT),
    (0b0111,   4, SILVER),
    (0b011111, 6, BISHOP),
    (0b111111, 6, ROOK),
    (0b01111,  5, GOLD),
]


# =====================================================================
# BitStream reader
# =====================================================================

class BitStream:
    def __init__(self, data: bytes):
        self.data = data
        self.cursor = 0

    def read_one_bit(self) -> int:
        byte_idx = self.cursor >> 3
        bit_idx = self.cursor & 7
        self.cursor += 1
        return (self.data[byte_idx] >> bit_idx) & 1

    def read_n_bits(self, n: int) -> int:
        val = 0
        for i in range(n):
            val |= self.read_one_bit() << i
        return val


# =====================================================================
# Decode one piece from bitstream using Huffman table
# =====================================================================

def decode_piece(stream: BitStream) -> int:
    """Decode a piece type from the Huffman-coded bitstream."""
    code = 0
    bits = 0
    while bits < 6:
        code |= stream.read_one_bit() << bits
        bits += 1
        for hcode, hbits, piece in HUFFMAN_PIECES:
            if hbits == bits and hcode == code:
                return piece
    return NO_PIECE  # Should not reach here


# =====================================================================
# Decode PackedSfen → board array + hands + turn
# =====================================================================

def decode_packed_sfen(sfen_bytes: bytes):
    """
    Decode 32-byte PackedSfen into board state.

    Returns:
        board: int[81] — piece on each square (0=empty, 1-14=BLACK, 17-30=WHITE)
        hands: [dict, dict] — hand pieces for [BLACK, WHITE]
        turn: 0=BLACK, 1=WHITE

    Piece encoding: color_offset + piece_type
        BLACK pieces: 0 + type (1-14)
        WHITE pieces: 16 + type (1-14)
    """
    stream = BitStream(sfen_bytes)

    # 1. Turn (1 bit)
    turn = stream.read_one_bit()

    # 2. King squares (7 bits each)
    king_sq_black = stream.read_n_bits(7)
    king_sq_white = stream.read_n_bits(7)

    # 3. Board (81 squares, Huffman coded)
    board = [0] * 81
    if king_sq_black < 81:
        board[king_sq_black] = KING  # BLACK king
    if king_sq_white < 81:
        board[king_sq_white] = 16 + KING  # WHITE king

    for sq in range(81):
        if sq == king_sq_black or sq == king_sq_white:
            continue

        piece = decode_piece(stream)
        if piece == NO_PIECE:
            continue

        # Read promotion flag (not for GOLD)
        promoted = False
        if piece != GOLD:
            promoted = stream.read_one_bit() == 1

        # Read color (0=BLACK, 1=WHITE)
        color = stream.read_one_bit()

        if promoted:
            piece += 8  # Promote

        # Encode: BLACK pieces = type, WHITE pieces = 16 + type
        board[sq] = piece + (16 if color == WHITE else 0)

    # 4. Hand pieces (Huffman coded)
    hands = [{}, {}]  # [BLACK, WHITE]
    piece_max_hand = {PAWN: 18, LANCE: 4, KNIGHT: 4, SILVER: 4,
                      BISHOP: 2, ROOK: 2, GOLD: 4}

    while stream.cursor < 256:
        piece = decode_piece(stream)
        if piece == NO_PIECE:
            break

        # Check if enough bits remain for color
        if stream.cursor >= 256:
            break

        color = stream.read_one_bit()

        pt_name = {PAWN: 'P', LANCE: 'L', KNIGHT: 'N', SILVER: 'S',
                   BISHOP: 'B', ROOK: 'R', GOLD: 'G'}.get(piece, '?')
        hands[color][pt_name] = hands[color].get(pt_name, 0) + 1

    return board, hands, turn


# =====================================================================
# Convert decoded board to SFEN string
# =====================================================================

PIECE_CHARS = {
    PAWN: 'P', LANCE: 'L', KNIGHT: 'N', SILVER: 'S',
    BISHOP: 'B', ROOK: 'R', GOLD: 'G', KING: 'K',
    PRO_PAWN: '+P', PRO_LANCE: '+L', PRO_KNIGHT: '+N', PRO_SILVER: '+S',
    HORSE: '+B', DRAGON: '+R',
}


def board_to_sfen(board, hands, turn, game_ply=1):
    """Convert decoded board to SFEN string."""
    ranks = []
    for rank in range(9):
        empty = 0
        rank_str = ""
        for file in range(9):
            # YaneuraOu square = file * 9 + rank (file-major)
            sq = file * 9 + rank
            piece_val = board[sq]
            if piece_val == 0:
                empty += 1
            else:
                if empty > 0:
                    rank_str += str(empty)
                    empty = 0
                color = 1 if piece_val >= 16 else 0
                piece_type = piece_val & 15
                char = PIECE_CHARS.get(piece_type, '?')
                if color == WHITE:
                    char = char.lower()
                rank_str += char
        if empty > 0:
            rank_str += str(empty)
        ranks.append(rank_str)

    board_str = "/".join(ranks)
    turn_str = "b" if turn == BLACK else "w"

    # Hand pieces
    hand_str = ""
    for color in [BLACK, WHITE]:
        for pt in ['R', 'B', 'G', 'S', 'N', 'L', 'P']:
            count = hands[color].get(pt, 0)
            if count > 0:
                ch = pt if color == BLACK else pt.lower()
                if count > 1:
                    hand_str += str(count) + ch
                else:
                    hand_str += ch
    if not hand_str:
        hand_str = "-"

    return f"{board_str} {turn_str} {hand_str} {game_ply}"


# =====================================================================
# Convert score to WDL using sigmoid (Eval_Coef=600)
# =====================================================================

def score_to_wdl(score, eval_coef=600.0):
    """Convert evaluation score to (win, draw, loss) probabilities."""
    # Clamp extreme scores
    score = max(-32000, min(32000, score))

    # Win probability from the side to move's perspective
    win_prob = 1.0 / (1.0 + np.exp(-score / eval_coef))

    # Simple WDL: no explicit draw, just W and L
    return win_prob, 0.0, 1.0 - win_prob


# =====================================================================
# PSV file reader (generator)
# =====================================================================

def read_psv_file(filepath, max_records=None):
    """
    Read PSV file and yield (sfen_bytes, score, move, game_ply, result) tuples.
    """
    RECORD_SIZE = 40
    count = 0
    with open(filepath, 'rb') as f:
        while True:
            rec = f.read(RECORD_SIZE)
            if len(rec) < RECORD_SIZE:
                break
            sfen = rec[:32]
            score = struct.unpack('<h', rec[32:34])[0]
            move = struct.unpack('<H', rec[34:36])[0]
            game_ply = struct.unpack('<H', rec[36:38])[0]
            result = struct.unpack('<b', rec[38:39])[0]
            yield sfen, score, move, game_ply, result
            count += 1
            if max_records and count >= max_records:
                break


# =====================================================================
# Test
# =====================================================================

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt2/shogi_data/Knowledge_distilled_dataset_by_DLSuisho15b/hao_depth_9_shuffled_01.bin"

    print(f"Reading {path}...")
    for i, (sfen, score, move, ply, result) in enumerate(read_psv_file(path, max_records=5)):
        board, hands, turn = decode_packed_sfen(sfen)
        sfen_str = board_to_sfen(board, hands, turn, ply)
        w, d, l = score_to_wdl(score)
        print(f"  [{i}] score={score:6d} result={result:2d} ply={ply:3d} "
              f"WDL=({w:.3f},{d:.3f},{l:.3f})")
        print(f"       SFEN: {sfen_str}")
    print("Done!")
