"""
PSV (Packed Sfen Value) Dataset for PyTorch training.

Reads YaneuraOu's PackedSfenValue binary files directly.
Each 40-byte record: packed_sfen(32) + score(i16) + move(u16) + ply(u16) + result(i8) + pad(1)

Supports:
  - Direct binary reading (no conversion needed)
  - Sharded loading (one file at a time, low memory)
  - Packed sfen → input planes
  - YaneuraOu move encoding → policy index
  - Score → WDL target

Usage:
    dataset = PSVDataset("/path/to/shard.bin", max_positions=10000000)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
"""

import math
import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset

from psv_loader import decode_packed_sfen, PAWN, LANCE, KNIGHT, SILVER, BISHOP, ROOK, GOLD, KING
from psv_loader import PRO_PAWN, PRO_LANCE, PRO_KNIGHT, PRO_SILVER, HORSE, DRAGON
from psv_loader import BLACK, WHITE

from shogi_model_v2 import (make_direction_policy_index, POLICY_SIZE,
                            NUM_DIRECTIONS, NUM_PROMO_DIRECTIONS)


# =====================================================================
# YaneuraOu move decoding
# =====================================================================

# YaneuraOu square encoding: file * 9 + rank (file-major, 0-based)
# USI notation: file is 1-9 (right to left), rank is a-i (top to bottom)

def yaneuraou_sq_to_file_rank(sq):
    """Convert YaneuraOu square (0-80) to (file_0based, rank_0based)."""
    file = sq // 9
    rank = sq % 9
    return file, rank

def yaneuraou_sq_to_usi(sq):
    """Convert YaneuraOu square to USI string like '7g'."""
    f, r = yaneuraou_sq_to_file_rank(sq)
    return str(f + 1) + chr(ord('a') + r)


# YaneuraOu move encoding (16 bits):
#   bits 0-6:  to square (0-80)
#   bits 7-13: from square (0-80) for board moves, or piece_type for drops
#   bit 14:    promote flag
#   bit 15:    drop flag (when from field = piece type + special flag)
#
# Actually YaneuraOu uses a different encoding:
#   Move = to | (from << 7) | flags
#   For drops: from = piece_type + SQ_NB (81)
#   Promote flag at bit 14

MOVE_DROP_FLAG = 81  # from >= 81 means drop

def decode_yaneuraou_move(move_raw, turn):
    """
    Decode YaneuraOu 16-bit move to USI string.
    turn: 0=BLACK(sente), 1=WHITE(gote)
    Returns USI move string like '7g7f', '7g7f+', 'P*5e'
    """
    if move_raw == 0:
        return None

    to_sq = move_raw & 0x7F
    from_raw = (move_raw >> 7) & 0x7F
    promote = (move_raw >> 14) & 1

    if from_raw >= MOVE_DROP_FLAG:
        # Drop move
        piece_type = from_raw - MOVE_DROP_FLAG
        # YaneuraOu piece types: 1=PAWN, 2=LANCE, 3=KNIGHT, 4=SILVER, 5=BISHOP, 6=ROOK, 7=GOLD
        PT_TO_CHAR = {1: 'P', 2: 'L', 3: 'N', 4: 'S', 5: 'B', 6: 'R', 7: 'G'}
        pt_char = PT_TO_CHAR.get(piece_type, '?')
        to_usi = yaneuraou_sq_to_usi(to_sq)
        return f"{pt_char}*{to_usi}"
    else:
        # Board move
        from_usi = yaneuraou_sq_to_usi(from_raw)
        to_usi = yaneuraou_sq_to_usi(to_sq)
        promo_str = "+" if promote else ""
        return f"{from_usi}{to_usi}{promo_str}"


# =====================================================================
# Packed sfen → input planes (same encoding as sfen_to_planes in shogi_train.py)
# =====================================================================

# Piece type to plane index
PIECE_TO_PLANE = {
    PAWN: 0, LANCE: 1, KNIGHT: 2, SILVER: 3, BISHOP: 4, ROOK: 5, GOLD: 6, KING: 7,
    PRO_PAWN: 8, PRO_LANCE: 9, PRO_KNIGHT: 10, PRO_SILVER: 11, HORSE: 12, DRAGON: 13,
}

# Hand piece types to hand plane index
HAND_PT_TO_PLANE = {'P': 0, 'L': 1, 'N': 2, 'S': 3, 'B': 4, 'R': 5, 'G': 6}


def packed_sfen_to_planes(sfen_bytes):
    """
    Convert 32-byte packed sfen directly to (48, 9, 9) input planes.
    Returns (planes, turn, board, hands) or None on error.
    """
    try:
        board, hands, turn = decode_packed_sfen(sfen_bytes)
    except Exception:
        return None

    flip = (turn == WHITE)
    planes = np.zeros((48, 9, 9), dtype=np.float32)

    # Board pieces
    for sq in range(81):
        piece_val = board[sq]
        if piece_val == 0:
            continue

        color = WHITE if piece_val >= 16 else BLACK
        piece_type = piece_val & 15
        plane_idx = PIECE_TO_PLANE.get(piece_type)
        if plane_idx is None:
            continue

        # YaneuraOu square: file * 9 + rank
        file = sq // 9
        rank = sq % 9

        if flip:
            is_ours = (color == WHITE)
            file, rank = 8 - file, 8 - rank
        else:
            is_ours = (color == BLACK)

        offset = 0 if is_ours else 14
        planes[offset + plane_idx, rank, file] = 1.0

    # Plane 28: repetition (not available from PSV)
    # planes[28] = 0

    # Hand pieces (planes 29-35: ours, 36-42: theirs)
    for color_idx, color_name in enumerate([BLACK, WHITE]):
        for pt_char, pt_plane in HAND_PT_TO_PLANE.items():
            count = hands[color_idx].get(pt_char, 0)
            if count > 0:
                if flip:
                    is_ours = (color_name == WHITE)
                else:
                    is_ours = (color_name == BLACK)
                base = 29 if is_ours else 36
                planes[base + pt_plane, :, :] = count / 18.0  # Normalize

    # Aux planes
    planes[43, :, :] = 1.0  # All ones
    # planes[44-47]: entering king info (skip for PSV)

    return planes, turn


# =====================================================================
# PSV Dataset
# =====================================================================

RECORD_SIZE = 40

class PSVDataset(Dataset):
    """
    PyTorch Dataset that reads PSV binary files directly.
    Loads one shard at a time for memory efficiency.
    """

    def __init__(self, psv_path, max_positions=None, eval_coef=600.0):
        self.psv_path = psv_path
        self.eval_coef = eval_coef

        file_size = os.path.getsize(psv_path)
        total_records = file_size // RECORD_SIZE
        if max_positions:
            total_records = min(total_records, max_positions)
        self.num_records = total_records

        # Read all records into memory (40 bytes × N)
        # For 500M records: 20GB — too much. Use memory-mapped file instead.
        self.data = np.memmap(psv_path, dtype=np.uint8, mode='r',
                              shape=(self.num_records, RECORD_SIZE))

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        rec = bytes(self.data[idx])

        sfen_bytes = rec[:32]
        score = struct.unpack('<h', rec[32:34])[0]
        move_raw = struct.unpack('<H', rec[34:36])[0]
        game_ply = struct.unpack('<H', rec[36:38])[0]
        game_result = struct.unpack('<b', rec[38:39])[0]

        # Decode sfen to planes
        result = packed_sfen_to_planes(sfen_bytes)
        if result is None:
            # Return dummy data on decode error
            return (torch.zeros(48, 9, 9), torch.tensor(-1, dtype=torch.long),
                    torch.tensor([0.0, 1.0, 0.0]))

        planes, turn = result
        flip = (turn == WHITE)

        # Policy target: decode YaneuraOu move → USI → policy index
        policy_idx = -1
        if move_raw != 0:
            usi_move = decode_yaneuraou_move(move_raw, turn)
            if usi_move:
                policy_idx = make_direction_policy_index(usi_move, flip)

        # Value target: WDL from score + game result blend
        win_rate = 1.0 / (1.0 + math.exp(-score / self.eval_coef))

        # Game result: 1=win, 0=draw, -1=loss (from side-to-move perspective)
        if game_result == 1:
            hard = [1.0, 0.0, 0.0]
        elif game_result == 0:
            hard = [0.0, 1.0, 0.0]
        else:
            hard = [0.0, 0.0, 1.0]

        # Blend: 70% score-based, 30% game result
        soft = [win_rate, 0.0, 1.0 - win_rate]
        wdl = [0.7 * s + 0.3 * h for s, h in zip(soft, hard)]

        return (torch.tensor(planes),
                torch.tensor(policy_idx, dtype=torch.long),
                torch.tensor(wdl, dtype=torch.float32))


class PSVShardedDataset(Dataset):
    """
    Memory-efficient PSV dataset that loads one shard file at a time.
    Supports multiple .bin files in a directory.
    """

    def __init__(self, psv_dir, max_per_shard=None, eval_coef=600.0):
        self.eval_coef = eval_coef
        self.max_per_shard = max_per_shard

        # Find all .bin files
        self.shard_paths = sorted([
            os.path.join(psv_dir, f)
            for f in os.listdir(psv_dir)
            if f.endswith('.bin')
        ])
        self.num_shards = len(self.shard_paths)

        if self.num_shards == 0:
            raise FileNotFoundError(f"No .bin files in {psv_dir}")

        # Load first shard to report stats
        self.current_shard = -1
        self.dataset = None
        self.load_shard(0)

        total = sum(os.path.getsize(p) // RECORD_SIZE for p in self.shard_paths)
        print(f"PSV: {self.num_shards} shards, ~{len(self.dataset):,} positions/shard")
        print(f"Total: ~{total:,} ({total/1e9:.1f}B) positions")

    def load_shard(self, shard_id):
        if shard_id == self.current_shard:
            return
        self.current_shard = shard_id
        self.dataset = PSVDataset(self.shard_paths[shard_id],
                                  max_positions=self.max_per_shard,
                                  eval_coef=self.eval_coef)

    def shard_order(self, shuffle=True):
        order = list(range(self.num_shards))
        if shuffle:
            import random
            random.shuffle(order)
        return order

    def __len__(self):
        return len(self.dataset) if self.dataset else 0

    def __getitem__(self, idx):
        return self.dataset[idx]


# =====================================================================
# Test
# =====================================================================

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt2/shogi_data/shogi_hao_depth9/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=000.bin"

    print(f"Testing PSVDataset: {path}")
    ds = PSVDataset(path, max_positions=10)
    for i in range(min(5, len(ds))):
        planes, policy_idx, wdl = ds[i]
        print(f"  [{i}] planes={planes.shape}, policy={policy_idx.item()}, "
              f"wdl=[{wdl[0]:.3f},{wdl[1]:.3f},{wdl[2]:.3f}]")
    print("OK!")
