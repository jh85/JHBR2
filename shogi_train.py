"""
Training script for ShogiBT4.

Supports supervised learning from SFEN game records (kifu).
Training data format: one line per position:
    sfen <SFEN string> moves <move1> <move2> ... result <W|D|L>

Example:
    sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 bestmove 7g7f result W

Usage:
    python shogi_train.py --data train.sfen --epochs 10 --batch 64
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add lc0 src to path for the Shogi C++ modules (via ctypes or subprocess).
# For now we do everything in Python.
from shogi_model import ShogiBT4, ShogiBT4Config, generate_attn_policy_map


# =====================================================================
# Shogi board encoding (pure Python, mirrors C++ encoder)
# =====================================================================

def sfen_to_planes(sfen_str):
    """
    Convert an SFEN string to a (48, 9, 9) input tensor.
    Encodes from side-to-move's perspective (flipped for WHITE).
    """
    parts = sfen_str.strip().split()
    board_str = parts[0]
    side = parts[1]  # 'b' or 'w'
    hand_str = parts[2]
    flip = (side == 'w')

    planes = np.zeros((48, 9, 9), dtype=np.float32)

    # Piece type mapping: char → (color, plane_index)
    # Our pieces = planes 0-13, their pieces = planes 14-27
    PIECE_PLANE = {
        'P': 0, 'L': 1, 'N': 2, 'S': 3, 'B': 4, 'R': 5, 'G': 6, 'K': 7,
    }
    PROMOTED_PLANE = {
        'P': 8, 'L': 9, 'N': 10, 'S': 11, 'B': 12, 'R': 13,
    }

    # Parse board
    r, f = 0, 8  # Start top-left: rank a (idx 0), file 9 (idx 8)
    promoted = False
    for ch in board_str:
        if ch == '/':
            r += 1
            f = 8
            continue
        if ch == '+':
            promoted = True
            continue
        if ch.isdigit():
            f -= int(ch)
            continue

        is_black = ch.isupper()
        base_ch = ch.upper()
        if promoted and base_ch in PROMOTED_PLANE:
            plane_idx = PROMOTED_PLANE[base_ch]
        else:
            plane_idx = PIECE_PLANE.get(base_ch, -1)
        promoted = False
        if plane_idx < 0:
            f -= 1
            continue

        # Determine "ours" vs "theirs"
        if flip:
            is_ours = not is_black
            sq_f, sq_r = 8 - f, 8 - r  # flip 180°
        else:
            is_ours = is_black
            sq_f, sq_r = f, r

        offset = 0 if is_ours else 14
        planes[offset + plane_idx, sq_r, sq_f] = 1.0
        f -= 1

    # Plane 28: repetition (TODO: requires history)
    # planes[28] = 0

    # Hand pieces
    HAND_PLANE = {'P': 0, 'L': 1, 'N': 2, 'S': 3, 'B': 4, 'R': 5, 'G': 6}
    if hand_str != '-':
        count = 0
        for ch in hand_str:
            if ch.isdigit():
                count = count * 10 + int(ch)
                continue
            if count == 0:
                count = 1
            is_black = ch.isupper()
            base = ch.upper()
            idx = HAND_PLANE.get(base, -1)
            if idx >= 0:
                if flip:
                    is_ours = not is_black
                else:
                    is_ours = is_black
                plane = 29 + idx if is_ours else 36 + idx
                planes[plane] = float(count)
            count = 0

    # Plane 43: all ones
    planes[43] = 1.0

    # Planes 44-47: Entering-king (nyugyoku) progress features.
    # Compute points and piece counts for both sides.
    # We need to parse the board to count pieces in enemy camp.
    MAJOR_PIECES = {'B', 'R'}  # bishop, rook (and promoted forms count too)
    for color_is_ours in [True, False]:
        points = 0
        pieces_in_camp = 0

        # Enemy camp: ranks 0,1,2 for "us" (after flip, us=BLACK perspective).
        # For the opponent, enemy camp is ranks 6,7,8 (from our perspective).
        if color_is_ours:
            camp_ranks = {0, 1, 2}
        else:
            camp_ranks = {6, 7, 8}

        # Count board pieces in enemy camp
        for r in range(9):
            for f in range(9):
                # Check the piece planes
                for pt_idx in range(14):
                    offset = 0 if color_is_ours else 14
                    if planes[offset + pt_idx, r, f] > 0.5:
                        if r in camp_ranks:
                            if pt_idx == 7:  # King
                                pass  # King doesn't count for points or piece count
                            else:
                                pieces_in_camp += 1
                                # Major pieces: Bishop(4), Rook(5), Horse(12), Dragon(13)
                                if pt_idx in (4, 5, 12, 13):
                                    points += 5
                                else:
                                    points += 1

        # Add hand piece points
        for i in range(7):
            hand_plane = 29 + i if color_is_ours else 36 + i
            count = planes[hand_plane, 0, 0]  # scalar broadcast
            if i in (4, 5):  # Bishop, Rook
                points += int(count) * 5
            else:
                points += int(count)

        if color_is_ours:
            planes[44] = float(points) / 28.0
            planes[46] = float(pieces_in_camp) / 10.0
        else:
            planes[45] = float(points) / 28.0
            planes[47] = float(pieces_in_camp) / 10.0

    return planes


def move_to_policy_index(move_str, flip):
    """
    Convert a USI move string to a policy index (0-3848).
    If flip=True, the move is from WHITE's perspective and needs 180° rotation.
    """
    if move_str[1] == '*':
        # Drop: "P*5e"
        pt_char = move_str[0].upper()
        pt_map = {'P': 0, 'L': 1, 'N': 2, 'S': 3, 'B': 4, 'R': 5, 'G': 6}
        pt = pt_map[pt_char]
        to_f = int(move_str[2]) - 1
        to_r = ord(move_str[3]) - ord('a')
        if flip:
            to_f = 8 - to_f
            to_r = 8 - to_r
        to_sq = to_f * 9 + to_r
        # Drop index: after all board+promo moves
        # We need the actual policy map to find the index.
        return ('drop', pt, to_sq)
    else:
        # Board move: "7g7f" or "7g7f+"
        from_f = int(move_str[0]) - 1
        from_r = ord(move_str[1]) - ord('a')
        to_f = int(move_str[2]) - 1
        to_r = ord(move_str[3]) - ord('a')
        promote = len(move_str) >= 5 and move_str[4] == '+'

        if flip:
            from_f, from_r = 8 - from_f, 8 - from_r
            to_f, to_r = 8 - to_f, 8 - to_r

        from_sq = from_f * 9 + from_r
        to_sq = to_f * 9 + to_r
        return ('board', from_sq, to_sq, promote)


# =====================================================================
# Dataset
# =====================================================================

class PrecomputedDataset(Dataset):
    """
    Fast dataset loading pre-computed .npz shard files.

    Expects files named: {prefix}_000.npz, {prefix}_001.npz, ...
    Each shard contains: planes (float16), policy (int32), wdl (float32).

    Memory-efficient: keeps shards as separate arrays, uses index mapping
    to find the right shard + offset for each sample. No concatenation.

    Usage:
        dataset = PrecomputedDataset("/path/to/prefix")
    """

    def __init__(self, prefix):
        self.shards = []       # list of (planes, policy, wdl) per shard
        self.shard_offsets = [] # cumulative offset for each shard
        total = 0

        shard_idx = 0
        while True:
            path = f"{prefix}_{shard_idx:03d}.npz"
            if not os.path.exists(path):
                break
            data = np.load(path)
            planes = data['planes']  # (N, 48, 9, 9) float16
            policy = data['policy']  # (N,) int32
            wdl = data['wdl']        # (N, 3) float32
            self.shards.append((planes, policy, wdl))
            self.shard_offsets.append(total)
            total += len(planes)
            shard_idx += 1
            if shard_idx % 20 == 0:
                print(f"  Loaded {shard_idx} shards ({total/1e6:.1f}M positions)...")

        if shard_idx == 0:
            raise FileNotFoundError(f"No shard files found: {prefix}_000.npz")

        self.total = total
        print(f"Loaded {total:,} positions from {shard_idx} shards "
              f"(memory: ~{total * 48 * 81 * 2 / 1e9:.1f} GB)")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Binary search for the right shard
        # (simple linear scan is fine for ~100 shards)
        shard_id = len(self.shard_offsets) - 1
        for i in range(len(self.shard_offsets) - 1):
            if idx < self.shard_offsets[i + 1]:
                shard_id = i
                break

        local_idx = idx - self.shard_offsets[shard_id]
        planes, policy, wdl = self.shards[shard_id]

        return (torch.from_numpy(planes[local_idx].astype(np.float32)),
                torch.tensor(policy[local_idx], dtype=torch.long),
                torch.tensor(wdl[local_idx], dtype=torch.float32))


class ShogiDataset(Dataset):
    """
    Loads training data from a text file (slow — use PrecomputedDataset instead).
    Each line: sfen <SFEN> [bestmove <move>] [score <score>] result <W|D|L>
    """

    def __init__(self, filepath, move_index_fn):
        self.samples = []
        self.move_index_fn = move_index_fn

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    sample = self._parse_line(line)
                    if sample:
                        self.samples.append(sample)
                except Exception:
                    continue

        print(f"Loaded {len(self.samples)} positions from {filepath}")

    def _parse_line(self, line):
        # Format: sfen <SFEN> [bestmove <move>] [score <score>] result <W|D|L>
        # bestmove is optional — if missing, policy_idx = -1 (value-only training)
        parts = line.split()
        if parts[0] == 'sfen':
            sfen_parts = []
            i = 1
            while i < len(parts) and parts[i] not in ('bestmove', 'result', 'score'):
                sfen_parts.append(parts[i])
                i += 1
            sfen = ' '.join(sfen_parts)

            bestmove = None
            result = None
            score = None
            while i < len(parts):
                if parts[i] == 'bestmove' and i + 1 < len(parts):
                    bestmove = parts[i + 1]
                    i += 2
                elif parts[i] == 'score' and i + 1 < len(parts):
                    score = int(parts[i + 1])
                    i += 2
                elif parts[i] == 'result' and i + 1 < len(parts):
                    result = parts[i + 1]
                    i += 2
                else:
                    i += 1

            if result:
                # Value-only record (no bestmove) or full record
                return (sfen, bestmove, result, score)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sfen, bestmove, result, score = self.samples[idx]
        flip = sfen.split()[1] == 'w'

        # Input planes
        planes = sfen_to_planes(sfen)

        # Policy target (-1 if no bestmove available)
        if bestmove:
            move_info = move_to_policy_index(bestmove, flip)
            policy_idx = self.move_index_fn(move_info)
        else:
            policy_idx = -1  # No policy target

        # Value target: WDL
        # If we have a score, use soft WDL derived from evaluation
        if score is not None:
            import math
            win_rate = 1.0 / (1.0 + math.exp(-score / 600.0))
            # Blend with game result for a softer target
            if result == 'W':
                hard = [1.0, 0.0, 0.0]
            elif result == 'D':
                hard = [0.0, 1.0, 0.0]
            else:
                hard = [0.0, 0.0, 1.0]
            # 70% score-based, 30% game result
            soft = [win_rate, 0.0, 1.0 - win_rate]
            wdl = [0.7 * s + 0.3 * h for s, h in zip(soft, hard)]
        elif result == 'W':
            wdl = [1.0, 0.0, 0.0]
        elif result == 'D':
            wdl = [0.0, 1.0, 0.0]
        else:  # L
            wdl = [0.0, 0.0, 1.0]

        return (torch.tensor(planes),
                torch.tensor(policy_idx, dtype=torch.long),
                torch.tensor(wdl))


# =====================================================================
# Training loop
# =====================================================================

def build_move_index():
    """Build the policy move index lookup tables."""
    board_idx = {}
    promo_idx = {}
    drop_idx = {}
    current = 0

    BOARD = 9
    def in_bounds(f, r): return 0 <= f < BOARD and 0 <= r < BOARD

    piece_moves_def = {
        'pawn':[(0,-1,1)],'lance':[(0,-1,8)],'knight':[(-1,-2,1),(1,-2,1)],
        'silver':[(0,-1,1),(-1,-1,1),(1,-1,1),(-1,1,1),(1,1,1)],
        'gold':[(0,-1,1),(-1,-1,1),(1,-1,1),(-1,0,1),(1,0,1),(0,1,1)],
        'bishop':[(-1,-1,8),(-1,1,8),(1,-1,8),(1,1,8)],
        'rook':[(0,-1,8),(0,1,8),(-1,0,8),(1,0,8)],
        'king':[(df,dr,1) for df in(-1,0,1) for dr in(-1,0,1) if(df,dr)!=(0,0)],
        'horse':[(-1,-1,8),(-1,1,8),(1,-1,8),(1,1,8),(0,-1,1),(0,1,1),(-1,0,1),(1,0,1)],
        'dragon':[(0,-1,8),(0,1,8),(-1,0,8),(1,0,8),(-1,-1,1),(-1,1,1),(1,-1,1),(1,1,1)],
    }
    valid_pairs = set()
    for moves in piece_moves_def.values():
        for f in range(9):
            for r in range(9):
                for df,dr,md in moves:
                    for dist in range(1,md+1):
                        nf,nr=f+df*dist,r+dr*dist
                        if not in_bounds(nf,nr): break
                        valid_pairs.add((f*9+r,nf*9+nr))

    for f,t in sorted(valid_pairs):
        board_idx[(f,t)] = current; current += 1
    promo_pairs = {(f,t) for f,t in valid_pairs if f%9<=2 or t%9<=2}
    for f,t in sorted(promo_pairs):
        promo_idx[(f,t)] = current; current += 1
    for pt in range(7):
        for sq in range(81):
            drop_idx[(pt,sq)] = current; current += 1

    def move_info_to_idx(info):
        if info[0] == 'drop':
            _, pt, sq = info
            return drop_idx.get((pt, sq), 0)
        elif info[0] == 'board':
            _, f, t, promote = info
            if promote:
                return promo_idx.get((f, t), 0)
            else:
                return board_idx.get((f, t), 0)
        return 0

    return move_info_to_idx


def train(args):
    # --- Device setup ---
    num_gpus = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, GPUs: {num_gpus}")

    # Generate policy map
    attn_map = generate_attn_policy_map()
    move_info_to_idx = build_move_index()

    # --- Model ---
    cfg = ShogiBT4Config()
    if args.d_model:
        cfg.embedding_size = args.d_model
        cfg.policy_d_model = args.d_model
    if args.encoders:
        cfg.num_encoders = args.encoders
    if args.heads:
        cfg.num_heads = args.heads

    model = ShogiBT4(cfg, attn_map).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.wd)

    # --- Resume from checkpoint ---
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            # Handle DataParallel state_dict keys (strip 'module.' prefix)
            state_dict = ckpt['model']
            if any(k.startswith('module.') for k in state_dict):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0)
            print(f"  Resumed at epoch {start_epoch}")
        else:
            print(f"Checkpoint not found: {args.resume}, starting from scratch")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, last_epoch=start_epoch - 1 if start_epoch > 0 else -1)

    # --- Multi-GPU with DataParallel ---
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)

    # --- Dataset ---
    # Auto-detect format: if prefix_000.npz exists, use pre-computed; else text
    if os.path.exists(f"{args.data}_000.npz"):
        print("Using pre-computed binary dataset (fast)")
        dataset = PrecomputedDataset(args.data)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)
    elif os.path.exists(args.data):
        print("Using text dataset (slow — consider running precompute.py first)")
        dataset = ShogiDataset(args.data, move_info_to_idx)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)
    else:
        print(f"Training data not found: {args.data}")
        print("Creating a small synthetic dataset for testing...")
        create_synthetic_data(args.data + ".sfen")
        dataset = ShogiDataset(args.data + ".sfen", move_info_to_idx)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        n_batches = 0
        t0 = time.time()

        for planes, policy_target, wdl_target in loader:
            planes = planes.to(device, non_blocking=True)
            policy_target = policy_target.to(device, non_blocking=True)
            wdl_target = wdl_target.to(device, non_blocking=True)

            policy_logits, wdl_logits, mlh = model(planes)

            # Policy loss: cross-entropy (only for samples with valid move targets)
            has_policy = policy_target >= 0
            if has_policy.any():
                policy_loss = F.cross_entropy(
                    policy_logits[has_policy], policy_target[has_policy])
            else:
                policy_loss = torch.tensor(0.0, device=device)

            # Value loss: cross-entropy on WDL
            value_loss = F.cross_entropy(wdl_logits, wdl_target)

            # Combined loss
            loss = policy_loss + args.value_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

            # Print progress within epoch
            if n_batches % 1000 == 0:
                avg_loss = total_loss / n_batches
                samples = n_batches * args.batch
                elapsed = time.time() - t0
                speed = samples / elapsed
                print(f"  Epoch {epoch+1} batch {n_batches}: "
                      f"loss={avg_loss:.4f} "
                      f"({samples/1e6:.1f}M samples, {speed:.0f} samples/sec)")

        scheduler.step()

        elapsed = time.time() - t0
        samples_per_sec = (n_batches * args.batch) / max(elapsed, 1)
        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"loss={total_loss/n_batches:.4f}  "
              f"policy={total_policy_loss/n_batches:.4f}  "
              f"value={total_value_loss/n_batches:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}  "
              f"time={elapsed:.1f}s  "
              f"speed={samples_per_sec:.0f} samples/sec")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            # Save the underlying model (unwrap DataParallel if needed)
            raw_model = model.module if hasattr(model, 'module') else model
            path = f"shogi_bt4_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg': vars(cfg),
                'attn_map': attn_map,
            }, path)
            print(f"  Saved {path}")

    # Export to ONNX
    if args.export_onnx:
        raw_model = model.module if hasattr(model, 'module') else model
        export_onnx(raw_model, cfg, args.export_onnx, device)


def export_onnx(model, cfg, path, device):
    """Export the trained model to ONNX format."""
    model.eval()
    dummy = torch.randn(1, cfg.input_planes, 9, 9, device=device)

    torch.onnx.export(
        model, dummy, path,
        input_names=['input_planes'],
        output_names=['policy', 'wdl', 'mlh'],
        dynamic_axes={
            'input_planes': {0: 'batch'},
            'policy': {0: 'batch'},
            'wdl': {0: 'batch'},
            'mlh': {0: 'batch'},
        },
        opset_version=18,
    )
    print(f"Exported ONNX model to {path}")


def create_synthetic_data(path, n=1000):
    """Create synthetic training data for testing the pipeline."""
    import random

    PIECES = "PLNSBRGK"
    with open(path, 'w') as f:
        for _ in range(n):
            # Just use starting position with random "best moves"
            sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
            moves = ["7g7f", "2g2f", "3g3f", "6i7h", "5g5f", "4g4f",
                      "2h7h", "2h3h", "2h6h", "1g1f"]
            move = random.choice(moves)
            result = random.choice(['W', 'D', 'L'])
            f.write(f"sfen {sfen} bestmove {move} result {result}\n")
    print(f"Created synthetic data: {path} ({n} positions)")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ShogiBT4")
    parser.add_argument("--data", default="train.sfen", help="Training data file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--encoders", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--export-onnx", default=None, help="Export ONNX after training")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint (.pt file)")
    args = parser.parse_args()
    train(args)
