"""
Quick test: play a game using our ShogiBT4 model vs random.
Shows that the full pipeline works end-to-end.
"""

import sys
import os
import random
import numpy as np

# Use the C++ board via our Python-accessible encoder
sys.path.insert(0, os.path.dirname(__file__))

from shogi_train import sfen_to_planes, move_to_policy_index
from shogi_model import generate_attn_policy_map

# =====================================================================
# Minimal Shogi board in Python (for playing games)
# =====================================================================

class MiniShogiBoard:
    """Thin wrapper around SFEN strings + python-shogi-like move gen.
    Uses our C++ board via subprocess for move generation."""

    def __init__(self):
        self.sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        self._load_board()

    def _load_board(self):
        """Load board state from SFEN using our C++ code."""
        # We'll use a subprocess to call our C++ test binary for move gen.
        # For now, use a simple Python implementation.
        pass

    def set_sfen(self, sfen):
        self.sfen = sfen

    @property
    def side_to_move(self):
        return self.sfen.split()[1]

    @property
    def ply(self):
        parts = self.sfen.split()
        try:
            return int(parts[3]) if len(parts) > 3 else 1
        except:
            return 1


def play_game_with_onnx(onnx_path, max_moves=200, verbose=True):
    """Play a full game: model (BLACK) vs random (WHITE)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # We need the C++ board for legal move generation.
    # Build a small test binary that accepts SFEN and returns legal moves.
    # For now, use a subprocess approach with our compiled binary.

    # Actually, let's build a quick move-gen helper
    import subprocess, tempfile

    # Compile a small helper that reads SFEN from stdin and outputs legal moves
    helper_src = r'''
#include "shogi/board.h"
#include "shogi/board.cc"
#include "shogi/bitboard.cc"
#include <iostream>
#include <sstream>
using namespace lczero;

int main() {
    ShogiTables::Init();
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        ShogiBoard board;
        if (!board.SetFromSfen(line)) {
            std::cout << "ERROR" << std::endl;
            continue;
        }

        // Output legal moves
        MoveList moves = board.GenerateLegalMoves();
        for (size_t i = 0; i < moves.size(); ++i) {
            if (i > 0) std::cout << " ";
            std::cout << moves[i].ToString();
        }
        std::cout << std::endl;

        // Check game result
        if (moves.empty()) {
            std::cout << "CHECKMATE" << std::endl;
        }
    }
    return 0;
}
'''

    helper_path = "/tmp/shogi_movegen_helper"
    src_path = "/tmp/shogi_movegen_helper.cpp"

    if not os.path.exists(helper_path):
        with open(src_path, 'w') as f:
            f.write(helper_src)
        lc0_src = os.path.join(os.path.dirname(__file__), 'lc0', 'src')
        result = subprocess.run(
            ['g++', '-std=c++20', '-O2', f'-I{lc0_src}', '-o', helper_path, src_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Failed to compile helper: {result.stderr}")
            return
        print("Compiled move generation helper")

    # Start the helper process
    proc = subprocess.Popen(
        [helper_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1
    )

    def get_legal_moves(sfen):
        proc.stdin.write(sfen + '\n')
        proc.stdin.flush()
        line = proc.stdout.readline().strip()
        if line == "ERROR" or not line:
            return []
        return line.split()

    def apply_move_to_sfen(sfen, move_str):
        """Apply a move to SFEN and return new SFEN.
        Uses the C++ board for correctness."""
        apply_src = f'''
#include "shogi/board.h"
#include "shogi/board.cc"
#include "shogi/bitboard.cc"
#include <iostream>
using namespace lczero;
int main() {{
    ShogiTables::Init();
    ShogiBoard board;
    board.SetFromSfen("{sfen}");
    Move m = Move::Parse("{move_str}");
    board.DoMove(m);
    std::cout << board.ToSfen() << std::endl;
    return 0;
}}
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(apply_src)
            tmp_src = f.name
        tmp_bin = tmp_src.replace('.cpp', '')
        lc0_src = os.path.join(os.path.dirname(__file__), 'lc0', 'src')
        subprocess.run(['g++', '-std=c++20', '-O2', f'-I{lc0_src}', '-o', tmp_bin, tmp_src],
                       capture_output=True)
        result = subprocess.run([tmp_bin], capture_output=True, text=True)
        os.unlink(tmp_src)
        os.unlink(tmp_bin)
        return result.stdout.strip()

    # Build policy index lookup (same as training)
    attn_map = generate_attn_policy_map()
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

    board_idx = {}; promo_idx = {}; drop_idx = {}; current = 0
    for f,t in sorted(valid_pairs):
        board_idx[(f,t)] = current; current += 1
    promo_pairs = {(f,t) for f,t in valid_pairs if f%9<=2 or t%9<=2}
    for f,t in sorted(promo_pairs):
        promo_idx[(f,t)] = current; current += 1
    for pt in range(7):
        for sq in range(81):
            drop_idx[(pt,sq)] = current; current += 1

    def move_str_to_policy_idx(move_str, flip):
        info = move_to_policy_index(move_str, flip)
        if info[0] == 'drop':
            _, pt, sq = info
            return drop_idx.get((pt, sq), -1)
        elif info[0] == 'board':
            _, f, t, promote = info
            if promote:
                return promo_idx.get((f, t), -1)
            else:
                return board_idx.get((f, t), -1)
        return -1

    # --- Play the game ---
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    move_history = []

    print("=== Game: Model (BLACK) vs Random (WHITE) ===\n")

    for move_num in range(max_moves):
        side = sfen.split()[1]
        flip = (side == 'w')

        # Get legal moves
        legal = get_legal_moves(sfen)
        if not legal:
            winner = "WHITE" if side == 'b' else "BLACK"
            print(f"\nCheckmate! {winner} wins after {move_num} moves.")
            break

        if side == 'b':
            # Model's turn (BLACK) — use NN policy
            planes = sfen_to_planes(sfen)
            planes_batch = planes[np.newaxis].astype(np.float32)
            policy, wdl, mlh = sess.run(None, {'input_planes': planes_batch})
            policy = policy[0]  # (3849,)

            # Get policy scores for legal moves
            best_move = None
            best_score = -1e10
            for m in legal:
                idx = move_str_to_policy_idx(m, flip)
                if idx >= 0 and idx < len(policy):
                    score = policy[idx]
                    if score > best_score:
                        best_score = score
                        best_move = m

            if best_move is None:
                best_move = random.choice(legal)

            chosen = best_move
            W, D, L = wdl[0]
            if verbose and move_num < 20:
                print(f"Move {move_num+1}. BLACK(model): {chosen}  "
                      f"WDL=({W:.2f},{D:.2f},{L:.2f})  "
                      f"legal={len(legal)}")
            elif verbose and move_num == 20:
                print("  ... (suppressing output)")
        else:
            # Random player (WHITE)
            chosen = random.choice(legal)
            if verbose and move_num < 20:
                print(f"Move {move_num+1}. WHITE(random): {chosen}  "
                      f"legal={len(legal)}")

        move_history.append(chosen)
        sfen = apply_move_to_sfen(sfen, chosen)

    else:
        print(f"\nGame drawn by move limit ({max_moves} moves).")

    proc.terminate()
    print(f"\nTotal moves: {len(move_history)}")
    print(f"Moves: {' '.join(move_history[:30])}{'...' if len(move_history)>30 else ''}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="shogi_bt4_test.onnx")
    ap.add_argument("--max-moves", type=int, default=200)
    args = ap.parse_args()
    play_game_with_onnx(args.onnx, args.max_moves)
