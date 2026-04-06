"""
MCTS (Monte Carlo Tree Search) for Shogi with neural network evaluation.

Implements PUCT search as used in AlphaZero/Leela Chess Zero:
  - UCB score = Q + U where U = cpuct * P * sqrt(N_parent) / (1 + N_child)
  - Dirichlet noise at root for exploration during training
  - Virtual loss for parallel search (future)
  - Batched NN evaluation

Usage:
    from shogi_mcts import MCTS, MCTSConfig
    mcts = MCTS(nn_evaluator, config=MCTSConfig())
    best_move, info = mcts.search(sfen)
"""

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class MCTSConfig:
    # PUCT constants (lc0 defaults)
    cpuct_init: float = 1.745
    cpuct_base: float = 38739.0
    cpuct_factor: float = 3.894

    # First Play Urgency
    fpu_value: float = 0.330       # Non-root FPU reduction
    fpu_root: float = 1.0          # Root FPU (absolute)
    fpu_relative: bool = True      # Use relative FPU (default in lc0)

    # Dirichlet noise (for training self-play)
    noise_epsilon: float = 0.0     # 0 = disabled, 0.25 = typical training
    noise_alpha: float = 0.15      # Shogi: smaller than chess 0.3 (more legal moves)

    # Search limits
    max_nodes: int = 800           # Total nodes to expand
    max_time: float = 0.0          # Time limit in seconds (0 = unlimited)
    max_depth: int = 200           # Maximum tree depth

    # Game limits (tournament rules)
    max_game_moves: int = 320      # Draw if no winner after this many moves

    # Temperature for move selection
    temperature: float = 0.0       # 0 = argmax, >0 = proportional to N^(1/T)
    temp_moves: int = 30           # Apply temperature for first N moves of game

    # Draw handling
    draw_score: float = 0.0        # Score adjustment for draws


# =====================================================================
# Tree node
# =====================================================================

class Node:
    """A single node in the MCTS tree."""

    __slots__ = [
        'move',         # Move that led to this node (None for root)
        'parent',       # Parent node
        'children',     # List of child nodes (None if unexpanded)
        'n',            # Visit count
        'w',            # Total value (from this node's perspective)
        'd',            # Total draw accumulator
        'p',            # Policy prior (from parent's NN evaluation)
        'is_terminal',  # True if game is over at this node
        'terminal_wdl', # (W, D, L) if terminal, else None
    ]

    def __init__(self, move=None, parent=None, prior=0.0):
        self.move = move
        self.parent = parent
        self.children = None    # Lazily expanded
        self.n = 0
        self.w = 0.0
        self.d = 0.0
        self.p = prior
        self.is_terminal = False
        self.terminal_wdl = None

    @property
    def q(self):
        """Average value (from this node's perspective)."""
        if self.n == 0:
            return 0.0
        return self.w / self.n

    @property
    def d_avg(self):
        """Average draw probability."""
        if self.n == 0:
            return 0.0
        return self.d / self.n

    def is_expanded(self):
        return self.children is not None

    def is_root(self):
        return self.parent is None


# =====================================================================
# Board interface (uses compiled C++ helper)
# =====================================================================

class BoardHelper:
    """
    Interface to our C++ Shogi board for move generation and SFEN updates.
    Uses a subprocess for the compiled C++ movegen helper.
    """

    def __init__(self, lc0_src_path=None):
        import subprocess, os

        if lc0_src_path is None:
            lc0_src_path = os.path.join(os.path.dirname(__file__), 'lc0', 'src')

        self.helper_path = "/tmp/shogi_mcts_helper"
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

        // Commands:
        // LEGAL <sfen>          → list legal moves
        // APPLY <sfen> <move>   → return new SFEN after move

        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "LEGAL") {
            std::string rest;
            std::getline(ss >> std::ws, rest);
            ShogiBoard board;
            if (!board.SetFromSfen(rest)) {
                std::cout << "ERROR" << std::endl;
                continue;
            }
            auto moves = board.GenerateLegalMoves();
            if (moves.empty()) {
                std::cout << "CHECKMATE" << std::endl;
            } else {
                for (size_t i = 0; i < moves.size(); ++i) {
                    if (i > 0) std::cout << " ";
                    std::cout << moves[i].ToString();
                }
                std::cout << std::endl;
            }
        } else if (cmd == "APPLY") {
            // Parse: APPLY <move> <sfen...>
            std::string move_str;
            ss >> move_str;
            std::string sfen;
            std::getline(ss >> std::ws, sfen);
            ShogiBoard board;
            if (!board.SetFromSfen(sfen)) {
                std::cout << "ERROR" << std::endl;
                continue;
            }
            Move m = Move::Parse(move_str);
            board.DoMove(m);
            std::cout << board.ToSfen() << std::endl;
        }
    }
    return 0;
}
'''
        if not os.path.exists(self.helper_path):
            src_path = self.helper_path + ".cpp"
            with open(src_path, 'w') as f:
                f.write(helper_src)
            result = subprocess.run(
                ['g++', '-std=c++20', '-O2', f'-I{lc0_src_path}',
                 '-o', self.helper_path, src_path],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to compile helper: {result.stderr}")

        self.proc = subprocess.Popen(
            [self.helper_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1
        )

    def get_legal_moves(self, sfen):
        """Return list of legal move strings, or empty if checkmate."""
        self.proc.stdin.write(f"LEGAL {sfen}\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        if line == "CHECKMATE" or line == "ERROR":
            return []
        return line.split()

    def apply_move(self, sfen, move_str):
        """Apply a move and return the resulting SFEN."""
        self.proc.stdin.write(f"APPLY {move_str} {sfen}\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def close(self):
        self.proc.terminate()

    def __del__(self):
        try:
            self.close()
        except:
            pass


# =====================================================================
# Neural network evaluator interface
# =====================================================================

class NNEvaluator:
    """
    Wraps an ONNX model for MCTS evaluation.

    The NN takes (batch, 44, 9, 9) and returns:
      - policy (batch, 3849) raw logits
      - wdl (batch, 3) logits
      - mlh (batch, 1)
    """

    def __init__(self, onnx_path, policy_index_fn, use_gpu=True):
        import onnxruntime as ort

        providers = []
        if use_gpu:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.policy_index_fn = policy_index_fn  # move_str → policy_idx

    def evaluate(self, sfen, legal_moves):
        """
        Evaluate a position.

        Returns:
            value: float (from side-to-move perspective, W-L)
            draw:  float (draw probability)
            policy: dict {move_str: prior_probability}
        """
        from shogi_train import sfen_to_planes

        planes = sfen_to_planes(sfen)
        planes_batch = planes[np.newaxis].astype(np.float32)

        policy_logits, wdl_logits, mlh = self.sess.run(
            None, {'input_planes': planes_batch}
        )

        # WDL → value
        wdl = _softmax(wdl_logits[0])
        value = float(wdl[0] - wdl[2])  # W - L
        draw = float(wdl[1])

        # Policy: extract logits for legal moves, softmax
        flip = sfen.split()[1] == 'w'
        logits = policy_logits[0]

        move_priors = {}
        max_logit = -1e10
        for m in legal_moves:
            idx = self.policy_index_fn(m, flip)
            if idx >= 0 and idx < len(logits):
                move_priors[m] = logits[idx]
                max_logit = max(max_logit, logits[idx])

        # Softmax over legal moves
        if move_priors:
            total = 0.0
            for m in move_priors:
                move_priors[m] = math.exp(move_priors[m] - max_logit)
                total += move_priors[m]
            if total > 0:
                for m in move_priors:
                    move_priors[m] /= total

        return value, draw, move_priors


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# =====================================================================
# MCTS
# =====================================================================

class MCTS:
    """
    Monte Carlo Tree Search with PUCT selection.

    Usage:
        mcts = MCTS(nn_evaluator, board_helper, config)
        move, info = mcts.search(sfen, game_ply=1)
    """

    def __init__(self, nn_eval: NNEvaluator, board: BoardHelper,
                 config: MCTSConfig = None):
        self.nn = nn_eval
        self.board = board
        self.cfg = config or MCTSConfig()

    def search(self, sfen, game_ply=1):
        """
        Run MCTS from the given position.

        Args:
            sfen: position string
            game_ply: current game move number (for temperature)

        Returns:
            (best_move_str, info_dict)
        """
        cfg = self.cfg

        # Create root node
        root = Node()
        legal_moves = self.board.get_legal_moves(sfen)

        if not legal_moves:
            return None, {"result": "checkmate"}

        if len(legal_moves) == 1:
            return legal_moves[0], {"nodes": 0, "forced": True}

        # Evaluate root
        value, draw, priors = self.nn.evaluate(sfen, legal_moves)

        # Create children
        root.children = []
        for m in legal_moves:
            prior = priors.get(m, 1e-6)
            root.children.append(Node(move=m, parent=root, prior=prior))

        # Apply Dirichlet noise at root
        if cfg.noise_epsilon > 0:
            self._add_dirichlet_noise(root)

        # Backpropagate root evaluation
        root.n = 1
        root.w = value
        root.d = draw

        # Main search loop
        t0 = time.time()
        nodes_expanded = 0

        # Cache SFEN for each node path (root → current)
        sfen_cache = {id(root): sfen}

        while nodes_expanded < cfg.max_nodes:
            # Check time limit
            if cfg.max_time > 0 and (time.time() - t0) >= cfg.max_time:
                break

            # 1. Select: walk down tree using PUCT
            node, path = self._select(root)

            # 2. If terminal, backpropagate known value
            if node.is_terminal:
                self._backpropagate(path, node.terminal_wdl[0] - node.terminal_wdl[2],
                                    node.terminal_wdl[1])
                continue

            # 3. Expand: get SFEN for this node
            node_sfen = self._get_sfen(node, path, sfen_cache, sfen)

            # 4. Get legal moves and evaluate
            legal = self.board.get_legal_moves(node_sfen)

            if not legal:
                # Checkmate (side to move loses)
                node.is_terminal = True
                node.terminal_wdl = (0.0, 0.0, 1.0)  # Loss for side to move
                self._backpropagate(path, -1.0, 0.0)
                continue

            val, drw, priors = self.nn.evaluate(node_sfen, legal)

            # Create children
            node.children = []
            for m in legal:
                prior = priors.get(m, 1e-6)
                node.children.append(Node(move=m, parent=node, prior=prior))

            # 5. Backpropagate
            self._backpropagate(path, val, drw)
            nodes_expanded += 1

        elapsed = time.time() - t0

        # Select best move
        best_move = self._select_move(root, game_ply)

        # Collect info
        info = {
            "nodes": nodes_expanded,
            "time": elapsed,
            "nps": nodes_expanded / max(elapsed, 1e-6),
            "root_q": root.q,
            "root_d": root.d_avg,
            "best_n": max(c.n for c in root.children),
            "pv": self._get_pv(root),
            "children": [(c.move, c.n, c.q, c.p) for c in
                         sorted(root.children, key=lambda c: -c.n)[:10]],
        }

        return best_move, info

    def _cpuct(self, parent_n):
        """Dynamic cpuct that grows with visit count."""
        cfg = self.cfg
        return cfg.cpuct_init + cfg.cpuct_factor * math.log(
            (parent_n + cfg.cpuct_base) / cfg.cpuct_base
        )

    def _select(self, root):
        """Walk from root to a leaf using PUCT scores. Returns (leaf, path)."""
        node = root
        path = [node]

        while node.is_expanded() and not node.is_terminal:
            if not node.children:
                break
            child = self._best_child(node)
            path.append(child)
            node = child

        return node, path

    def _best_child(self, node):
        """Select child with highest PUCT score."""
        cfg = self.cfg
        cpuct = self._cpuct(node.n)
        sqrt_parent = math.sqrt(max(node.n, 1))
        is_root = node.is_root()

        # FPU (First Play Urgency)
        if cfg.fpu_relative:
            visited_policy = sum(c.p for c in node.children if c.n > 0)
            if is_root:
                fpu = -node.q - cfg.fpu_root * math.sqrt(max(visited_policy, 0))
            else:
                fpu = -node.q - cfg.fpu_value * math.sqrt(max(visited_policy, 0))
        else:
            fpu = cfg.fpu_value if not is_root else cfg.fpu_root

        best_score = -1e10
        best_child = None

        for child in node.children:
            # Q value (from parent's perspective, so negate child's Q)
            if child.n > 0:
                q = -child.q
            else:
                q = fpu

            # U = cpuct * P * sqrt(N_parent) / (1 + N_child)
            u = cpuct * child.p * sqrt_parent / (1 + child.n)

            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _get_sfen(self, node, path, cache, root_sfen):
        """Get the SFEN for a node by applying moves from root."""
        if id(node) in cache:
            return cache[id(node)]

        # Build SFEN by replaying moves from the nearest cached ancestor
        for i in range(len(path) - 1, -1, -1):
            if id(path[i]) in cache:
                sfen = cache[id(path[i])]
                for j in range(i + 1, len(path)):
                    sfen = self.board.apply_move(sfen, path[j].move)
                    cache[id(path[j])] = sfen
                return sfen

        return root_sfen

    def _backpropagate(self, path, value, draw):
        """Backpropagate value up the tree, flipping perspective at each level."""
        v = value
        d = draw
        for node in reversed(path):
            node.n += 1
            node.w += v
            node.d += d
            v = -v  # Flip perspective

    def _add_dirichlet_noise(self, root):
        """Add Dirichlet noise to root policy priors."""
        cfg = self.cfg
        n = len(root.children)
        if n == 0:
            return
        noise = np.random.dirichlet([cfg.noise_alpha] * n)
        for i, child in enumerate(root.children):
            child.p = child.p * (1 - cfg.noise_epsilon) + cfg.noise_epsilon * noise[i]

    def _select_move(self, root, game_ply):
        """Select move based on visit counts and temperature."""
        cfg = self.cfg
        children = root.children

        if cfg.temperature == 0 or game_ply > cfg.temp_moves:
            # Argmax: pick most visited
            return max(children, key=lambda c: c.n).move
        else:
            # Proportional to N^(1/T)
            visits = np.array([c.n for c in children], dtype=np.float64)
            if cfg.temperature != 1.0:
                visits = visits ** (1.0 / cfg.temperature)
            probs = visits / visits.sum()
            idx = np.random.choice(len(children), p=probs)
            return children[idx].move

    def _get_pv(self, root):
        """Extract principal variation (sequence of best moves)."""
        pv = []
        node = root
        while node.children:
            best = max(node.children, key=lambda c: c.n)
            pv.append(best.move)
            node = best
            if not node.is_expanded():
                break
        return pv


# =====================================================================
# Convenience: create MCTS with ONNX model
# =====================================================================

def create_mcts(onnx_path, config=None, lc0_src_path=None):
    """
    Create an MCTS instance with ONNX model and board helper.

    Returns (mcts, cleanup_fn)
    """
    from shogi_train import move_to_policy_index
    from shogi_model import generate_attn_policy_map

    # Build policy index lookup
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

    def move_str_to_idx(move_str, flip):
        info = move_to_policy_index(move_str, flip)
        if info[0] == 'drop':
            _, pt, sq = info
            return drop_idx.get((pt, sq), -1)
        elif info[0] == 'board':
            _, f, t, promote = info
            return (promo_idx if promote else board_idx).get((f, t), -1)
        return -1

    board = BoardHelper(lc0_src_path)
    nn = NNEvaluator(onnx_path, move_str_to_idx)
    mcts = MCTS(nn, board, config or MCTSConfig())

    return mcts, board.close


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Shogi MCTS search")
    ap.add_argument("--onnx", default="shogi_bt4_test.onnx")
    ap.add_argument("--sfen", default=None, help="SFEN position")
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--games", type=int, default=1, help="Play N games vs random")
    args = ap.parse_args()

    sfen = args.sfen or "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

    print("Initializing MCTS...")
    mcts, cleanup = create_mcts(args.onnx, MCTSConfig(max_nodes=args.nodes))

    if args.games > 1:
        # Play multiple games: MCTS vs random
        import random
        wins = draws = losses = 0
        for g in range(args.games):
            game_sfen = sfen
            ply = 0
            result = None
            while ply < 400:
                side = game_sfen.split()[1]
                legal = mcts.board.get_legal_moves(game_sfen)
                if not legal:
                    result = "WHITE" if side == 'b' else "BLACK"
                    break
                if side == 'b':
                    move, _ = mcts.search(game_sfen, game_ply=ply)
                else:
                    move = random.choice(legal)
                game_sfen = mcts.board.apply_move(game_sfen, move)
                ply += 1
            if result == "BLACK":
                wins += 1
            elif result == "WHITE":
                losses += 1
            else:
                draws += 1
            print(f"Game {g+1}: {'BLACK wins' if result=='BLACK' else 'WHITE wins' if result=='WHITE' else 'Draw'} ({ply} moves)")

        print(f"\nResults: {wins}W / {draws}D / {losses}L out of {args.games}")
    else:
        # Single search
        print(f"Position: {sfen}")
        move, info = mcts.search(sfen)
        print(f"\nBest move: {move}")
        print(f"Nodes: {info['nodes']}")
        print(f"Time: {info['time']:.2f}s ({info['nps']:.0f} nps)")
        print(f"Root Q: {info['root_q']:+.4f}")
        print(f"PV: {' '.join(info['pv'][:10])}")
        print(f"\nTop moves:")
        for m, n, q, p in info['children'][:10]:
            print(f"  {m:8s}  N={n:4d}  Q={q:+.3f}  P={p:.3f}")

    cleanup()
