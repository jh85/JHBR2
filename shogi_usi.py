"""
USI (Universal Shogi Interface) protocol implementation.

Connects our ShogiBT4 model + MCTS search to Shogi GUIs and tournament software.

USI protocol reference:
  http://shogidokoro.starfree.jp/usi.html

Usage:
    python shogi_usi.py --onnx shogi_bt4.onnx --nodes 800

The engine reads USI commands from stdin and writes responses to stdout.
"""

import sys
import os
import time
import threading
import math

import cshogi
import numpy as np

from shogi_model import generate_attn_policy_map
from shogi_train import sfen_to_planes, move_to_policy_index
from shogi_mcts import MCTSConfig


# =====================================================================
# NN Evaluator using ONNX + cshogi
# =====================================================================

class OnnxEvaluator:
    """Neural network evaluator using ONNX Runtime."""

    def __init__(self, onnx_path, use_gpu=True):
        import onnxruntime as ort
        providers = []
        if use_gpu:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self._build_policy_index()

    def _build_policy_index(self):
        """Build policy index lookup tables."""
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

        self.board_idx = {}
        self.promo_idx = {}
        self.drop_idx = {}
        current = 0
        for f,t in sorted(valid_pairs):
            self.board_idx[(f,t)] = current; current += 1
        promo_pairs = {(f,t) for f,t in valid_pairs if f%9<=2 or t%9<=2}
        for f,t in sorted(promo_pairs):
            self.promo_idx[(f,t)] = current; current += 1
        for pt in range(7):
            for sq in range(81):
                self.drop_idx[(pt,sq)] = current; current += 1

    def move_to_index(self, move_usi, flip):
        """Convert USI move string to policy index."""
        info = move_to_policy_index(move_usi, flip)
        if info[0] == 'drop':
            _, pt, sq = info
            return self.drop_idx.get((pt, sq), -1)
        elif info[0] == 'board':
            _, f, t, promote = info
            return (self.promo_idx if promote else self.board_idx).get((f, t), -1)
        return -1

    def evaluate(self, sfen, legal_moves_usi):
        """
        Evaluate a position.
        Returns: (value, draw, policy_dict)
            value: float (W-L from side-to-move)
            draw: float
            policy_dict: {move_usi: probability}
        """
        planes = sfen_to_planes(sfen)
        planes_batch = planes[np.newaxis].astype(np.float32)

        policy_logits, wdl_logits, mlh = self.sess.run(
            None, {'input_planes': planes_batch})

        # WDL
        wdl = wdl_logits[0]
        wdl = np.exp(wdl - wdl.max())
        wdl /= wdl.sum()
        value = float(wdl[0] - wdl[2])
        draw = float(wdl[1])

        # Policy
        flip = sfen.split()[1] == 'w'
        logits = policy_logits[0]

        priors = {}
        max_logit = -1e10
        for m in legal_moves_usi:
            idx = self.move_to_index(m, flip)
            if idx >= 0 and idx < len(logits):
                priors[m] = logits[idx]
                max_logit = max(max_logit, logits[idx])

        if priors:
            total = 0.0
            for m in priors:
                priors[m] = math.exp(priors[m] - max_logit)
                total += priors[m]
            if total > 0:
                for m in priors:
                    priors[m] /= total

        return value, draw, priors


# =====================================================================
# MCTS Search using cshogi
# =====================================================================

class MCTSNode:
    __slots__ = ['move', 'parent', 'children', 'n', 'w', 'p', 'is_terminal', 'terminal_v']

    def __init__(self, move=None, parent=None, prior=0.0):
        self.move = move
        self.parent = parent
        self.children = None
        self.n = 0
        self.w = 0.0
        self.p = prior
        self.is_terminal = False
        self.terminal_v = 0.0

    @property
    def q(self):
        return self.w / self.n if self.n > 0 else 0.0


class MCTSEngine:
    """MCTS search engine using cshogi for board management."""

    def __init__(self, evaluator, config=None):
        self.eval = evaluator
        self.cfg = config or MCTSConfig()
        self.board = cshogi.Board()

    def search(self, board, max_nodes=None, max_time=None):
        """
        Run MCTS search from current board position.
        Returns (best_move_usi, info_dict)
        """
        cfg = self.cfg
        if max_nodes is None:
            max_nodes = cfg.max_nodes
        if max_time is None:
            max_time = cfg.max_time

        sfen = board.sfen()
        legal_moves = list(board.legal_moves)
        legal_usi = [cshogi.move_to_usi(m) for m in legal_moves]

        if not legal_moves:
            return None, {"result": "checkmate"}
        if len(legal_moves) == 1:
            return legal_usi[0], {"nodes": 0, "forced": True}

        # Evaluate root
        value, draw, priors = self.eval.evaluate(sfen, legal_usi)

        # Create root
        root = MCTSNode()
        root.children = []
        for m_usi in legal_usi:
            root.children.append(MCTSNode(move=m_usi, parent=root,
                                          prior=priors.get(m_usi, 1e-6)))

        # Add Dirichlet noise
        if cfg.noise_epsilon > 0:
            n = len(root.children)
            noise = np.random.dirichlet([cfg.noise_alpha] * n)
            for i, child in enumerate(root.children):
                child.p = child.p * (1 - cfg.noise_epsilon) + cfg.noise_epsilon * noise[i]

        root.n = 1
        root.w = value

        # Search loop
        t0 = time.time()
        nodes = 0

        while nodes < max_nodes:
            if max_time > 0 and (time.time() - t0) >= max_time:
                break

            # Select
            node, moves_to_apply = self._select(root)

            if node.is_terminal:
                self._backprop([node], node.terminal_v)
                continue

            # Get position SFEN by applying moves from root
            b = cshogi.Board(sfen)
            for m_usi in moves_to_apply:
                b.push_usi(m_usi)

            node_sfen = b.sfen()
            node_legal = list(b.legal_moves)
            node_legal_usi = [cshogi.move_to_usi(m) for m in node_legal]

            if not node_legal:
                node.is_terminal = True
                node.terminal_v = -1.0  # Side to move loses
                self._backprop([node], -1.0)
                continue

            # Evaluate
            val, drw, priors = self.eval.evaluate(node_sfen, node_legal_usi)

            # Expand
            node.children = []
            for m_usi in node_legal_usi:
                node.children.append(MCTSNode(move=m_usi, parent=node,
                                              prior=priors.get(m_usi, 1e-6)))

            # Backprop
            self._backprop([node], val)
            nodes += 1

        elapsed = time.time() - t0

        # Select best move
        best_child = max(root.children, key=lambda c: c.n)
        best_move = best_child.move

        # Build info
        info = {
            "nodes": nodes,
            "time_ms": int(elapsed * 1000),
            "nps": int(nodes / max(elapsed, 0.001)),
            "score_cp": int(90 * math.tan(1.5637541897 * max(-0.999, min(0.999, root.q)))),
            "pv": self._get_pv(root),
            "children": [(c.move, c.n, c.q, c.p) for c in
                         sorted(root.children, key=lambda c: -c.n)[:5]],
        }

        return best_move, info

    def _select(self, root):
        """Select a leaf node using PUCT. Returns (node, moves_from_root)."""
        node = root
        moves = []
        while node.children:
            child = self._best_child(node)
            moves.append(child.move)
            node = child
            if not node.children:
                break
        return node, moves

    def _best_child(self, node):
        cfg = self.cfg
        cpuct = cfg.cpuct_init + cfg.cpuct_factor * math.log(
            (node.n + cfg.cpuct_base) / cfg.cpuct_base)
        sqrt_n = math.sqrt(max(node.n, 1))

        visited_policy = sum(c.p for c in node.children if c.n > 0)
        fpu = -node.q - cfg.fpu_value * math.sqrt(max(visited_policy, 0))

        best_score = -1e10
        best = None
        for child in node.children:
            q = -child.q if child.n > 0 else fpu
            u = cpuct * child.p * sqrt_n / (1 + child.n)
            score = q + u
            if score > best_score:
                best_score = score
                best = child
        return best

    def _backprop(self, path_tail, value):
        """Backpropagate from a node to root."""
        node = path_tail[0]
        v = value
        while node is not None:
            node.n += 1
            node.w += v
            v = -v
            node = node.parent

    def _get_pv(self, root):
        pv = []
        node = root
        while node.children:
            best = max(node.children, key=lambda c: c.n)
            pv.append(best.move)
            node = best
        return pv


# =====================================================================
# USI Protocol Handler
# =====================================================================

class USIEngine:
    """USI protocol handler."""

    ENGINE_NAME = "JHBR2"
    ENGINE_AUTHOR = "JHBR2 Team"

    def __init__(self, onnx_path=None, config=None):
        self.onnx_path = onnx_path
        self.config = config or MCTSConfig()
        self.board = cshogi.Board()
        self.engine = None  # Initialized on isready
        self.game_ply = 0

        # Options
        self.options = {
            "USI_Ponder": False,
            "Threads": 1,
            "MaxNodes": self.config.max_nodes,
            "OnnxModel": onnx_path or "shogi_bt4.onnx",
            "NoiseEpsilon": 0.0,
        }

    def run(self):
        """Main USI loop — read commands from stdin, write responses to stdout."""
        while True:
            try:
                line = input().strip()
            except EOFError:
                break

            if not line:
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "usi":
                self._cmd_usi()
            elif cmd == "isready":
                self._cmd_isready()
            elif cmd == "setoption":
                self._cmd_setoption(parts)
            elif cmd == "usinewgame":
                self._cmd_usinewgame()
            elif cmd == "position":
                self._cmd_position(parts)
            elif cmd == "go":
                self._cmd_go(parts)
            elif cmd == "stop":
                self._cmd_stop()
            elif cmd == "quit":
                break
            elif cmd == "gameover":
                self._cmd_gameover(parts)
            # Debug commands
            elif cmd == "d":
                self._cmd_debug()

    def _send(self, msg):
        """Send a message to the GUI."""
        print(msg, flush=True)

    def _cmd_usi(self):
        self._send(f"id name {self.ENGINE_NAME}")
        self._send(f"id author {self.ENGINE_AUTHOR}")
        self._send(f"option name MaxNodes type spin default {self.options['MaxNodes']} min 1 max 1000000")
        self._send(f"option name OnnxModel type string default {self.options['OnnxModel']}")
        self._send(f"option name NoiseEpsilon type string default {self.options['NoiseEpsilon']}")
        self._send("usiok")

    def _cmd_isready(self):
        """Initialize the engine (load NN model)."""
        if self.engine is None:
            onnx_path = self.options["OnnxModel"]
            log(f"Loading model: {onnx_path}")
            evaluator = OnnxEvaluator(onnx_path)
            self.config.max_nodes = int(self.options["MaxNodes"])
            self.config.noise_epsilon = float(self.options["NoiseEpsilon"])
            self.engine = MCTSEngine(evaluator, self.config)
            log(f"Model loaded, max_nodes={self.config.max_nodes}")
        self._send("readyok")

    def _cmd_setoption(self, parts):
        """Parse: setoption name <NAME> value <VALUE>"""
        try:
            name_idx = parts.index("name") + 1
            value_idx = parts.index("value") + 1
            name = parts[name_idx]
            value = parts[value_idx]
            if name in self.options:
                self.options[name] = value
                log(f"Set option {name} = {value}")
        except (ValueError, IndexError):
            pass

    def _cmd_usinewgame(self):
        self.board = cshogi.Board()
        self.game_ply = 0

    def _cmd_position(self, parts):
        """Parse: position startpos [moves m1 m2 ...] or position sfen <sfen> [moves ...]"""
        self.board = cshogi.Board()
        idx = 1

        if parts[idx] == "startpos":
            idx += 1
        elif parts[idx] == "sfen":
            idx += 1
            sfen_parts = []
            while idx < len(parts) and parts[idx] != "moves":
                sfen_parts.append(parts[idx])
                idx += 1
            sfen = " ".join(sfen_parts)
            self.board.set_sfen(sfen)

        # Apply moves
        if idx < len(parts) and parts[idx] == "moves":
            idx += 1
            while idx < len(parts):
                self.board.push_usi(parts[idx])
                idx += 1

        self.game_ply = self.board.move_number

    def _cmd_go(self, parts):
        """Parse go command and start search."""
        # Parse time controls
        btime = wtime = 0
        byoyomi = 0
        binc = winc = 0
        max_nodes = int(self.options.get("MaxNodes", self.config.max_nodes))
        max_time = 0.0

        i = 1
        while i < len(parts):
            if parts[i] == "btime" and i + 1 < len(parts):
                btime = int(parts[i + 1]); i += 2
            elif parts[i] == "wtime" and i + 1 < len(parts):
                wtime = int(parts[i + 1]); i += 2
            elif parts[i] == "byoyomi" and i + 1 < len(parts):
                byoyomi = int(parts[i + 1]); i += 2
            elif parts[i] == "binc" and i + 1 < len(parts):
                binc = int(parts[i + 1]); i += 2
            elif parts[i] == "winc" and i + 1 < len(parts):
                winc = int(parts[i + 1]); i += 2
            elif parts[i] == "nodes" and i + 1 < len(parts):
                max_nodes = int(parts[i + 1]); i += 2
            elif parts[i] == "infinite":
                max_time = 0; max_nodes = 1000000; i += 1
            elif parts[i] == "ponder":
                i += 1  # ignore ponder for now
            else:
                i += 1

        # Simple time management
        if byoyomi > 0:
            max_time = byoyomi / 1000.0 * 0.9  # Use 90% of byoyomi
        elif btime > 0 or wtime > 0:
            my_time = btime if self.board.turn == cshogi.BLACK else wtime
            my_inc = binc if self.board.turn == cshogi.BLACK else winc
            # Use ~5% of remaining time + 80% of increment
            max_time = (my_time * 0.05 + my_inc * 0.8) / 1000.0
            max_time = max(max_time, 0.1)  # At least 100ms

        # Check for entering-king declaration
        if cshogi.Board(self.board.sfen()).is_nyugyoku():
            self._send("bestmove win")
            return

        # Run search
        best_move, info = self.engine.search(
            self.board, max_nodes=max_nodes, max_time=max_time)

        if best_move is None:
            self._send("bestmove resign")
            return

        # Send info
        pv_str = " ".join(info.get("pv", [best_move]))
        score_cp = info.get("score_cp", 0)
        nodes = info.get("nodes", 0)
        time_ms = info.get("time_ms", 0)
        nps = info.get("nps", 0)

        self._send(f"info depth 1 score cp {score_cp} nodes {nodes} "
                   f"time {time_ms} nps {nps} pv {pv_str}")
        self._send(f"bestmove {best_move}")

    def _cmd_stop(self):
        """Stop search (for now, search is synchronous so this is a no-op)."""
        pass

    def _cmd_gameover(self, parts):
        """Game is over."""
        if len(parts) > 1:
            log(f"Game over: {parts[1]}")

    def _cmd_debug(self):
        """Print debug info."""
        log(f"Position: {self.board.sfen()}")
        log(f"Legal moves: {len(list(self.board.legal_moves))}")


def log(msg):
    """Write to stderr (USI protocol uses stdout for communication)."""
    print(f"info string {msg}", file=sys.stdout, flush=True)


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JHBR2 Shogi USI Engine")
    parser.add_argument("--onnx", default="shogi_bt4_test.onnx",
                        help="ONNX model path")
    parser.add_argument("--nodes", type=int, default=800,
                        help="Default search nodes")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Dirichlet noise epsilon")
    args = parser.parse_args()

    config = MCTSConfig(max_nodes=args.nodes, noise_epsilon=args.noise)
    engine = USIEngine(onnx_path=args.onnx, config=config)
    engine.run()
