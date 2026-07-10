"""
Sanity-check a training checkpoint by playing a game with it — no ONNX or
TensorRT needed. Loads the .pt directly, masks the policy to legal moves,
and plays greedy (or sampled) self-play, printing the model's move choice,
WDL and MLH every ply.

What "reasonable" looks like after one epoch: normal opening moves (7g7f,
2g2f, 6i7h class), win% near 50% early and drifting with the position, MLH
starting near ~100 plies and falling. Red flags: drops/sacrifices for
nothing every move, win% pinned at 0/100 from move 1, MLH constant.

Usage:
    python play_checkpoint.py checkpoints/shogi_bt4_epoch1.pt
    python play_checkpoint.py ckpt.pt --moves 120 --temperature 0.3
    python play_checkpoint.py ckpt.pt --vs-random     # model plays BLACK only

Prints a "position startpos moves ..." line at the end — paste it into a
USI GUI (ShogiGUI etc.) to replay the game visually.
"""

import argparse
import random
import sys

import numpy as np
import torch
import cshogi

from shogi_model_v2 import ShogiBT4v2, ShogiBT4v2Config
from shogi_train import sfen_to_planes, move_to_policy_index


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ShogiBT4v2Config()
    for k, v in ckpt.get("cfg", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = ShogiBT4v2(cfg)
    state = ckpt["model"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded {path} (epoch {epoch}, {model.count_parameters():,} params, "
          f"d={cfg.embedding_size} x{cfg.num_encoders})")
    return model


@torch.no_grad()
def evaluate(model, board, device):
    """Returns (list of (usi, prob) over legal moves, win%, draw%, mlh)."""
    sfen = board.sfen()
    flip = board.turn == cshogi.WHITE
    planes = sfen_to_planes(sfen, in_check=board.is_check())
    x = torch.from_numpy(planes).unsqueeze(0).to(device)
    policy, wdl_logits, mlh = model(x)

    legal = [(cshogi.move_to_usi(m),
              move_to_policy_index(cshogi.move_to_usi(m), flip))
             for m in board.legal_moves]
    idxs = torch.tensor([i for _, i in legal], device=device)
    logits = policy[0, idxs]
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    wdl = torch.softmax(wdl_logits[0], dim=0).cpu().numpy()
    moves = sorted(zip((u for u, _ in legal), probs),
                   key=lambda t: -t[1])
    return moves, float(wdl[0]), float(wdl[1]), float(mlh[0, 0])


def pick(moves, temperature):
    if temperature <= 0:
        return moves[0][0]
    us, ps = zip(*moves)
    ps = np.asarray(ps, dtype=np.float64) ** (1.0 / temperature)
    ps /= ps.sum()
    return np.random.choice(us, p=ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--moves", type=int, default=100, help="Max plies")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy argmax; >0 samples from the policy")
    ap.add_argument("--vs-random", action="store_true",
                    help="Model plays BLACK, random legal mover plays WHITE "
                         "(default: model self-play)")
    ap.add_argument("--sfen", default=None, help="Start position (default: startpos)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    model = load_model(args.checkpoint, args.device)
    board = cshogi.Board(args.sfen) if args.sfen else cshogi.Board()

    played = []
    print(f"\n ply  side  move    win%   draw%   MLH   (top-3 policy)")
    for ply in range(1, args.moves + 1):
        legal = list(board.legal_moves)
        if not legal:
            side = "BLACK" if board.turn == cshogi.BLACK else "WHITE"
            print(f"\nCheckmate — {side} to move has no legal moves.")
            break
        if board.is_draw() == cshogi.REPETITION_DRAW:
            print("\nSennichite (repetition draw).")
            break

        side = "b" if board.turn == cshogi.BLACK else "w"
        if args.vs_random and board.turn == cshogi.WHITE:
            usi = cshogi.move_to_usi(random.choice(legal))
            print(f"{ply:4d}   {side}   {usi:7s} (random)")
        else:
            moves, win, draw, mlh_v = evaluate(model, board, args.device)
            usi = pick(moves, args.temperature)
            top3 = "  ".join(f"{u}:{p:.2f}" for u, p in moves[:3])
            print(f"{ply:4d}   {side}   {usi:7s} {win*100:5.1f}  {draw*100:5.1f}"
                  f"  {mlh_v:6.1f}   {top3}")
        board.push_usi(usi)
        played.append(usi)

    print(f"\nposition {'sfen ' + args.sfen if args.sfen else 'startpos'} "
          f"moves {' '.join(played)}")


if __name__ == "__main__":
    main()
