"""
BT4-v2 Transformer Model for Shogi.

Changes from v1:
  - Policy uses dlshogi-style (destination, direction) encoding (2187 classes)
  - Promotion is encoded as separate directions (no promo_offset)
  - Policy head uses attention + 1x1 conv hybrid
  - Bilinear promotion scoring conditioned on (from, to)

Architecture:
  - PE-Dense input embedding (same as v1)
  - Encoder stack with multi-head attention + smolgen (same as v1)
  - NEW: Direction-based attention policy head (81 × 27 = 2187 moves)
  - WDL value head (same as v1)
  - Moves-left head (same as v1)

Move encoding (dlshogi-compatible):
  Directions 0-9:   non-promotion (UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT,
                     DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT)
  Directions 10-19:  promotion (same 10 directions)
  Directions 20-26:  drops (P, L, N, S, B, R, G)
  Total: 27 directions × 81 squares = 2187 policy outputs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


# =====================================================================
# Move encoding constants
# =====================================================================

# Direction vectors (df, dr) for BLACK's perspective.
# Piece moves FROM (to_f - df*dist, to_r - dr*dist) TO (to_f, to_r).
DIRECTION_VECTORS = [
    (0, -1),   # 0: UP
    (-1, -1),  # 1: UP_LEFT
    (1, -1),   # 2: UP_RIGHT
    (-1, 0),   # 3: LEFT
    (1, 0),    # 4: RIGHT
    (0, 1),    # 5: DOWN
    (-1, 1),   # 6: DOWN_LEFT
    (1, 1),    # 7: DOWN_RIGHT
    (-1, -2),  # 8: UP2_LEFT (knight)
    (1, -2),   # 9: UP2_RIGHT (knight)
]

NUM_DIRECTIONS = 10
NUM_PROMO_DIRECTIONS = 10  # same 10 with promotion
NUM_DROP_TYPES = 7
MAX_MOVE_LABEL = NUM_DIRECTIONS + NUM_PROMO_DIRECTIONS + NUM_DROP_TYPES  # 27
POLICY_SIZE = 81 * MAX_MOVE_LABEL  # 2187

# Sliding directions (can move multiple squares)
SLIDING_DIRS = {0, 1, 2, 3, 4, 5, 6, 7}
# Step-only directions
STEP_DIRS = {8, 9}  # knight jumps


def move_to_direction(from_sq, to_sq):
    """Convert (from, to) to direction index. Returns -1 if not a valid direction."""
    from_f, from_r = from_sq // 9, from_sq % 9
    to_f, to_r = to_sq // 9, to_sq % 9
    df = to_f - from_f
    dr = to_r - from_r

    for idx, (vf, vr) in enumerate(DIRECTION_VECTORS):
        if idx in SLIDING_DIRS:
            # Check if (df, dr) is a positive multiple of (vf, vr)
            if vf == 0 and vr == 0:
                continue
            if vf == 0:
                if df == 0 and dr != 0 and (dr > 0) == (vr > 0) and dr % vr == 0:
                    return idx
            elif vr == 0:
                if dr == 0 and df != 0 and (df > 0) == (vf > 0) and df % vf == 0:
                    return idx
            else:
                if df != 0 and dr != 0 and df % vf == 0 and dr % vr == 0:
                    dist_f = df // vf
                    dist_r = dr // vr
                    if dist_f == dist_r and dist_f > 0:
                        return idx
        else:
            # Step: exact match
            if df == vf and dr == vr:
                return idx
    return -1


def make_direction_policy_index(move_str, flip):
    """
    Convert a USI move string to a direction-based policy index (0-2186).
    If flip=True, rotate 180° (for WHITE's perspective).
    """
    if move_str[1] == '*':
        # Drop: "P*5e"
        pt_map = {'P': 0, 'L': 1, 'N': 2, 'S': 3, 'B': 4, 'R': 5, 'G': 6}
        pt = pt_map[move_str[0].upper()]
        to_f = int(move_str[2]) - 1
        to_r = ord(move_str[3]) - ord('a')
        if flip:
            to_f, to_r = 8 - to_f, 8 - to_r
        to_sq = to_f * 9 + to_r
        direction = NUM_DIRECTIONS + NUM_PROMO_DIRECTIONS + pt  # 20-26
        return direction * 81 + to_sq

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

    direction = move_to_direction(from_sq, to_sq)
    if direction < 0:
        return -1  # invalid

    if promote:
        direction += NUM_DIRECTIONS  # 0-9 → 10-19

    return direction * 81 + to_sq


# =====================================================================
# Config
# =====================================================================

@dataclass
class ShogiBT4v2Config:
    # Input
    num_piece_planes: int = 28
    hand_planes: int = 14
    aux_planes: int = 6
    input_planes: int = 48
    num_squares: int = 81
    board_size: int = 9

    # Embedding
    embedding_dense_size: int = 128
    embedding_size: int = 256
    activation: str = "mish"

    # Encoder
    num_encoders: int = 6
    num_heads: int = 8
    ffn_multiplier: float = 1.5
    smolgen_hidden: int = 64
    smolgen_compress: int = 8
    smolgen_gen_size: int = 64

    # Policy head (direction-based, 2187 outputs)
    policy_d_model: int = 256
    policy_size: int = POLICY_SIZE  # 2187

    # Value head
    value_embedding: int = 8
    value_hidden: int = 128

    # Moves-left head
    mlh_embedding: int = 8
    mlh_hidden: int = 64

    # Normalization
    norm_type: str = "layernorm"
    no_qkv_bias: bool = True

    @property
    def ffn_hidden(self):
        return int(self.embedding_size * self.ffn_multiplier)


# =====================================================================
# Shared components (same as v1)
# =====================================================================

def get_activation(name):
    return {"relu": nn.ReLU(), "mish": nn.Mish(), "silu": nn.SiLU()}.get(name, nn.Mish())

def make_norm(d, cfg):
    return nn.RMSNorm(d) if cfg.norm_type == "rmsnorm" else nn.LayerNorm(d)


class InputEmbedding(nn.Module):
    """PE-Dense embedding (same as v1)."""
    def __init__(self, cfg):
        super().__init__()
        sq = cfg.num_squares
        d = cfg.embedding_size
        ds = cfg.embedding_dense_size
        pp = cfg.num_piece_planes

        self.mult_gate = nn.Parameter(torch.ones(sq, d))
        self.add_gate = nn.Parameter(torch.zeros(sq, d))
        self.preproc = nn.Linear(sq * pp, sq * ds)
        self.embed = nn.Linear(cfg.input_planes - pp + ds, d)
        self.ln = make_norm(d, cfg)
        self.ffn1 = nn.Linear(d, cfg.ffn_hidden)
        self.ffn2 = nn.Linear(cfg.ffn_hidden, d)
        self.act = get_activation(cfg.activation)

    def forward(self, x):
        B = x.shape[0]
        piece_planes = torch.flatten(x[:, :28], 1)
        other_planes = x[:, 28:].flatten(2).transpose(1, 2)  # (B, C-28, 81) -> (B, 81, C-28)
        dense = self.preproc(piece_planes).unflatten(1, (81, -1))
        combined = torch.cat([dense, other_planes], dim=-1)
        h = self.act(self.embed(combined))
        h = h * self.mult_gate + self.add_gate
        h = self.ln(h)
        h = h + self.act(self.ffn2(self.act(self.ffn1(h))))
        return h


class Smolgen(nn.Module):
    """Smolgen: generates per-position attention biases (same as v1)."""
    def __init__(self, cfg, global_gen):
        super().__init__()
        d = cfg.embedding_size
        sq = cfg.num_squares
        self.compress = nn.Linear(d, cfg.smolgen_compress)
        self.dense1 = nn.Linear(sq * cfg.smolgen_compress, cfg.smolgen_hidden)
        self.ln1 = nn.LayerNorm(cfg.smolgen_hidden)
        self.dense2 = nn.Linear(cfg.smolgen_hidden, cfg.num_heads * cfg.smolgen_gen_size)
        self.ln2 = nn.LayerNorm(cfg.num_heads * cfg.smolgen_gen_size)
        self.act = nn.SiLU()
        self.global_gen = global_gen
        self.cfg = cfg

    def forward(self, x):
        B = x.shape[0]
        h = torch.flatten(self.act(self.compress(x)), 1)
        h = self.act(self.ln1(self.dense1(h)))
        h = self.act(self.ln2(self.dense2(h)))
        h = h.unflatten(1, (self.cfg.num_heads, self.cfg.smolgen_gen_size))
        w = self.global_gen(h)
        return w.unflatten(2, (81, 81))


class EncoderBlock(nn.Module):
    """Transformer encoder block with smolgen (same as v1)."""
    def __init__(self, cfg, global_gen):
        super().__init__()
        d = cfg.embedding_size
        no_qkv_bias = cfg.no_qkv_bias
        self.q_proj = nn.Linear(d, d, bias=not no_qkv_bias)
        self.k_proj = nn.Linear(d, d, bias=not no_qkv_bias)
        self.v_proj = nn.Linear(d, d, bias=not no_qkv_bias)
        self.out_proj = nn.Linear(d, d)
        self.ln1 = make_norm(d, cfg)
        self.smolgen = Smolgen(cfg, global_gen)
        self.ffn1 = nn.Linear(d, cfg.ffn_hidden)
        self.ffn2 = nn.Linear(cfg.ffn_hidden, d)
        self.ffn_act = get_activation(cfg.activation)
        self.ffn_ln = make_norm(d, cfg)
        self.cfg = cfg

    def forward(self, x):
        B, S, D = x.shape
        heads = self.cfg.num_heads
        depth = D // heads
        smol_bias = self.smolgen(x)
        Q = self.q_proj(x).unflatten(2, (heads, depth)).permute(0, 2, 1, 3)
        K = self.k_proj(x).unflatten(2, (heads, depth)).permute(0, 2, 1, 3)
        V = self.v_proj(x).unflatten(2, (heads, depth)).permute(0, 2, 1, 3)
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(depth) + smol_bias
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).permute(0, 2, 1, 3).flatten(2)
        h = self.ln1(self.out_proj(out) + x)
        f = self.ffn_act(self.ffn1(h))
        h = self.ffn_ln(self.ffn2(f) + h)
        return h


# =====================================================================
# NEW: Direction-based Attention Policy Head
# =====================================================================

class DirectionPolicyHead(nn.Module):
    """
    Direction-based policy head for Shogi (dlshogi-compatible encoding).

    For board moves: uses Q @ K.T attention to score (from, to) pairs,
    then converts to (direction, destination) format.

    For promotion: uses bilinear scoring conditioned on both Q[from] and
    K[to], trained independently from move selection.

    For drops: learned query vectors (same as v1).

    Output: (B, 2187) = 81 squares × 27 move types
    """

    def __init__(self, cfg: ShogiBT4v2Config):
        super().__init__()
        d = cfg.policy_d_model
        self.cfg = cfg

        # Shared projections for attention
        self.embed = nn.Linear(cfg.embedding_size, d)
        self.act = get_activation(cfg.activation)
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)

        # Drop move embeddings: 7 learned query vectors
        self.drop_queries = nn.Parameter(torch.randn(NUM_DROP_TYPES, d) * 0.02)

        # Build gather indices for ONNX-compatible (from,to) → (direction,to) mapping.
        # For each of the 2187 policy slots, precompute which board_flat indices to
        # gather and take max over (handles sliding pieces with multiple sources).
        MAX_SOURCES = 8  # max sliding distance

        # gather_idx[p][k] = index into board_flat (81*81), or 0 (masked out)
        # gather_mask[p][k] = 1.0 if valid, 0.0 if padding
        gather_idx = torch.zeros(POLICY_SIZE, MAX_SOURCES, dtype=torch.long)
        gather_mask = torch.zeros(POLICY_SIZE, MAX_SOURCES)

        # Build mapping: for each policy slot, collect all (from, to) pairs
        from collections import defaultdict
        policy_to_sources = defaultdict(list)  # policy_idx → [board_flat_idx, ...]

        for from_sq in range(81):
            for to_sq in range(81):
                d_idx = move_to_direction(from_sq, to_sq)
                if d_idx < 0:
                    continue
                flat_idx = from_sq * 81 + to_sq
                from_r = from_sq % 9
                to_r = to_sq % 9

                # Non-promotion
                p = d_idx * 81 + to_sq
                policy_to_sources[p].append(flat_idx)

                # Promotion (if eligible)
                if from_r <= 2 or to_r <= 2:
                    p_promo = (d_idx + NUM_DIRECTIONS) * 81 + to_sq
                    policy_to_sources[p_promo].append(flat_idx)

        for p, sources in policy_to_sources.items():
            for k, src in enumerate(sources[:MAX_SOURCES]):
                gather_idx[p][k] = src
                gather_mask[p][k] = 1.0

        self.register_buffer('gather_idx', gather_idx)
        self.register_buffer('gather_mask', gather_mask)

    def forward(self, x):
        """x: (B, 81, d_model) → (B, 2187) policy logits"""
        d = self.cfg.policy_d_model

        pe = self.act(self.embed(x))
        Q = self.wq(pe)  # (B, 81, d)
        K = self.wk(pe)  # (B, 81, d)

        # Board move logits via attention: (B, 81, 81) = (from, to) scores
        scale = 1.0 / math.sqrt(d)
        board_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale
        board_flat = board_logits.flatten(1)  # (B, 6561)

        # Gather candidate logits for each policy slot: (B, 2187, MAX_SOURCES)
        # gather_idx is (2187, 8) with indices into the 6561-dim board_flat.
        # Advanced indexing: board_flat[:, idx] → (B, 2187, 8). No reshape needed.
        gathered = board_flat[:, self.gather_idx]  # (B, 2187, 8)

        # Apply mask: set padding positions to -inf so they don't affect max
        mask = self.gather_mask.unsqueeze(0)  # (1, 2187, 8)
        gathered = gathered * mask + (1.0 - mask) * (-1e10)

        # Take max over sources: (B, 2187)
        board_policy = gathered.max(dim=2).values

        # Drop logits: drop_queries @ K^T
        dq = self.drop_queries.unsqueeze(0).expand(K.shape[0], -1, -1)
        drop_logits = torch.matmul(dq, K.transpose(-2, -1)) * scale  # (B, 7, 81)

        # Write drops into the policy (slots 1620-2186)
        drop_start = (NUM_DIRECTIONS + NUM_PROMO_DIRECTIONS) * 81
        # Replace the drop portion of board_policy with drop logits
        policy = board_policy.clone()
        policy[:, drop_start:drop_start + NUM_DROP_TYPES * 81] = drop_logits.flatten(1)

        return policy


# =====================================================================
# Value and MLH heads (same as v1)
# =====================================================================

class ValueHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.embedding_size
        self.embed = nn.Linear(d, cfg.value_embedding)
        self.act = get_activation(cfg.activation)
        self.fc1 = nn.Linear(cfg.num_squares * cfg.value_embedding, cfg.value_hidden)
        self.fc2 = nn.Linear(cfg.value_hidden, 3)

    def forward(self, x):
        h = self.act(self.embed(x))
        h = torch.flatten(h, 1)  # (B, 81, emb) -> (B, 81*emb)
        h = self.act(self.fc1(h))
        return self.fc2(h)


class MovesLeftHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.embedding_size
        self.embed = nn.Linear(d, cfg.mlh_embedding)
        self.act = get_activation(cfg.activation)
        self.fc1 = nn.Linear(cfg.num_squares * cfg.mlh_embedding, cfg.mlh_hidden)
        self.fc2 = nn.Linear(cfg.mlh_hidden, 1)

    def forward(self, x):
        h = self.act(self.embed(x))
        h = torch.flatten(h, 1)  # (B, 81, emb) -> (B, 81*emb)
        h = self.act(self.fc1(h))
        return F.relu(self.fc2(h))


# =====================================================================
# Full Model
# =====================================================================

class ShogiBT4v2(nn.Module):
    """
    BT4-v2 Transformer for Shogi.

    Input:  (batch, 48, 9, 9) float tensor
    Output: policy (batch, 2187), wdl (batch, 3), mlh (batch, 1)
    """

    def __init__(self, cfg: ShogiBT4v2Config = None):
        super().__init__()
        if cfg is None:
            cfg = ShogiBT4v2Config()
        self.cfg = cfg

        self.embedding = InputEmbedding(cfg)
        self.smolgen_global = nn.Linear(cfg.smolgen_gen_size, 81 * 81, bias=False)
        self.encoders = nn.ModuleList([
            EncoderBlock(cfg, self.smolgen_global)
            for _ in range(cfg.num_encoders)
        ])
        self.policy_head = DirectionPolicyHead(cfg)
        self.value_head = ValueHead(cfg)
        self.mlh_head = MovesLeftHead(cfg)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.embedding(x)
        for enc in self.encoders:
            h = enc(h)
        policy = self.policy_head(h)
        wdl = self.value_head(h)
        mlh = self.mlh_head(h)
        return policy, wdl, mlh

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
# Quick test
# =====================================================================

if __name__ == "__main__":
    cfg = ShogiBT4v2Config()
    model = ShogiBT4v2(cfg)

    print(f"Model: ShogiBT4v2 (direction-based policy)")
    print(f"  d_model:      {cfg.embedding_size}")
    print(f"  encoders:     {cfg.num_encoders}")
    print(f"  heads:        {cfg.num_heads}")
    print(f"  policy_size:  {cfg.policy_size} (vs v1's 3849)")
    print(f"  Parameters:   {model.count_parameters():,}")

    x = torch.randn(2, cfg.input_planes, 9, 9)
    policy, wdl, mlh = model(x)
    print(f"\nForward pass:")
    print(f"  Input:  {x.shape}")
    print(f"  Policy: {policy.shape} (should be [2, 2187])")
    print(f"  WDL:    {wdl.shape}")
    print(f"  MLH:    {mlh.shape}")

    # Check promotion vs non-promotion for a specific move
    # UP direction (idx 0) to square 10 (file 1, rank 1 = 2b)
    nopromo_idx = 0 * 81 + 10  # direction 0, square 10
    promo_idx = 10 * 81 + 10   # direction 10 (UP_PROMOTE), square 10
    print(f"\n  UP to 2b (non-promo): policy[{nopromo_idx}] = {policy[0, nopromo_idx]:.4f}")
    print(f"  UP to 2b (promo):     policy[{promo_idx}] = {policy[0, promo_idx]:.4f}")
    print(f"  Difference: {policy[0, promo_idx] - policy[0, nopromo_idx]:.4f}")
    print(f"  (Should be ~0 with random init, no systematic bias)")

    # Verify move encoding
    print(f"\nMove encoding test:")
    test_moves = [("7g7f", False), ("8h2b+", False), ("P*5e", False), ("3c3d", True)]
    for move, flip in test_moves:
        idx = make_direction_policy_index(move, flip)
        print(f"  {move} (flip={flip}): index={idx}")
