"""
BT4-style Transformer Model for Shogi.

Architecture adapted from Leela Chess Zero's BT4:
  - PE-Dense input embedding
  - Encoder stack with multi-head attention + smolgen
  - Attention policy head (88×81 → 3849 moves)
  - WDL value head
  - Moves-left head

Usage:
    model = ShogiBT4(cfg)
    policy, wdl, mlh = model(input_planes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Configuration
# =====================================================================

class ShogiBT4Config:
    """Model hyperparameters."""
    # Board
    board_size: int = 9
    num_squares: int = 81
    input_planes: int = 44
    policy_size: int = 3849

    # Embedding
    embedding_size: int = 256       # d_model (smaller than chess BT4-1024 for faster training)
    embedding_dense_size: int = 128 # PE-dense preprocessing output per square
    num_piece_planes: int = 28      # First 28 planes are piece positions (14 ours + 14 theirs)

    # Encoder
    num_encoders: int = 6           # Number of transformer blocks
    num_heads: int = 8              # Attention heads
    ffn_multiplier: float = 1.5     # FFN hidden = embedding_size * ffn_multiplier
    smolgen_hidden: int = 64        # Smolgen hidden dimension
    smolgen_compress: int = 8       # Smolgen compression size
    smolgen_gen_size: int = 64      # Smolgen generation size per head

    # Policy head
    policy_d_model: int = 256       # Policy attention Q/K dimension
    num_drop_types: int = 7         # Drop piece types
    virtual_drop_squares: int = 7   # Virtual source squares for drops (81..87)

    # Value head
    value_embedding: int = 32
    value_hidden: int = 128

    # MLH head
    mlh_embedding: int = 8
    mlh_hidden: int = 64

    # Activation
    activation: str = "mish"

    @property
    def ffn_hidden(self):
        return int(self.embedding_size * self.ffn_multiplier)

    @property
    def head_dim(self):
        return self.embedding_size // self.num_heads


def get_activation(name):
    if name == "mish":
        return nn.Mish()
    if name == "relu":
        return nn.ReLU()
    if name == "swish" or name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


# =====================================================================
# Building blocks
# =====================================================================

class InputEmbedding(nn.Module):
    """PE-Dense input embedding for Shogi."""

    def __init__(self, cfg: ShogiBT4Config):
        super().__init__()
        self.cfg = cfg
        sq = cfg.num_squares
        n_piece = cfg.num_piece_planes

        # Dense preprocessing: extract piece planes, flatten, project.
        # Input: (batch, 81, num_piece_planes) → (batch, 81*dense_size)
        self.preproc = nn.Linear(sq * n_piece, sq * cfg.embedding_dense_size)

        # Square embedding: (batch, 81, input_planes + dense_size) → (batch, 81, d_model)
        input_c = cfg.input_planes + cfg.embedding_dense_size
        self.embed = nn.Linear(input_c, cfg.embedding_size)
        self.ln = nn.LayerNorm(cfg.embedding_size)
        self.act = get_activation(cfg.activation)

        # Input gating
        self.mult_gate = nn.Parameter(torch.ones(sq, cfg.embedding_size))
        self.add_gate = nn.Parameter(torch.zeros(sq, cfg.embedding_size))

        # Embedding FFN
        self.ffn1 = nn.Linear(cfg.embedding_size, cfg.ffn_hidden)
        self.ffn2 = nn.Linear(cfg.ffn_hidden, cfg.embedding_size)
        self.ffn_act = get_activation(cfg.activation)
        self.ffn_ln = nn.LayerNorm(cfg.embedding_size)

    def forward(self, x):
        """x: (batch, input_planes, 9, 9) → (batch, 81, d_model)"""
        B = x.shape[0]
        sq = self.cfg.num_squares
        n_piece = self.cfg.num_piece_planes

        # Reshape NCHW → (batch, 81, channels)
        x = x.reshape(B, self.cfg.input_planes, sq).permute(0, 2, 1)  # (B, 81, C)

        # PE-dense preprocessing on first 28 (piece) planes
        pos_info = x[:, :, :n_piece].reshape(B, sq * n_piece)         # (B, 81*28)
        pos_proc = self.preproc(pos_info).reshape(B, sq, -1)          # (B, 81, dense)

        # Concatenate: (B, 81, 44) + (B, 81, dense) → (B, 81, 44+dense)
        x = torch.cat([x, pos_proc], dim=-1)

        # Square embedding + LN + activation
        h = self.act(self.embed(x))
        h = self.ln(h)

        # Gating
        h = h * self.mult_gate + self.add_gate

        # Embedding FFN with skip + LN
        f = self.ffn_act(self.ffn1(h))
        f = self.ffn2(f)
        alpha = (2.0 * self.cfg.num_encoders) ** -0.25
        h = self.ffn_ln(f * alpha + h)

        return h  # (B, 81, d_model)


class Smolgen(nn.Module):
    """Smolgen: generates per-position attention biases."""

    def __init__(self, cfg: ShogiBT4Config, global_gen):
        super().__init__()
        d = cfg.embedding_size
        sq = cfg.num_squares

        self.compress = nn.Linear(d, cfg.smolgen_compress)
        self.dense1 = nn.Linear(sq * cfg.smolgen_compress, cfg.smolgen_hidden)
        self.ln1 = nn.LayerNorm(cfg.smolgen_hidden)
        self.dense2 = nn.Linear(cfg.smolgen_hidden, cfg.num_heads * cfg.smolgen_gen_size)
        self.ln2 = nn.LayerNorm(cfg.num_heads * cfg.smolgen_gen_size)
        self.act = nn.SiLU()  # swish
        self.global_gen = global_gen  # shared across encoders
        self.cfg = cfg

    def forward(self, x):
        """x: (B, 81, d_model) → (B, heads, 81, 81) attention bias"""
        B = x.shape[0]
        c = self.compress(x)                        # (B, 81, compress)
        h = c.reshape(B, -1)                        # (B, 81*compress)
        h = self.act(self.ln1(self.dense1(h)))      # (B, hidden)
        h = self.act(self.ln2(self.dense2(h)))      # (B, heads*gen)
        h = h.reshape(B, self.cfg.num_heads, -1)    # (B, heads, gen)
        w = self.global_gen(h)                      # (B, heads, 81*81)
        return w.reshape(B, self.cfg.num_heads, 81, 81)


class EncoderBlock(nn.Module):
    """Transformer encoder block with smolgen."""

    def __init__(self, cfg: ShogiBT4Config, global_gen):
        super().__init__()
        d = cfg.embedding_size

        # MHA
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d)

        # Smolgen
        self.smolgen = Smolgen(cfg, global_gen)

        # FFN
        self.ffn1 = nn.Linear(d, cfg.ffn_hidden)
        self.ffn2 = nn.Linear(cfg.ffn_hidden, d)
        self.ffn_act = get_activation(cfg.activation)
        self.ln2 = nn.LayerNorm(d)

        self.cfg = cfg
        self.alpha = (2.0 * cfg.num_encoders) ** -0.25

    def forward(self, x):
        """x: (B, 81, d_model) → (B, 81, d_model)"""
        B, S, D = x.shape
        heads = self.cfg.num_heads
        depth = D // heads

        # Smolgen bias
        smol_bias = self.smolgen(x)  # (B, heads, 81, 81)

        # MHA
        Q = self.q_proj(x).reshape(B, S, heads, depth).permute(0, 2, 1, 3)
        K = self.k_proj(x).reshape(B, S, heads, depth).permute(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, S, heads, depth).permute(0, 2, 1, 3)

        att = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(depth)
        att = att + smol_bias
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, V)
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        mha = self.out_proj(out)

        # LN1 + skip (DeepNorm)
        h = self.ln1(mha * self.alpha + x)

        # FFN
        f = self.ffn_act(self.ffn1(h))
        f = self.ffn2(f)

        # LN2 + skip (DeepNorm)
        return self.ln2(f * self.alpha + h)


class AttentionPolicyHead(nn.Module):
    """
    Attention-based policy head for Shogi.

    Outputs raw logits for 3849 moves via:
      1. 81×81 from-to attention (non-promotion board moves)
      2. 81×81 from-to attention (promotion board moves)
      3. 7×81 drop attention (7 piece types × 81 destinations)
    """

    def __init__(self, cfg: ShogiBT4Config, attn_policy_map):
        super().__init__()
        d = cfg.embedding_size

        # Policy embedding
        self.embed = nn.Linear(d, cfg.policy_d_model)
        self.act = get_activation(cfg.activation)

        # WQ, WK for board move attention
        self.wq = nn.Linear(cfg.policy_d_model, cfg.policy_d_model)
        self.wk = nn.Linear(cfg.policy_d_model, cfg.policy_d_model)

        # Promotion offset: learn an additive bias for promotion moves.
        # For each destination square, learn a promotion offset.
        self.promo_offset = nn.Linear(cfg.policy_d_model, 1)

        # Drop move embeddings: 7 learned query vectors for drop types.
        self.drop_queries = nn.Parameter(torch.randn(cfg.num_drop_types, cfg.policy_d_model) * 0.02)

        self.cfg = cfg

        # Register the attention policy map as a buffer (not a parameter).
        self.register_buffer('attn_map', torch.tensor(attn_policy_map, dtype=torch.long))

    def forward(self, x):
        """x: (B, 81, d_model) → (B, 3849) policy logits"""
        B = x.shape[0]
        d = self.cfg.policy_d_model
        sq = self.cfg.num_squares

        pe = self.act(self.embed(x))  # (B, 81, pol_d)
        Q = self.wq(pe)               # (B, 81, pol_d)
        K = self.wk(pe)               # (B, 81, pol_d)

        # Board move logits: Q @ K^T / sqrt(d)
        scale = 1.0 / math.sqrt(d)
        board_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, 81, 81)

        # Promotion logits: board_logits + promotion offset per destination
        promo_off = self.promo_offset(K).squeeze(-1)  # (B, 81)
        promo_logits = board_logits + promo_off.unsqueeze(1)  # (B, 81, 81) broadcast

        # Drop logits: drop_queries @ K^T / sqrt(d)
        # drop_queries: (7, pol_d)
        dq = self.drop_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 7, pol_d)
        drop_logits = torch.matmul(dq, K.transpose(-2, -1)) * scale  # (B, 7, 81)

        # Concatenate raw: board(6561) + promo(6561) + drop(567) = 13689
        raw = torch.cat([
            board_logits.reshape(B, -1),   # (B, 6561)
            promo_logits.reshape(B, -1),   # (B, 6561)
            drop_logits.reshape(B, -1),    # (B, 567)
        ], dim=-1)  # (B, 13689)

        # Map to 3849 policy outputs using attention policy map.
        # attn_map has -1 for invalid entries. We gather valid ones.
        valid_mask = self.attn_map >= 0
        valid_raw_indices = torch.where(valid_mask)[0]  # indices into raw
        valid_pol_indices = self.attn_map[valid_mask]    # indices into policy

        policy = torch.full((B, self.cfg.policy_size), -1e10,
                            device=x.device, dtype=x.dtype)
        policy[:, valid_pol_indices] = raw[:, valid_raw_indices]

        return policy


class ValueHead(nn.Module):
    """WDL (Win/Draw/Loss) value head."""

    def __init__(self, cfg: ShogiBT4Config):
        super().__init__()
        d = cfg.embedding_size
        self.embed = nn.Linear(d, cfg.value_embedding)
        self.act = get_activation(cfg.activation)
        self.fc1 = nn.Linear(cfg.num_squares * cfg.value_embedding, cfg.value_hidden)
        self.fc2 = nn.Linear(cfg.value_hidden, 3)  # W, D, L

    def forward(self, x):
        """x: (B, 81, d_model) → (B, 3) WDL logits"""
        h = self.act(self.embed(x))           # (B, 81, val_emb)
        h = h.reshape(x.shape[0], -1)        # (B, 81*val_emb)
        h = self.act(self.fc1(h))             # (B, val_hidden)
        return self.fc2(h)                    # (B, 3) — softmax applied in loss


class MovesLeftHead(nn.Module):
    """Moves-left prediction head."""

    def __init__(self, cfg: ShogiBT4Config):
        super().__init__()
        d = cfg.embedding_size
        self.embed = nn.Linear(d, cfg.mlh_embedding)
        self.act = get_activation(cfg.activation)
        self.fc1 = nn.Linear(cfg.num_squares * cfg.mlh_embedding, cfg.mlh_hidden)
        self.fc2 = nn.Linear(cfg.mlh_hidden, 1)

    def forward(self, x):
        """x: (B, 81, d_model) → (B, 1) moves left"""
        h = self.act(self.embed(x))
        h = h.reshape(x.shape[0], -1)
        h = self.act(self.fc1(h))
        return F.relu(self.fc2(h))  # Non-negative


# =====================================================================
# Full Model
# =====================================================================

class ShogiBT4(nn.Module):
    """
    BT4-style Transformer for Shogi.

    Input:  (batch, 44, 9, 9) float tensor
    Output: policy (batch, 3849), wdl (batch, 3), mlh (batch, 1)
    """

    def __init__(self, cfg: ShogiBT4Config = None, attn_policy_map=None):
        super().__init__()
        if cfg is None:
            cfg = ShogiBT4Config()
        self.cfg = cfg

        # Input embedding
        self.embedding = InputEmbedding(cfg)

        # Shared smolgen weight generator
        self.smolgen_global = nn.Linear(cfg.smolgen_gen_size, 81 * 81, bias=False)

        # Encoder stack
        self.encoders = nn.ModuleList([
            EncoderBlock(cfg, self.smolgen_global)
            for _ in range(cfg.num_encoders)
        ])

        # Heads
        if attn_policy_map is None:
            attn_policy_map = self._default_attn_map()
        self.policy_head = AttentionPolicyHead(cfg, attn_policy_map)
        self.value_head = ValueHead(cfg)
        self.mlh_head = MovesLeftHead(cfg)

        # Initialize weights
        self._init_weights()

    def _default_attn_map(self):
        """Generate attention policy map (same logic as encoder.cc)."""
        # This duplicates the C++ logic in Python for convenience.
        from shogi_policy_gen import generate_attn_policy_map
        return generate_attn_policy_map()

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
        """
        x: (batch, 44, 9, 9)
        Returns: (policy_logits, wdl_logits, mlh)
        """
        h = self.embedding(x)       # (B, 81, d_model)

        for enc in self.encoders:
            h = enc(h)              # (B, 81, d_model)

        policy = self.policy_head(h) # (B, 3849)
        wdl = self.value_head(h)     # (B, 3)
        mlh = self.mlh_head(h)       # (B, 1)

        return policy, wdl, mlh

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_flops(self, batch_size=1):
        """Rough FLOP estimate (MACs)."""
        d = self.cfg.embedding_size
        sq = self.cfg.num_squares
        ffn_h = self.cfg.ffn_hidden
        n_enc = self.cfg.num_encoders
        heads = self.cfg.num_heads
        depth = d // heads

        flops = 0
        # Embedding
        flops += sq * self.cfg.num_piece_planes * sq * self.cfg.embedding_dense_size
        flops += sq * (self.cfg.input_planes + self.cfg.embedding_dense_size) * d
        flops += sq * d * ffn_h * 2

        # Encoders
        per_enc = (3 * sq * d * d +         # Q,K,V
                   heads * sq * depth * sq + # QK^T
                   heads * sq * sq * depth +  # att*V
                   sq * d * d +              # output proj
                   sq * d * ffn_h * 2)       # FFN
        flops += n_enc * per_enc

        # Policy
        flops += sq * d * self.cfg.policy_d_model * 3

        return flops


# =====================================================================
# Policy map generator (standalone)
# =====================================================================

def generate_attn_policy_map():
    """Generate the 13689 → 3849 attention policy map."""
    BOARD = 9
    RAW_BOARD = 81 * 81
    RAW_PROMO = 81 * 81
    RAW_DROP = 7 * 81

    def in_bounds(f, r):
        return 0 <= f < BOARD and 0 <= r < BOARD

    # Movement definitions for BLACK
    piece_moves = {
        'pawn':   [(0,-1,1)],
        'lance':  [(0,-1,8)],
        'knight': [(-1,-2,1),(1,-2,1)],
        'silver': [(0,-1,1),(-1,-1,1),(1,-1,1),(-1,1,1),(1,1,1)],
        'gold':   [(0,-1,1),(-1,-1,1),(1,-1,1),(-1,0,1),(1,0,1),(0,1,1)],
        'bishop': [(-1,-1,8),(-1,1,8),(1,-1,8),(1,1,8)],
        'rook':   [(0,-1,8),(0,1,8),(-1,0,8),(1,0,8)],
        'king':   [(df,dr,1) for df in (-1,0,1) for dr in (-1,0,1) if (df,dr)!=(0,0)],
        'horse':  [(-1,-1,8),(-1,1,8),(1,-1,8),(1,1,8),(0,-1,1),(0,1,1),(-1,0,1),(1,0,1)],
        'dragon': [(0,-1,8),(0,1,8),(-1,0,8),(1,0,8),(-1,-1,1),(-1,1,1),(1,-1,1),(1,1,1)],
    }

    valid_pairs = set()
    for moves in piece_moves.values():
        for f in range(9):
            for r in range(9):
                for df, dr, md in moves:
                    for dist in range(1, md+1):
                        nf, nr = f+df*dist, r+dr*dist
                        if not in_bounds(nf, nr): break
                        valid_pairs.add((f*9+r, nf*9+nr))

    promo_pairs = {(f,t) for f,t in valid_pairs if f%9<=2 or t%9<=2}

    move_list = []
    board_idx = {}
    promo_idx = {}
    drop_idx = {}

    for f, t in sorted(valid_pairs):
        board_idx[(f,t)] = len(move_list)
        move_list.append(('board', f, t, False))

    for f, t in sorted(promo_pairs):
        promo_idx[(f,t)] = len(move_list)
        move_list.append(('board', f, t, True))

    for pt in range(7):
        for sq in range(81):
            drop_idx[(pt,sq)] = len(move_list)
            move_list.append(('drop', pt, sq))

    attn_map = [-1] * (RAW_BOARD + RAW_PROMO + RAW_DROP)

    for (f,t), idx in board_idx.items():
        attn_map[f*81+t] = idx

    for (f,t), idx in promo_idx.items():
        attn_map[RAW_BOARD + f*81+t] = idx

    for (pt,sq), idx in drop_idx.items():
        attn_map[RAW_BOARD + RAW_PROMO + pt*81+sq] = idx

    return attn_map


# =====================================================================
# Quick test
# =====================================================================

if __name__ == "__main__":
    # Generate policy map
    attn_map = generate_attn_policy_map()
    valid = sum(1 for x in attn_map if x >= 0)
    print(f"Policy map: {len(attn_map)} raw → {valid} valid moves")

    cfg = ShogiBT4Config()
    model = ShogiBT4(cfg, attn_map)

    print(f"\nModel: ShogiBT4")
    print(f"  d_model:    {cfg.embedding_size}")
    print(f"  encoders:   {cfg.num_encoders}")
    print(f"  heads:      {cfg.num_heads}")
    print(f"  FFN hidden: {cfg.ffn_hidden}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  FLOPs:      {model.count_flops()/1e6:.1f}M")

    # Test forward pass
    x = torch.randn(2, 44, 9, 9)
    policy, wdl, mlh = model(x)
    print(f"\nForward pass:")
    print(f"  Input:  {x.shape}")
    print(f"  Policy: {policy.shape}")
    print(f"  WDL:    {wdl.shape}")
    print(f"  MLH:    {mlh.shape}")

    # Check policy has valid values at mapped positions
    valid_mask = torch.tensor(attn_map) >= 0
    valid_count = valid_mask.sum()
    invalid_in_policy = (policy[0] < -1e9).sum()
    print(f"  Policy valid entries:   {cfg.policy_size - invalid_in_policy.item()}")
    print(f"  Policy invalid (-1e10): {invalid_in_policy.item()}")

    print("\nModel test passed!")
