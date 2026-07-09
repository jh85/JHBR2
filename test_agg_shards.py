"""
Smoke tests for gen_agg_shards.py + the aggregated-shard training path.

Covers (see JHBR2 audit):
  1. Synthetic .pack with known duplicates -> exact policy/value/MLH
     aggregation math (temperature alpha=0.5 and count methods).
  2. Real .pack sample -> shard schema, key uniqueness (dedup), policy
     normalization, every nonzero policy entry is a legal move, WDL/MLH
     ranges and conventions.
  3. Real .bin (PSV) sample -> same checks + MLH masked (-1) by default,
     PSV->cshogi move conversion produces legal moves.
  4. ShardedDataset loads aggregated shards; a forward pass + one full
     training-loss step (soft policy CE + WDL CE + masked MLH Huber) runs
     and is finite. Regression: hard-label (gen_pack_shards) shards still
     load and train.

Usage:
    python test_agg_shards.py --pack-file ../kif_*.pack --psv-file ../*.bin
    python test_agg_shards.py            # synthetic + loader tests only
"""

import argparse
import glob
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import cshogi
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shogi_train import ShardedDataset, move_to_policy_index
from shogi_model_v2 import ShogiBT4v2, ShogiBT4v2Config, POLICY_SIZE

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def run_agg(out_dir, extra):
    cmd = [sys.executable, os.path.join(HERE, "gen_agg_shards.py"),
           "--output-dir", out_dir, "--workers", "1", "--partitions", "4",
           "--shard-size", "100000", "--store-key"] + extra
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise RuntimeError("gen_agg_shards.py failed")
    return r.stdout


def load_all(out_dir):
    """Concatenate every aggshard in out_dir into one dict of arrays."""
    paths = sorted(glob.glob(os.path.join(out_dir, "aggshard_*.npz")))
    assert paths, f"no shards in {out_dir}"
    shards = [dict(np.load(p)) for p in paths]
    out = {}
    for k in shards[0]:
        if k == "policy_offsets":
            # Re-base ragged offsets when concatenating.
            off = [np.zeros(1, dtype=np.int64)]
            base = 0
            for s in shards:
                off.append(s[k][1:] + base)
                base += s[k][-1]
            out[k] = np.concatenate(off)
        else:
            out[k] = np.concatenate([s[k] for s in shards])
    return out


def policy_of(data, i):
    s, e = data["policy_offsets"][i], data["policy_offsets"][i + 1]
    return data["policy_indices"][s:e], data["policy_weights"][s:e]


# =====================================================================
# 1. Synthetic pack: exact aggregation math
# =====================================================================

def write_synth_pack(path, games):
    """games: list of (usi_moves, result_abs, eval_cp). Startpos games."""
    board = cshogi.Board()
    data = bytearray()
    for moves, result, ev in games:
        board.reset()
        data.append(1)                       # startpos marker
        for u in moves:
            m16 = board.move_from_usi(u) & 0xFFFF
            data += m16.to_bytes(2, "little")
            data += int(ev).to_bytes(2, "little", signed=True)
            board.push_usi(u)
        data += (result + (result << 7)).to_bytes(2, "little")
        data.append(0)                       # game-end reason
    with open(path, "wb") as f:
        f.write(data)


def test_synthetic(tmp):
    print("\n[1] synthetic pack: aggregation math")
    pack_dir = os.path.join(tmp, "synth_pack")
    os.makedirs(pack_dir)
    # startpos: 7g7f x2, 2g2f x1.  Position-after-7g7f: 3c3d x1, 8c8d x1.
    games = [
        (["7g7f", "3c3d"], 1, 120),   # black wins
        (["7g7f", "8c8d"], 2, 120),   # white wins
        (["2g2f", "3c3d"], 0, 120),   # draw
    ]
    write_synth_pack(os.path.join(pack_dir, "synth.pack"), games)

    out = os.path.join(tmp, "synth_out")
    run_agg(out, ["--pack-dir", pack_dir, "--policy-method", "temperature",
                  "--alpha", "0.5"])
    data = load_all(out)

    # 6 raw records -> 4 unique positions (startpos x3, after-7g7f x2,
    # after-2g2f x1, after 7g7f+3c3d and after 2g2f+3c3d... note the last
    # move of each game IS recorded, so positions after ply1 also exist).
    startpos_hcp = np.empty(1, dtype=cshogi.HuffmanCodedPos)
    cshogi.Board().to_hcp(startpos_hcp)
    keys = data["key"]
    sp = np.where((keys == np.frombuffer(startpos_hcp.tobytes(),
                                         dtype=np.uint8)).all(axis=1))[0]
    check("startpos present exactly once", len(sp) == 1, f"got {len(sp)}")
    i = int(sp[0])
    check("startpos count == 3", int(data["count"][i]) == 3)

    idx_7g7f = move_to_policy_index("7g7f", False)
    idx_2g2f = move_to_policy_index("2g2f", False)
    pi, pw = policy_of(data, i)
    got = {int(a): float(b) for a, b in zip(pi, pw)}
    check("startpos keeps BOTH observed moves", set(got) == {idx_7g7f,
                                                             idx_2g2f})
    # temperature alpha=0.5: sqrt(2):sqrt(1) -> 0.5858 / 0.4142
    w_expect = math.sqrt(2) / (math.sqrt(2) + 1)
    check("temperature(0.5) weights",
          abs(got[idx_7g7f] - w_expect) < 2e-3 and
          abs(got[idx_2g2f] - (1 - w_expect)) < 2e-3, f"got {got}")
    check("weights normalized", abs(sum(got.values()) - 1.0) < 2e-3)
    check("top-1 policy == heaviest move",
          int(data["policy"][i]) == idx_7g7f)

    # value: eval +120 stm-black each time; results +1, -1, 0 -> hard (1,1,1)/3
    wr = 1.0 / (1.0 + math.exp(-120 / 600.0))
    wdl_expect = (0.7 * wr + 0.1, 0.1, 0.7 * (1 - wr) + 0.1)
    check("wdl = mean blended target",
          np.allclose(data["wdl"][i].astype(np.float64), wdl_expect,
                      atol=2e-3), f"{data['wdl'][i]} vs {wdl_expect}")
    # mlh: 2-ply games, startpos at i=0 -> n_moves - i - 1 = 1 in each game
    check("mlh = mean remaining plies (1.0)",
          abs(float(data["mlh"][i]) - 1.0) < 1e-3, data["mlh"][i])

    # count method on the same input
    out2 = os.path.join(tmp, "synth_out_count")
    run_agg(out2, ["--pack-dir", pack_dir, "--policy-method", "count"])
    d2 = load_all(out2)
    sp2 = np.where((d2["key"] == np.frombuffer(
        startpos_hcp.tobytes(), dtype=np.uint8)).all(axis=1))[0]
    pi2, pw2 = policy_of(d2, int(sp2[0]))
    got2 = {int(a): float(b) for a, b in zip(pi2, pw2)}
    check("count weights = 2/3, 1/3",
          abs(got2[idx_7g7f] - 2 / 3) < 2e-3 and
          abs(got2[idx_2g2f] - 1 / 3) < 2e-3, f"got {got2}")


# =====================================================================
# 2/3. Real data: schema + legality + conventions
# =====================================================================

def verify_shard_semantics(data, label, expect_mlh):
    n = len(data["policy"])
    print(f"  ({label}: {n} aggregated samples, "
          f"{int(data['count'].sum())} raw records)")
    check("planes shape/dtype",
          data["planes"].shape[1:] == (148, 9, 9) and
          data["planes"].dtype == np.float16)
    check("policy dtype int32 / indices int16 / weights f16 / offsets i64",
          data["policy"].dtype == np.int32 and
          data["policy_indices"].dtype == np.int16 and
          data["policy_weights"].dtype == np.float16 and
          data["policy_offsets"].dtype == np.int64)
    check("wdl shape/dtype", data["wdl"].shape == (n, 3) and
          data["wdl"].dtype == np.float16)
    check("mlh dtype f16", data["mlh"].dtype == np.float16)
    check("offsets consistent",
          data["policy_offsets"][-1] == len(data["policy_indices"]) and
          np.all(np.diff(data["policy_offsets"]) >= 1))

    # dedup: canonical keys unique across ALL shards
    keys = data["key"]
    uniq = np.unique(keys.view([("", "V32")]))
    check("canonical keys unique (dedup complete)", len(uniq) == n,
          f"{len(uniq)} != {n}")

    # per-sample checks
    all_norm = True
    all_legal = True
    all_range = True
    top1_ok = True
    board = cshogi.Board()
    for i in range(n):
        pi, pw = policy_of(data, i)
        if abs(float(pw.astype(np.float64).sum()) - 1.0) > 5e-2:
            all_norm = False
        if not (np.all(pi >= 0) and np.all(pi < POLICY_SIZE)):
            all_range = False
        board.set_hcp(np.frombuffer(keys[i].tobytes(),
                                    dtype=cshogi.HuffmanCodedPos))
        flip = board.turn == cshogi.WHITE
        legal = {move_to_policy_index(cshogi.move_to_usi(m), flip)
                 for m in board.legal_moves}
        if not set(int(x) for x in pi) <= legal:
            all_legal = False
        if int(data["policy"][i]) != int(pi[int(np.argmax(pw))]):
            top1_ok = False
    check("policy weights normalized (all samples)", all_norm)
    check("policy indices in [0, 2187)", all_range)
    check("all nonzero policy entries are LEGAL moves", all_legal)
    check("top-1 'policy' matches argmax of weights", top1_ok)

    w = data["wdl"].astype(np.float64)
    check("wdl rows sum to 1, entries in [0,1]",
          np.allclose(w.sum(axis=1), 1.0, atol=5e-3) and
          w.min() >= 0 and w.max() <= 1)
    if expect_mlh:
        check("mlh present and >= 0 (pack has true game ends)",
              float(data["mlh"].min()) >= 0)
    else:
        check("mlh masked to -1 (PSV without --psv-mlh)",
              np.all(data["mlh"] == -1.0))


def test_real_pack(tmp, pack_file):
    print("\n[2] real .pack sample")
    pack_dir = os.path.join(tmp, "pack_in")
    os.makedirs(pack_dir)
    os.symlink(os.path.abspath(pack_file),
               os.path.join(pack_dir, "sample.pack"))
    out = os.path.join(tmp, "pack_out")
    stdout = run_agg(out, ["--pack-dir", pack_dir, "--limit", "2000"])
    print("\n".join("  | " + l for l in stdout.splitlines()
                    if "startpos" in l or "duplicate" in l or
                    "unique" in l or "raw records" in l))
    verify_shard_semantics(load_all(out), "pack", expect_mlh=True)
    return out


def test_real_psv(tmp, psv_file):
    print("\n[3] real .bin (PSV) sample")
    psv_dir = os.path.join(tmp, "psv_in")
    os.makedirs(psv_dir)
    os.symlink(os.path.abspath(psv_file), os.path.join(psv_dir, "sample.bin"))
    out = os.path.join(tmp, "psv_out")
    run_agg(out, ["--psv-dir", psv_dir, "--limit", "2000"])
    verify_shard_semantics(load_all(out), "psv", expect_mlh=False)
    return out


# =====================================================================
# 4. Loader + model + one training step
# =====================================================================

def tiny_model():
    cfg = ShogiBT4v2Config()
    cfg.embedding_size = 64
    cfg.policy_d_model = 64
    cfg.embedding_dense_size = 32
    cfg.num_encoders = 1
    cfg.num_heads = 2
    return ShogiBT4v2(cfg)


def train_step(dataset, soft_expected):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    planes, policy_t, wdl_t, mlh_t = next(iter(loader))
    check("planes batch shape", planes.shape[1:] == (148, 9, 9))
    if soft_expected:
        check("loader yields dense soft policy (B, 2187) float",
              policy_t.dim() == 2 and policy_t.shape[1] == POLICY_SIZE and
              policy_t.is_floating_point())
        check("soft targets normalized",
              torch.allclose(policy_t.sum(-1),
                             torch.ones(len(policy_t)), atol=5e-2))
    else:
        check("loader yields hard policy labels (B,) long",
              policy_t.dim() == 1 and policy_t.dtype == torch.long)

    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    policy_logits, wdl_logits, mlh = model(planes)
    check("forward shapes",
          policy_logits.shape == (len(planes), POLICY_SIZE) and
          wdl_logits.shape == (len(planes), 3) and
          mlh.shape == (len(planes), 1))

    # exactly the shogi_train.py loss
    if policy_t.dim() > 1:
        has = policy_t.sum(-1) > 0
        logp = F.log_softmax(policy_logits[has].float(), dim=-1)
        policy_loss = -(policy_t[has] * logp).sum(-1).mean()
    else:
        has = policy_t >= 0
        policy_loss = F.cross_entropy(policy_logits[has], policy_t[has])
    value_loss = F.cross_entropy(wdl_logits, wdl_t)
    mask = mlh_t >= 0
    mlh_loss = (F.smooth_l1_loss(mlh[mask].squeeze(-1),
                                 torch.clamp(mlh_t[mask], max=80.0))
                if mask.any() else torch.tensor(0.0))
    loss = policy_loss + value_loss + 0.1 * mlh_loss
    loss.backward()
    opt.step()
    check("loss finite after one optimizer step",
          torch.isfinite(loss).item(),
          f"p={policy_loss.item():.3f} v={value_loss.item():.3f} "
          f"m={mlh_loss.item():.3f}")


def test_loader_and_training(tmp, agg_dir, pack_file):
    print("\n[4] ShardedDataset + model forward + loss step (aggregated)")
    ds = ShardedDataset(os.path.join(agg_dir, "aggshard"))
    check("dataset detects sparse policy", ds.sparse_policy)
    train_step(ds, soft_expected=True)

    if pack_file:
        print("\n[4b] regression: gen_pack_shards one-hot shards still work")
        pack_dir = os.path.join(tmp, "pack_in")   # created by test 2
        out = os.path.join(tmp, "onehot_out")
        os.makedirs(out, exist_ok=True)
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "gen_pack_shards.py"),
             "--pack-dir", pack_dir, "--output-dir", out,
             "--limit", "300", "--shard-size", "100000", "--workers", "1"],
            cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            check("gen_pack_shards run", False)
            return
        ds2 = ShardedDataset(os.path.join(out, "shard"))
        check("dataset detects hard policy", not ds2.sparse_policy)
        train_step(ds2, soft_expected=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-file", default=None,
                    help="A real .pack file for tests 2/4b")
    ap.add_argument("--psv-file", default=None,
                    help="A real PSV .bin file for test 3")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="test_agg_")
    print(f"tmp: {tmp}")
    try:
        test_synthetic(tmp)
        agg_dir = None
        if args.pack_file:
            agg_dir = test_real_pack(tmp, args.pack_file)
        if args.psv_file:
            psv_dir = test_real_psv(tmp, args.psv_file)
            agg_dir = agg_dir or psv_dir
        test_loader_and_training(tmp, agg_dir or
                                 os.path.join(tmp, "synth_out"),
                                 args.pack_file)
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
