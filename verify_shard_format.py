"""
Prove the packed shard format is bit-identical to the trusted float16 format.

Generates both formats from the SAME positions and asserts planes / policy /
wdl / mlh match exactly (planes via the per-item unpack the DataLoader uses).
If this passes, training on packed shards is provably equivalent to training on
the float16 shards you've already validated — no long training run needed.

Usage:
    python verify_shard_format.py --pack-dir data --n 50000
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jhbr2_encoder


def gen(pack_dir, out_dir, n, packed):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "gen_pack_shards.py", "--pack-dir", pack_dir,
           "--output-dir", out_dir, "--limit", str(n), "--shard-size", str(n),
           "--workers", "1"]
    if packed:
        cmd.append("--packed")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    shards = sorted(f for f in os.listdir(out_dir) if f.endswith(".npz"))
    return np.load(os.path.join(out_dir, shards[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--n", type=int, default=50000, help="positions to compare")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="verify_shard_")
    try:
        print(f"Generating {args.n} positions in both formats ...")
        flt = gen(args.pack_dir, os.path.join(tmp, "float"), args.n, packed=False)
        pkd = gen(args.pack_dir, os.path.join(tmp, "packed"), args.n, packed=True)

        n = len(flt["policy"])
        assert len(pkd["policy"]) == n, "different position counts"
        print(f"Comparing {n} positions ...")

        # metadata must be identical
        for k in ("policy", "wdl", "mlh"):
            assert np.array_equal(flt[k], pkd[k]), f"{k} differs!"
        print("  policy / wdl / mlh: identical")

        # planes: unpack packed per-item (as the DataLoader does) vs float16
        planes_f = flt["planes"].astype(np.float32)
        p1, p2 = pkd["packed1"], pkd["packed2"]
        mism = 0
        for i in range(n):
            up = jhbr2_encoder.unpack_planes(p1[i], p2[i])   # (148,9,9) f32
            if not np.array_equal(up, planes_f[i]):
                mism += 1
                if mism <= 3:
                    print(f"  MISMATCH at position {i}")
        # storage comparison
        fsz = os.path.getsize(os.path.join(tmp, "float",
              sorted(os.listdir(os.path.join(tmp, "float")))[0]))
        psz = os.path.getsize(os.path.join(tmp, "packed",
              sorted(os.listdir(os.path.join(tmp, "packed")))[0]))

        print(f"  planes: {n - mism}/{n} bit-identical")
        print(f"\nshard file size: float16 {fsz/1e6:.1f} MB  |  packed {psz/1e6:.1f} MB "
              f"({fsz/psz:.2f}x)")
        if mism == 0:
            print("\n✅ PASS — packed format is bit-identical to the float16 format.")
            return 0
        print(f"\n❌ FAIL — {mism} planes mismatched.")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
