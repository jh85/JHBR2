"""
PSV (Packed Sfen Value) Dataset for PyTorch training.

Reads YaneuraOu's PackedSfenValue binary files directly.
Each 40-byte record: packed_sfen(32) + score(i16) + move(u16) + ply(u16) + result(i8) + pad(1)

Uses C extension (psv_decode_c.so) for fast decoding (~100x faster than Python).

Usage:
    dataset = PSVDataset("/path/to/shard.bin", max_positions=10000000)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
"""

import ctypes
import math
import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset

# Constants (only used by Python fallback and test)
RECORD_SIZE = 40

# =====================================================================
# Load C decoder
# =====================================================================

_c_lib = None

def _get_c_decoder():
    global _c_lib
    if _c_lib is not None:
        return _c_lib

    # Find and load the shared library
    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "psv_decode_c.so")
    if not os.path.exists(so_path):
        # Try to build it
        c_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "psv_decode_c.c")
        if os.path.exists(c_path):
            import subprocess
            print(f"Building {so_path}...")
            subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-o", so_path, c_path, "-lm"],
                           check=True)
        else:
            return None

    lib = ctypes.CDLL(so_path)

    # int decode_psv_batch(const uint8_t* records, int n,
    #                      float* planes, int* policy_idxs, float* wdls, float eval_coef)
    lib.decode_psv_batch.restype = ctypes.c_int
    lib.decode_psv_batch.argtypes = [
        ctypes.c_void_p,  # records
        ctypes.c_int,     # n
        ctypes.c_void_p,  # planes
        ctypes.c_void_p,  # policy_idxs
        ctypes.c_void_p,  # wdls
        ctypes.c_float,   # eval_coef
    ]

    _c_lib = lib
    return lib


# All decoding is done in C (psv_decode_c.so).
# Build: gcc -O3 -shared -fPIC -o psv_decode_c.so psv_decode_c.c -lm

class PSVDataset(Dataset):
    """
    PyTorch Dataset that reads PSV binary files directly.
    Uses C extension for fast decoding (~100x faster than Python).
    """

    def __init__(self, psv_path, max_positions=None, eval_coef=600.0):
        self.psv_path = psv_path
        self.eval_coef = eval_coef
        self.c_lib = _get_c_decoder()

        file_size = os.path.getsize(psv_path)
        total_records = file_size // RECORD_SIZE
        if max_positions:
            total_records = min(total_records, max_positions)
        self.num_records = total_records

        self.data = np.memmap(psv_path, dtype=np.uint8, mode='r',
                              shape=(self.num_records, RECORD_SIZE))

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        rec = bytes(self.data[idx])
        rec_arr = np.frombuffer(rec, dtype=np.uint8).copy()

        if self.c_lib is not None:
            # Fast C decoder
            planes = np.zeros(48 * 81, dtype=np.float32)
            policy_idx = ctypes.c_int(-1)
            wdl = np.zeros(3, dtype=np.float32)

            self.c_lib.decode_psv_record(
                rec_arr.ctypes.data_as(ctypes.c_void_p),
                planes.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(policy_idx),
                wdl.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_float(self.eval_coef))

            return (torch.from_numpy(planes.reshape(48, 9, 9)),
                    torch.tensor(policy_idx.value, dtype=torch.long),
                    torch.from_numpy(wdl))
        else:
            raise RuntimeError("C decoder not available. Build with: "
                               "gcc -O3 -shared -fPIC -o psv_decode_c.so psv_decode_c.c -lm")


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
