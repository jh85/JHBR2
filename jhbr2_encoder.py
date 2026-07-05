"""
Fast C++ board encoder for shard generation (ctypes wrapper around
pyext/libjhbr2_encoder.so, which calls the verified EncodeShogiPosition).

~100x faster than the pure-Python sfen_to_planes(); produces bit-identical
planes. Build the .so first:  bash pyext/build.sh

Usage:
    import jhbr2_encoder
    planes = jhbr2_encoder.encode_sfens([sfen1, sfen2, ...])  # (N,148,9,9) f32
"""

import ctypes
import os

import numpy as np

_LIB = None
NUM_PLANES = 148
NUM_F1 = 28              # features1 (positional bitmaps)
NUM_F2 = 120            # features2 (uniform one-hot, broadcast)
PACKED_F1_BYTES = 284   # ceil(28*81 / 8)
PACKED_F2_BYTES = 15    # ceil(120 / 8)


def available():
    """True if the compiled encoder .so can be loaded."""
    try:
        _load()
        return True
    except OSError:
        return False


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    so = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "pyext", "libjhbr2_encoder.so")
    lib = ctypes.CDLL(so)
    lib.jhbr2_init.restype = None
    lib.jhbr2_init.argtypes = []
    lib.jhbr2_num_planes.restype = ctypes.c_int
    lib.jhbr2_num_planes.argtypes = []
    lib.jhbr2_encode_sfens.restype = ctypes.c_int
    lib.jhbr2_encode_sfens.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    for fn in ("jhbr2_packed_f1_bytes", "jhbr2_packed_f2_bytes"):
        getattr(lib, fn).restype = ctypes.c_int
        getattr(lib, fn).argtypes = []
    lib.jhbr2_pack_sfens.restype = ctypes.c_int
    lib.jhbr2_pack_sfens.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte),
    ]
    lib.jhbr2_init()
    n = lib.jhbr2_num_planes()
    if n != NUM_PLANES:
        raise RuntimeError(f"encoder plane count {n} != {NUM_PLANES}; rebuild "
                           f"pyext/libjhbr2_encoder.so")
    _LIB = lib
    return _LIB


def encode_sfens(sfens):
    """Encode a list of SFEN strings -> (N, 148, 9, 9) float32 array."""
    lib = _load()
    n = len(sfens)
    if n == 0:
        return np.empty((0, NUM_PLANES, 9, 9), dtype=np.float32)
    arr = (ctypes.c_char_p * n)(*[s.encode("ascii") for s in sfens])
    out = np.empty((n, NUM_PLANES, 9, 9), dtype=np.float32)
    lib.jhbr2_encode_sfens(arr, n,
                           out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return out


def encode_sfen(sfen):
    """Encode a single SFEN -> (148, 9, 9) float32 array."""
    return encode_sfens([sfen])[0]


def pack_sfens(sfens):
    """Encode a list of SFENs -> bit-packed (packed1 (N,284), packed2 (N,15))
    uint8 arrays. Needs the compiled .so. Expand later with unpack_planes()."""
    lib = _load()
    n = len(sfens)
    p1 = np.empty((n, PACKED_F1_BYTES), dtype=np.uint8)
    p2 = np.empty((n, PACKED_F2_BYTES), dtype=np.uint8)
    if n:
        arr = (ctypes.c_char_p * n)(*[s.encode("ascii") for s in sfens])
        lib.jhbr2_pack_sfens(arr, n,
                             p1.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                             p2.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
    return p1, p2


def unpack_planes(packed1, packed2):
    """Expand bit-packed features -> float32 planes. PURE NUMPY — no .so needed,
    so training machines can load packed shards without building the encoder.

    packed1: (..., 284) uint8, packed2: (..., 15) uint8
    returns: (..., 148, 9, 9) float32   (bit-identical to sfen_to_planes)
    """
    packed1 = np.asarray(packed1, dtype=np.uint8)
    packed2 = np.asarray(packed2, dtype=np.uint8)
    lead = packed1.shape[:-1]                       # () for single, (N,) for batch
    # features1: 1 bit per square (LSB-first within each byte, matching the packer)
    b1 = np.unpackbits(packed1, axis=-1, bitorder="little")[..., :NUM_F1 * 81]
    f1 = b1.reshape(lead + (NUM_F1, 81)).astype(np.float32)
    # features2: 1 bit per plane, broadcast across all 81 squares
    b2 = np.unpackbits(packed2, axis=-1, bitorder="little")[..., :NUM_F2]
    f2 = np.broadcast_to(b2[..., None], lead + (NUM_F2, 81)).astype(np.float32)
    planes = np.concatenate([f1, f2], axis=-2)      # (..., 148, 81)
    return planes.reshape(lead + (NUM_PLANES, 9, 9))
