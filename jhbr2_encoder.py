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
