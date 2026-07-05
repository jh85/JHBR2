#!/bin/bash
# Build the fast C++ encoder shared library used by gen_pack_shards.py.
# Requires g++ (C++20). No pybind11 / external deps.
set -e
cd "$(dirname "$0")/.."   # repo root

CXX="${CXX:-g++}"
OUT="pyext/libjhbr2_encoder.so"

$CXX -O3 -march=native -std=c++20 -shared -fPIC -I. \
    pyext/jhbr2_encoder_capi.cc \
    shogi/board.cc shogi/bitboard.cc shogi/encoder.cc \
    -o "$OUT"

echo "Built $OUT"
