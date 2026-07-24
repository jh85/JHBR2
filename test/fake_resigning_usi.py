#!/usr/bin/env python3
"""Minimal deterministic USI engine used to smoke-test the match harness."""

import sys


for raw_line in sys.stdin:
    command = raw_line.strip()
    if command == "usi":
        print("id name fake-resigning-usi")
        print("id author JHBR2 test")
        print("usiok", flush=True)
    elif command == "isready":
        print("readyok", flush=True)
    elif command.startswith("go"):
        print("info depth 1 score cp 0 nodes 1 nps 1 time 1")
        print("bestmove resign", flush=True)
    elif command == "quit":
        break
