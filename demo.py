"""Convenience wrapper for the real-footage demo mode.

Equivalent to:
    python main.py --source demo --save-video
"""
from __future__ import annotations

import sys

from main import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not any(item in {"--source", "--input"} for item in argv):
        argv = ["--source", "demo", "--save-video", "--max-frames", "450", *argv]
    raise SystemExit(main(argv))
