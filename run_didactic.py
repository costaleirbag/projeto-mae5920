#!/usr/bin/env python3
"""Run the course-linked teaching extension without touching the old results."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(ROOT / "src"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "toys", "audit", "report"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    if args.stage == "toys":
        from imdb_lab.didactic_toys import run
        run()
    elif args.stage == "report":
        from imdb_lab.didactic_report import run
        run()
    else:
        from imdb_lab import didactic
        if args.stage == "prepare": didactic.prepare()
        elif args.stage == "train": didactic.train_all(args.device)
        elif args.stage == "audit": didactic.audit(args.device)

if __name__ == "__main__":
    main()
