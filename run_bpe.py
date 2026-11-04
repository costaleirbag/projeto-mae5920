#!/usr/bin/env python3
"""Controlled word/BPE TextCNN ablation, isolated from earlier frozen studies."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "audit", "replay", "verify", "report"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from imdb_lab import bpe_study as lab
    if args.stage in {"train", "audit", "replay"}: getattr(lab, args.stage)(args.device)
    elif args.stage == "report":
        from imdb_lab.bpe_report import build
        build()
    else: getattr(lab, args.stage)()


if __name__ == "__main__": main()
