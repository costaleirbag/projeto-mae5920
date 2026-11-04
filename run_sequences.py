#!/usr/bin/env python3
"""CNN/encoder teaching extension; immutable earlier experiments are untouched."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "pilot", "train", "audit", "report"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", type=Path, default=Path(".cache/sequence_pretrained"))
    args = parser.parse_args()
    from imdb_lab import sequences as lab
    if args.stage == "prepare": lab.prepare()
    elif args.stage == "pilot": lab.pilot(args.pretrained, args.device)
    elif args.stage == "train": lab.train_all(args.pretrained, args.device)
    elif args.stage == "audit": lab.audit(args.pretrained, args.device)
    else:
        from imdb_lab.sequence_report import build
        build()

if __name__ == "__main__": main()
