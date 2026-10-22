#!/usr/bin/env python3
"""Run one stage or the complete reproducible experiment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    stages = ("prepare", "baseline", "cnn", "bert", "analyze", "steer", "rewrite", "report")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*stages, "all"))
    args = parser.parse_args()
    if args.stage == "all":
        for stage in stages:
            subprocess.run([sys.executable, str(__file__), stage], check=True, cwd=ROOT)
        return
    from imdb_lab.common import config, environment, save_json, seed_all
    cfg = config()
    seed_all(cfg["seed"], cfg["threads"])
    status = ROOT / "logs" / f"{args.stage}_status.json"
    save_json(status, {"pid": os.getpid(), "stage": args.stage, "status": "running", "config": cfg})
    try:
        if args.stage == "prepare":
            from imdb_lab.data import prepare
            prepare()
        elif args.stage == "baseline":
            from imdb_lab.baseline import run
            run()
        elif args.stage in ("cnn", "bert"):
            from imdb_lab.training import run
            run(args.stage)
        else:
            import importlib
            module = {"analyze": "analysis", "steer": "steering", "rewrite": "rewrite", "report": "report"}[args.stage]
            importlib.import_module(f"imdb_lab.{module}").run()
        save_json(status, {"pid": os.getpid(), "stage": args.stage, "status": "complete", "environment": environment()})
    except Exception:
        save_json(status, {"pid": os.getpid(), "stage": args.stage, "status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
