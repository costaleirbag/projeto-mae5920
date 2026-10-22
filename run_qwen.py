#!/usr/bin/env python3
"""Frozen Qwen3-8B classification, RFM activation steering and text generation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    stages = ("pilot", "features", "rfm", "classify", "steer", "generate", "evaluate", "report")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*stages, "all"))
    args = parser.parse_args()
    if args.stage == "all":
        for stage in stages:
            subprocess.run([sys.executable, str(__file__), stage], cwd=ROOT, check=True)
        return
    from imdb_lab.common import environment, file_sha256, save_json, seed_all
    from imdb_lab.qwen_core import qconfig
    cfg = qconfig()
    seed_all(cfg["seed"], cfg["threads"])
    status = ROOT / "logs" / f"qwen_{args.stage}_status.json"
    record = {"pid": os.getpid(), "stage": args.stage, "config_sha256": file_sha256(ROOT / "qwen_config.json")}
    save_json(status, {**record, "status": "running"})
    try:
        if args.stage == "report":
            from imdb_lab.qwen_report import run
            run()
        else:
            from imdb_lab import qwen_experiment
            getattr(qwen_experiment, args.stage)()
        save_json(status, {**record, "status": "complete", "environment": environment()})
    except Exception:
        save_json(status, {**record, "status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
