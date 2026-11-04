#!/usr/bin/env python3
"""Create an isolated BPE rerun from restored inputs and frozen word weights."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imdb_lab.common import ROOT, file_sha256, read_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--reference-checkpoints", type=Path, default=ROOT / "artifacts/sequences")
    args = parser.parse_args(); target = args.destination.expanduser().resolve()
    assert not target.exists(), "Choose a new, nonexistent destination"
    files = ["bpe_config.json", "run_bpe.py", "sequence_config.json", "src/imdb_lab/__init__.py",
        "src/imdb_lab/common.py", "src/imdb_lab/models.py", "src/imdb_lab/sequences.py", "src/imdb_lab/bpe_study.py",
        "tools/bpe_audit_input.py", "tools/reuse_bpe_tokenizer.py", "data/train.parquet", "results/didactic/data_manifest.json",
        "results/sequences/data_manifest.json", "results/sequences/vocabulary.json"]
    weights = []
    for seed in read_json(ROOT / "bpe_config.json")["seeds"]:
        stem = f"results/sequences/runs/cnn_optimizer__adam__lr0.001__s{seed}"
        files += [stem + ".json", stem + "__validation.npz"]
        record = read_json(ROOT / (stem + ".json"))
        path = args.reference_checkpoints / Path(record["checkpoint"]["path"]).name
        if not path.is_file(): raise FileNotFoundError(f"Restore reference checkpoint first: {path.name}")
        assert file_sha256(path) == record["checkpoint"]["sha256"]
        weights.append((path, record["checkpoint"]["path"]))
    for name in files:
        if not (ROOT / name).is_file(): raise FileNotFoundError(f"Restore required input: {name}")
    assert file_sha256(ROOT / "data/train.parquet") == read_json(ROOT / "results/sequences/data_manifest.json")["source_sha256"]
    target.mkdir(parents=True)
    for source, name in [(ROOT / p, p) for p in files] + weights:
        dest = target / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)
    save_json(target / "source_workspace_manifest.json", {"purpose": "Isolated BPE rerun, published results preserved",
        "files": {p: file_sha256(target / p) for p in files + [n for _, n in weights]}})
    print("Created fresh BPE workspace with three verified reference checkpoints")


if __name__ == "__main__": main()
