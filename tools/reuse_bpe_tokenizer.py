#!/usr/bin/env python3
"""Use the published BPE rules in a freshly prepared, untrained workspace.

A new statistical BPE fit can resolve frequency ties differently. Reusing the
serialized tokenizer preserves the exact intervention whose results we report.
"""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imdb_lab.common import ROOT, file_sha256, read_json, save_json
from imdb_lab.bpe_study import BASE, INPUT, fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Published results/bpe directory")
    args = parser.parse_args(); source = args.source.resolve()
    assert source != BASE.resolve(), "Source and destination must differ"
    assert not (BASE / "selection_frozen.json").exists(), "Never change a frozen study"
    assert not (BASE / "runtime.json").exists() and not list((BASE / "runs").glob("*.json")), "Never change an already started study"
    original = read_json(source / "data_manifest.json"); current = read_json(BASE / "data_manifest.json")
    assert {k: v for k, v in current.items() if k != "tokenizer_sha256"} == {k: v for k, v in original.items() if k != "tokenizer_sha256"}
    for role in ["train", "validation", "audit"]:
        assert file_sha256(INPUT / f"{role}.parquet") == original["roles"][role]["input_sha256"]
    prepared = read_json(source / "protocol_prepared.json")
    assert prepared["code"] == fingerprints()
    assert prepared["data_manifest_sha256"] == file_sha256(source / "data_manifest.json")
    assert prepared["tokenizer_training_sha256"] == file_sha256(source / "tokenizer_training.json")
    assert prepared["tokenizer_sha256"] == file_sha256(source / "tokenizer.json") == original["tokenizer_sha256"]
    fitted = read_json(BASE / "tokenizer_training.json"); reference = read_json(source / "tokenizer_training.json")
    assert fitted["canonical_train_sha256"] == reference["canonical_train_sha256"]
    assert reference["fit_role"] == "train" and reference["labels_used"] is False
    replaced = file_sha256(BASE / "tokenizer.json")
    for name in ["tokenizer.json", "tokenizer_training.json", "data_manifest.json", "protocol_prepared.json"]:
        shutil.copy2(source / name, BASE / name)
    save_json(BASE / "tokenizer_reuse.json", {"purpose": "Reuse published rules after verifying identical scientific inputs; no training started",
        "tokenizer_sha256": prepared["tokenizer_sha256"], "discarded_independent_fit_sha256": replaced,
        "canonical_training_text_identical": True, "all_input_hashes_identical": True, "source_fit_used_labels": False})
    print("Reused exact published BPE rules after verifying inputs and training-only provenance")


if __name__ == "__main__": main()
