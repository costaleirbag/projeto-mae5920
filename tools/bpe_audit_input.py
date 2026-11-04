#!/usr/bin/env python3
"""Release the BPE holdout only after validating a frozen training protocol."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imdb_lab.common import ROOT, read_json, file_sha256, save_json
from imdb_lab.bpe_study import validate_freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    validate_freeze()
    base = ROOT / "results/bpe"; destination = ROOT / ".cache/bpe_inputs/audit.parquet"
    assert not destination.exists(), "Audit input already released"
    expected = read_json(base / "data_manifest.json")["roles"]["audit"]["input_sha256"]
    assert file_sha256(args.source) == expected
    shutil.copy2(args.source, destination)
    save_json(base / "audit_release.json", {"selection_frozen_sha256": file_sha256(base / "selection_frozen.json"),
        "audit_input_sha256": file_sha256(destination), "input_absent_before_release": True, "official_test_opened": False})
    print("Released hash-verified holdout after checkpoint freeze")


if __name__ == "__main__": main()
