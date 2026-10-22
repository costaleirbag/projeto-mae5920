#!/usr/bin/env python3
"""Package the public distribution without corpora, checkpoints or machine caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from tools.verify_publication import ROOT, distribution_paths, verify, write_manifest


def record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/imdb_qwen3_entrega.zip")
    args = parser.parse_args()
    sources = [p for p in distribution_paths()
               if p.suffix in {".py", ".md", ".toml", ".yml", ".yaml"}
               or (p.parent == ROOT and p.suffix in {".json", ".txt"} and p.name != "publication_manifest.json" and not p.name.startswith("logs_"))]
    save(ROOT / "results/source_manifest.json", {
        "purpose": "Published source and documentation checksums. Historical scientific source hashes remain in frozen stage metadata.",
        "files": {p.relative_to(ROOT).as_posix(): record(p)["sha256"] for p in sources},
    })
    delivery = ROOT / "results/delivery_manifest.json"
    paths = [p for p in distribution_paths() if p not in {delivery, ROOT / "publication_manifest.json"}]
    save(delivery, {
        "purpose": "Public delivery: actual saved predictions, generated texts, reports and reproducible source.",
        "omitted": ["Full prepared IMDb corpus; restore from the pinned public source.",
                    "Trained TF-IDF/CNN/BERT checkpoints and large numerical caches; optional separate release asset.",
                    "Qwen backbone weights, environment caches and generation resume duplicates.",
                    "Machine-local infrastructure configuration and credentials."],
        "inventory_note": "publication_manifest.json and this file are excluded from this inventory to avoid circular hashes; the publication manifest covers this inventory and every other distributed file.",
        "files": {p.relative_to(ROOT).as_posix(): record(p) for p in paths},
    })
    write_manifest()
    verify()
    paths = distribution_paths()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, Path("imdb_sentiment") / path.relative_to(ROOT))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(paths)
    print(f"Package verified: {len(paths)} files, {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
