#!/usr/bin/env python3
"""Package historical trained checkpoints and large numeric caches for a release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT, help="Directory containing the original completed run.")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/artefatos_reproducao.zip")
    args = parser.parse_args()
    source = args.source.resolve()
    paths = sorted(p for p in (source / "artifacts").rglob("*") if p.is_file()
                   and p.relative_to(source).parts[:3] != ("artifacts", "qwen", "tokenizer"))
    for folder in ("results/qwen/features", "results/qwen/rfm"):
        paths += sorted((source / folder).glob("*.npz"))
    embeddings = source / "results/analysis/embeddings.npz"
    if embeddings.exists():
        paths.append(embeddings)
    assert paths and (source / "artifacts/bert/model.safetensors").exists(), "Original trained checkpoints are required."
    assert all(not p.is_symlink() and p.resolve().is_relative_to(source) for p in paths)
    manifest = {
        "purpose": "Optional historical checkpoints and numeric caches complementing the public project. No corpus, Qwen backbone, credentials or infrastructure configuration.",
        "files": {p.relative_to(source).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in paths:
            archive.write(path, path.relative_to(source))
            print(f"Included {path.relative_to(source)}", flush=True)
        archive.writestr("reproduction_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for name, record in manifest["files"].items():
            with archive.open(name) as stream:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == record["sha256"], name
    print(f"Reproduction package verified: {len(paths)} files, {args.output.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
