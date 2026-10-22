#!/usr/bin/env python3
"""Verify the distributed files with only the Python standard library."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "publication_manifest.json"
OMIT = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".cache",
        ".ipynb_checkpoints", "tmp", "dist"}


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in OMIT or part.endswith(".egg-info") for part in rel.parts):
        return False
    if rel.parts[0] == "data":
        return rel.as_posix() == "data/manifest.json"
    if rel.parts[0] == "artifacts":
        return rel.parts[:3] == ("artifacts", "qwen", "tokenizer")
    if rel.as_posix() == "results/analysis/embeddings.npz":
        return False
    if rel.parts[:3] in {("results", "qwen", "features"), ("results", "qwen", "rfm")} and rel.suffix == ".npz":
        return False
    if rel.parts[:4] == ("results", "qwen", "generation", "parts"):
        return False
    if rel.parts[:2] == ("results", "qwen_improved") and rel.name.startswith("features_") and rel.suffix == ".npz":
        return False
    if rel.name.startswith(".env") or rel.suffix in {".pyc", ".zip", ".key", ".pem"}:
        return False
    if rel.name.startswith(".") and rel.name not in {".gitignore", ".gitattributes"}:
        return False
    return True


def distribution_paths() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*") if p.is_file() and included(p))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest() -> None:
    paths = [p for p in distribution_paths() if p != MANIFEST]
    obj = {
        "schema_version": 1,
        "repository": "https://github.com/costaleirbag/projeto-mae5920",
        "purpose": "Integrity of the published distribution; scientific frozen hashes remain in their original result files.",
        "files": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in paths},
    }
    MANIFEST.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def check_local_link(source: Path, link: str) -> None:
    parsed = urlsplit(link)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return
    target = (source.parent / unquote(parsed.path)).resolve()
    assert target.is_relative_to(ROOT), f"Link outside repository: {source.relative_to(ROOT)}"
    assert target.exists(), f"Broken link: {source.relative_to(ROOT)} -> {link}"


def verify() -> None:
    obj = json.loads(MANIFEST.read_text())
    expected = obj["files"]
    actual = {p.relative_to(ROOT).as_posix() for p in distribution_paths() if p != MANIFEST}
    assert actual == set(expected), f"File inventory mismatch: missing={sorted(set(expected)-actual)} extra={sorted(actual-set(expected))}"
    secret = re.compile(r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
    local = re.compile(r"/(?:Users|home|root)/[A-Za-z0-9_.-]+/")
    endpoint = re.compile(r"\bspark[12]\b|\bdstack-ceg\b|\bssh\s+[\w.-]+@[\w.-]+|\x2essh/", re.I)
    checked_links = 0
    for name, record in expected.items():
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT) and not path.is_symlink(), f"Unsafe path: {name}"
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], f"Hash mismatch: {name}"
        assert path.stat().st_size < 100_000_000, f"Large file must be a release asset: {name}"
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=name)
        if path.suffix in {".json", ".csv", ".txt", ".md", ".py", ".html", ".toml", ".ipynb", ".yml", ".yaml", ".svg"}:
            content = path.read_text()
            assert not secret.search(content), f"Credential pattern: {name}"
            assert not local.search(content), f"Machine-local path: {name}"
            assert not endpoint.search(content), f"Infrastructure configuration: {name}"
        if path.suffix == ".md":
            content = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
            links = re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", content)
            links += re.findall(r"(?:href|src)=[\"']([^\"']+)[\"']", content)
            for link in links:
                check_local_link(path, link)
                checked_links += 1
    for name in ("experimentos_imdb.ipynb", "qwen3_8b_steering.ipynb"):
        nb = json.loads((ROOT / "notebooks" / name).read_text())
        cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert cells and all(c.get("execution_count") is not None for c in cells), name
        assert not any(o.get("output_type") == "error" for c in cells for o in c.get("outputs", [])), name
    print(f"Publication verified: {len(expected)} files, {checked_links} documentation links, two executed notebooks; no credential or local infrastructure patterns.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Rebuild the distribution manifest before verification.")
    args = parser.parse_args()
    if args.write:
        write_manifest()
    verify()
