from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def config():
    return read_json(ROOT / "config.json")


def seed_all(seed=42, threads=4):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS operations do not all implement deterministic algorithms.
    torch.use_deterministic_algorithms(True, warn_only=True)


def device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def release_memory():
    import gc
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def environment():
    import psutil
    import torch
    packages = {}
    for p in ("torch", "transformers", "datasets", "numpy", "pandas", "scikit-learn", "scipy", "sentence-transformers"):
        packages[p] = importlib.metadata.version(p)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "logical_cpus": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1e9,
            "device": str(device()), "accelerator": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Apple MPS" if torch.backends.mps.is_available() else "CPU"), "packages": packages,
            "determinism": "Seeds fixed; warn-only deterministic torch; results may differ across devices and runtimes."}


@contextmanager
def measured(metadata, key):
    import psutil
    synchronize()
    start = time.perf_counter()
    yield
    synchronize()
    metadata[key + "_seconds"] = time.perf_counter() - start
    metadata["rss_gb_at_end"] = psutil.Process().memory_info().rss / 1e9


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
