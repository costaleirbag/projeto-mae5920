from __future__ import annotations

import hashlib
import html
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import ROOT, config, file_sha256, save_json

DATA = ROOT / "data"


def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fingerprint(text):
    normalized = clean_text(text).casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def normalize_frame(frame, split):
    frame = frame[["text", "label"]].copy()
    frame["id"] = [f"imdb_{split}_{i:05d}" for i in range(len(frame))]
    frame["official_split"] = split
    frame["clean_text"] = frame.text.map(clean_text)
    frame["text_sha256"] = frame.clean_text.map(fingerprint)
    frame["word_count"] = frame.clean_text.str.split().str.len()
    return frame


def remove_training_leakage(train, test):
    """Use test TEXT hashes for duplicate exclusion only, never its labels."""
    train = train.copy()
    test_hashes = set(test.text_sha256)
    cross = train.text_sha256.isin(test_hashes)
    train_without_cross = train.loc[~cross].copy()
    conflicts = train_without_cross.groupby("text_sha256").label.nunique()
    conflict_hashes = set(conflicts[conflicts > 1].index)
    bad = train_without_cross.text_sha256.isin(conflict_hashes)
    clean = train_without_cross.loc[~bad].drop_duplicates("text_sha256", keep="first").copy()
    stats = {"official_train_n": len(train), "official_test_n": len(test),
             "train_rows_matching_test_text": int(cross.sum()),
             "conflicting_train_rows_excluded": int(bad.sum()),
             "duplicate_train_rows_excluded": int(len(train_without_cross) - bad.sum() - len(clean)),
             "retained_train_pool_n": len(clean),
             "duplicate_test_rows_retained": int(test.text_sha256.duplicated().sum()),
             "test_policy": "Original test retained. Training duplicates removed using test text hashes only."}
    return clean, stats


def prepare():
    from datasets import load_dataset
    from huggingface_hub import HfApi

    cfg = config()
    DATA.mkdir(exist_ok=True)
    start = time.perf_counter()
    revision = cfg.get("dataset_revision") or HfApi().dataset_info(cfg["dataset"]).sha
    raw = load_dataset(cfg["dataset"], revision=revision)
    train = normalize_frame(raw["train"].to_pandas(), "train")
    test = normalize_frame(raw["test"].to_pandas(), "test")
    if len(train) != 25000 or len(test) != 25000:
        raise ValueError("Unexpected IMDb split sizes; inspect the downloaded dataset.")
    train, audit = remove_training_leakage(train, test)
    train, holdout = train_test_split(train, test_size=cfg["validation_fraction"],
                                    stratify=train.label, random_state=cfg["seed"])
    validation, calibration = train_test_split(holdout, test_size=0.5,
                                             stratify=holdout.label, random_state=cfg["seed"] + 1)
    splits = {"train": train, "validation": validation, "calibration": calibration, "test": test}
    hash_sets = {name: set(df.text_sha256) for name, df in splits.items()}
    for name, hashes in hash_sets.items():
        for other, other_hashes in hash_sets.items():
            if name != other and hashes & other_hashes:
                raise AssertionError(f"Text leakage between {name} and {other}")
    summaries = {}
    for name, frame in splits.items():
        frame = frame.sort_values("id").reset_index(drop=True)
        frame.to_parquet(DATA / f"{name}.parquet", index=False)
        summaries[name] = {"n": len(frame), "label_counts": {str(k): int(v) for k, v in frame.label.value_counts().items()},
                           "word_count_quantiles": {str(k): float(v) for k, v in frame.word_count.quantile([0, .25, .5, .75, .95, 1]).items()},
                           "sha256": file_sha256(DATA / f"{name}.parquet")}
    save_json(DATA / "manifest.json", {"source": cfg["dataset"], "revision": revision,
              "seed": cfg["seed"], "download_and_prepare_seconds": time.perf_counter() - start,
              "labels": {"0": "negative", "1": "positive"}, "splits": summaries,
              "audit": audit, "unsupervised_reviews_used": False,
              "neutral_class": "Absent. Probability close to 0.5 means uncertainty, not a neutral ground-truth label.",
              "roles": {"train": "fit vocabulary and weights", "validation": "choose hyperparameters and checkpoints",
                        "calibration": "fit temperature only, after selection", "test": "locked final evaluation and post-hoc descriptive analysis"}})
    print({name: info["n"] for name, info in summaries.items()}, flush=True)
    print(audit, flush=True)


def load_splits():
    return {name: pd.read_parquet(DATA / f"{name}.parquet") for name in ("train", "validation", "calibration", "test")}


def stratified_sample(frame, n, seed):
    if n >= len(frame):
        return frame.reset_index(drop=True)
    indices, _ = train_test_split(np.arange(len(frame)), train_size=n, stratify=frame.label, random_state=seed)
    return frame.iloc[indices].reset_index(drop=True)
