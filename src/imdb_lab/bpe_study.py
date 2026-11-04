"""Train BPE on training text only, then retrain the unchanged TextCNN.

The existing sequence module is imported, never patched. Its initializer,
optimizer, batching, trainer, pooling masks and prediction routine are shared.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from .common import ROOT, file_sha256, read_json, save_json
from .models import Vocabulary, tokenize
from . import sequences as shared

BASE = ROOT / "results/bpe"
INPUT = ROOT / ".cache/bpe_inputs"
WEIGHTS = ROOT / "artifacts/bpe"


def settings(): return read_json(ROOT / "bpe_config.json")


def canonicalize(text):
    """Hold lexical normalization constant with the original regex tokenizer."""
    return " ".join(tokenize(text))


def fit_bpe(texts, vocab_size=20000, min_frequency=2):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", end_of_word_suffix="</w>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=["<pad>", "<unk>"], end_of_word_suffix="</w>", show_progress=False)
    tokenizer.train_from_iterator((canonicalize(t) for t in texts), trainer, length=len(texts))
    assert tokenizer.token_to_id("<pad>") == 0 and tokenizer.token_to_id("<unk>") == 1
    return tokenizer


def fingerprints():
    names = ["bpe_config.json", "run_bpe.py", "src/imdb_lab/bpe_study.py",
             "sequence_config.json", "src/imdb_lab/sequences.py", "src/imdb_lab/models.py", "src/imdb_lab/common.py"]
    return {p: file_sha256(ROOT / p) for p in names}


def word_record(seed):
    return read_json(ROOT / "results/sequences/runs" / f"{settings()['word_reference']}__s{seed}.json")


def prepare():
    cfg = settings()
    assert not (BASE / "selection_frozen.json").exists(), "Use a fresh workspace to rerun"
    source = ROOT / "data/train.parquet"; frame = pd.read_parquet(source)
    assert frame.official_split.eq("train").all() and frame.id.is_unique and frame.text_sha256.is_unique
    seq = read_json(ROOT / "results/sequences/data_manifest.json")
    old = read_json(ROOT / "results/didactic/data_manifest.json")
    assert file_sha256(source) == seq["source_sha256"] == old["source_sha256"]
    excluded = set().union(*(set(m["roles"][r]["ids"]) for m in [old, seq] for r in m["roles"]))
    pool = frame[~frame.id.isin(excluded)]
    rng = np.random.default_rng(cfg["split_seed"])
    chosen = np.concatenate([rng.choice(pool[pool.label.eq(y)].index, cfg["data"]["audit_n"] // 2, replace=False) for y in [0, 1]])
    parts = {r: frame.set_index("id").loc[seq["roles"][r]["ids"]].reset_index() for r in ["train", "validation"]}
    parts["audit"] = frame.loc[chosen].sort_values("id").reset_index(drop=True)
    assert not set(parts["audit"].id) & excluded
    INPUT.mkdir(parents=True, exist_ok=True); BASE.mkdir(parents=True, exist_ok=True)
    manifest = {"source_sha256": file_sha256(source), "official_test_opened": False,
        "previous_manifests": {f"results/{name}/data_manifest.json": file_sha256(ROOT / f"results/{name}/data_manifest.json") for name in ["didactic", "sequences"]},
        "new_audit_excludes_all_previous_didactic_and_sequence_roles": True,
        "available_unused_pool_n": len(pool), "roles": {}}
    for role, rows in parts.items():
        texts = rows.clean_text.map(lambda t: shared.excerpt(t, cfg["whitespace_words"]))
        local = pd.DataFrame({"id": rows.id, "label": rows.label.astype(int), "text": texts})
        path = INPUT / f"{role}.parquet"; local.to_parquet(path, index=False)
        manifest["roles"][role] = {"n": len(rows), "ids": rows.id.tolist(), "text_sha256": rows.text_sha256.tolist(),
            "excerpt_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in texts],
            "labels_sha256": shared.array_hash(rows.label.to_numpy(dtype=np.int64)),
            "input_sha256": file_sha256(path), "label_counts": {str(k): int(v) for k, v in rows.label.value_counts().items()}}
        if role != "audit":
            manifest["roles"][role]["labels"] = rows.label.astype(int).tolist()
            assert manifest["roles"][role]["excerpt_sha256"] == seq["roles"][role]["excerpt_sha256"]
    train = pd.read_parquet(INPUT / "train.parquet")
    tick = time.perf_counter(); tokenizer = fit_bpe(train.text.tolist(), cfg["tokenizer"]["vocab_size"], cfg["tokenizer"]["min_frequency"])
    elapsed = time.perf_counter() - tick
    assert tokenizer.get_vocab_size() == cfg["tokenizer"]["vocab_size"]
    tokenizer.save(str(BASE / "tokenizer.json"))
    saved = read_json(BASE / "tokenizer.json")
    save_json(BASE / "tokenizer_training.json", {"fit_role": "train", "labels_used": False, "documents": len(train),
        "canonical_train_sha256": hashlib.sha256("\n".join(canonicalize(t) for t in train.text).encode()).hexdigest(),
        "tokenizers_version": importlib.metadata.version("tokenizers"), "fit_seconds": elapsed,
        "vocabulary_size": tokenizer.get_vocab_size(), "merge_count": len(saved["model"]["merges"]),
        "first_merges": saved["model"]["merges"][:16], "tokenizer_sha256": file_sha256(BASE / "tokenizer.json"),
        "algorithm": cfg["tokenizer"]["algorithm"], "normalization": cfg["tokenizer"]["canonicalization"]})
    manifest["tokenizer_sha256"] = file_sha256(BASE / "tokenizer.json")
    manifest["word_vocabulary_sha256"] = file_sha256(ROOT / "results/sequences/vocabulary.json")
    save_json(BASE / "data_manifest.json", manifest)
    save_json(BASE / "protocol_prepared.json", {"code": fingerprints(), "data_manifest_sha256": file_sha256(BASE / "data_manifest.json"),
        "tokenizer_training_sha256": file_sha256(BASE / "tokenizer_training.json"), "tokenizer_sha256": manifest["tokenizer_sha256"]})
    print("Prepared BPE training-only tokenizer and isolated roles", {r: len(p) for r, p in parts.items()}, flush=True)


def load_role(role):
    expected = read_json(BASE / "data_manifest.json")["roles"][role]
    path = INPUT / f"{role}.parquet"
    assert file_sha256(path) == expected["input_sha256"]
    frame = pd.read_parquet(path)
    assert frame.id.tolist() == expected["ids"]
    assert shared.array_hash(frame.label.to_numpy(dtype=np.int64)) == expected["labels_sha256"]
    return frame


def encoded(texts, kind):
    if kind == "word":
        vocab = Vocabulary(read_json(ROOT / "results/sequences/vocabulary.json"))
        return [[vocab.token_to_id.get(w, 1) for w in tokenize(t)] for t in texts]
    assert kind == "bpe"
    tokenizer = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    return [e.ids for e in tokenizer.encode_batch([canonicalize(t) for t in texts])]


def token_stats(sequences, cap):
    sizes = np.array([len(s) for s in sequences], dtype=np.int64)
    unknown = sum(s.count(1) for s in sequences)
    return {"documents": len(sequences), "tokens_before_cap": int(sizes.sum()), "unknown_tokens": unknown,
        "unknown_rate": float(unknown / max(1, sizes.sum())), "documents_with_unknown": sum(1 in s for s in sequences),
        "mean_tokens": float(sizes.mean()), "max_tokens": int(sizes.max()), "p95_tokens": float(np.quantile(sizes, .95)),
        "truncated_documents": int((sizes > cap).sum()), "empty_documents": int((sizes == 0).sum()),
        "cap": cap, "empty_policy": "A single <unk> when the canonicalized text is empty"}


def inputs(frame, kind, target):
    cap = settings()["token_cap"]; seq = encoded(frame.text.tolist(), kind)
    stats = token_stats(seq, cap); clipped = [s[:cap] or [1] for s in seq]
    width = max(5, max(map(len, clipped)))
    x = torch.zeros((len(clipped), width), dtype=torch.long, device=target)
    lengths = torch.tensor(list(map(len, clipped)), device=target)
    for i, s in enumerate(clipped): x[i, :len(s)] = torch.tensor(s, device=target)
    assert int(x.max()) < settings()["tokenizer"]["vocab_size"]
    return {"input_ids": x, "lengths": lengths}, stats


def save_run(result):
    run_id = f"cnn_bpe20k__s{result['seed']}"; WEIGHTS.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS / f"{run_id}.pt"; torch.save(result.pop("best_state"), path)
    pred = BASE / "runs" / f"{run_id}__validation.npz"; pred.parent.mkdir(exist_ok=True)
    np.savez_compressed(pred, **result.pop("best_predictions"))
    result.update(run_id=run_id, checkpoint={"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha256(path)},
                  predictions={"path": pred.relative_to(ROOT).as_posix(), "sha256": file_sha256(pred)})
    save_json(BASE / "runs" / f"{run_id}.json", result)
    return result


def train(device="cuda"):
    cfg = settings(); prepared = read_json(BASE / "protocol_prepared.json")
    assert prepared["code"] == fingerprints()
    assert file_sha256(BASE / "data_manifest.json") == prepared["data_manifest_sha256"]
    assert file_sha256(BASE / "tokenizer.json") == prepared["tokenizer_sha256"]
    assert not (INPUT / "audit.parquet").exists(), "Withhold audit input until checkpoint freeze"
    assert not (BASE / "selection_frozen.json").exists(), "Study already frozen"
    target = shared.configure(device)
    frames = {r: load_role(r) for r in ["train", "validation"]}
    data = {}; stats = {}
    for r, frame in frames.items():
        x, stats[r] = inputs(frame, "bpe", target)
        data[r] = x, torch.tensor(frame.label.to_numpy(), device=target, dtype=torch.long)
    save_json(BASE / "tokenization_train_validation.json", {"bpe": stats,
        "word": {r: token_stats(encoded(f.text.tolist(), "word"), cfg["token_cap"]) for r, f in frames.items()}})
    runtime = {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "tokenizers": importlib.metadata.version("tokenizers"), "accelerator": torch.cuda.get_device_name(0) if target.type == "cuda" else "CPU",
        "precision": "FP32, deterministic algorithms, TF32 disabled", "threads": cfg["threads"],
        "audit_input_absent_at_start": True, "prepared_sha256": file_sha256(BASE / "protocol_prepared.json")}
    save_json(BASE / "runtime.json", runtime)
    records = []
    for seed in cfg["seeds"]:
        path = BASE / "runs" / f"cnn_bpe20k__s{seed}.json"
        if path.exists():
            result = read_json(path)
            assert file_sha256(ROOT / result["checkpoint"]["path"]) == result["checkpoint"]["sha256"]
            assert file_sha256(ROOT / result["predictions"]["path"]) == result["predictions"]["sha256"]
        else:
            model = shared.initialize_cnn(cfg["cnn"], seed, target)
            spec = {**cfg["cnn"], "run_id": "cnn_bpe20k"}
            result = save_run(shared.train_one(model, data["train"], data["validation"], spec, seed, target))
            del model
        reference = word_record(seed)
        for field in ["initial_state_sha256", "initial_head_sha256", "batch_order_sha256", "parameters", "trainable_parameters"]:
            assert result[field] == reference[field], f"Pairing mismatch: {field}"
        assert reference["spec"]["lr"] == cfg["cnn"]["lr"]
        records.extend([{"kind": "word", "seed": seed, "metadata": f"results/sequences/runs/{reference['run_id']}.json", **reference["checkpoint"]},
                        {"kind": "bpe", "seed": seed, "metadata": f"results/bpe/runs/{result['run_id']}.json", **result["checkpoint"]}])
    assert not (INPUT / "audit.parquet").exists()
    for r in records:
        assert file_sha256(ROOT / r["path"]) == r["sha256"]
        r["metadata_sha256"] = file_sha256(ROOT / r["metadata"])
    save_json(BASE / "selection_frozen.json", {"code": fingerprints(), "prepared_sha256": file_sha256(BASE / "protocol_prepared.json"),
        "tokenizer_sha256": file_sha256(BASE / "tokenizer.json"), "records": records, "new_training_runs": 3,
        "reused_word_checkpoints": 3, "audit_input_absent_at_freeze": True, "audit_opened": False,
        "selection": cfg["selection"], "official_test_opened": False})
    print("Frozen three BPE models and three unchanged word references before audit release", flush=True)


def metrics(logits, labels):
    z = np.asarray(logits, dtype=np.float64); y = np.asarray(labels, dtype=np.int64)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    pred = z.argmax(1)
    return {"accuracy": float(accuracy_score(y, pred)), "ce": float(log_loss(y, p, labels=[0, 1])),
        "f1_macro": float(f1_score(y, pred, average="macro")), "roc_auc": float(roc_auc_score(y, p[:, 1])),
        "brier": float(np.mean((p[:, 1] - y) ** 2)), "correct": int((pred == y).sum())}


def validate_freeze():
    frozen = read_json(BASE / "selection_frozen.json")
    assert frozen["code"] == fingerprints()
    assert frozen["prepared_sha256"] == file_sha256(BASE / "protocol_prepared.json")
    assert frozen["tokenizer_sha256"] == file_sha256(BASE / "tokenizer.json")
    for r in frozen["records"]: assert file_sha256(ROOT / r["metadata"]) == r["metadata_sha256"]
    return frozen


def audit(device="cuda"):
    frozen = validate_freeze()
    assert not (BASE / "audit_predictions.npz").exists(), "Do not reopen frozen audit; replay instead"
    release = read_json(BASE / "audit_release.json")
    assert release["selection_frozen_sha256"] == file_sha256(BASE / "selection_frozen.json")
    target = shared.configure(device); frame = load_role("audit")
    data = {}; stats = {}
    for kind in ["word", "bpe"]: data[kind], stats[kind] = inputs(frame, kind, target)
    y = torch.tensor(frame.label.to_numpy(), device=target, dtype=torch.long)
    arrays = {"labels": frame.label.to_numpy(dtype=np.int64)}; rows = []
    for r in frozen["records"]:
        path = ROOT / r["path"]; assert file_sha256(path) == r["sha256"]
        model = shared.initialize_cnn(settings()["cnn"], r["seed"], target)
        model.load_state_dict(torch.load(path, map_location=target, weights_only=True))
        shared.sync(target); tick = time.perf_counter(); result = shared.predict(model, data[r["kind"]], y, 64, target); shared.sync(target)
        key = f"{r['kind']}__s{r['seed']}"; arrays[key] = result["logits"]
        rows.append({"run_id": key, "kind": r["kind"], "seed": r["seed"], **metrics(result["logits"], frame.label), "inference_seconds": time.perf_counter() - tick})
        del model
    np.savez_compressed(BASE / "audit_predictions.npz", **arrays)
    save_json(BASE / "audit_metrics.json", {"selection_frozen_sha256": file_sha256(BASE / "selection_frozen.json"),
        "predictions_sha256": file_sha256(BASE / "audit_predictions.npz"), "metrics": rows, "tokenization": stats,
        "official_test_opened": False})
    print("Evaluated six frozen checkpoints on the fresh internal holdout", flush=True)


def replay(device="cuda"):
    frozen = validate_freeze(); target = shared.configure(device); frame = load_role("audit")
    data = {k: inputs(frame, k, target)[0] for k in ["word", "bpe"]}
    y = torch.tensor(frame.label.to_numpy(), device=target, dtype=torch.long)
    stored = np.load(BASE / "audit_predictions.npz"); checked = []
    for r in frozen["records"]:
        path = ROOT / r["path"]; assert file_sha256(path) == r["sha256"]
        model = shared.initialize_cnn(settings()["cnn"], r["seed"], target)
        model.load_state_dict(torch.load(path, map_location=target, weights_only=True))
        result = shared.predict(model, data[r["kind"]], y, 64, target)
        key = f"{r['kind']}__s{r['seed']}"; delta = float(np.max(np.abs(result["logits"] - stored[key])))
        assert delta == 0.0
        checked.append({"run_id": key, "checkpoint_sha256": r["sha256"], "max_absolute_logit_difference": delta})
        del model
    save_json(BASE / "replay_verification.json", {"status": "passed", "fresh_process": True, "device": str(target),
        "selection_frozen_sha256": file_sha256(BASE / "selection_frozen.json"), "predictions_sha256": file_sha256(BASE / "audit_predictions.npz"), "checks": checked})
    print("All six checkpoints exactly reproduce their saved audit logits", flush=True)


def verify():
    frozen = validate_freeze(); cfg = settings(); manifest = read_json(BASE / "data_manifest.json")
    prepared = read_json(BASE / "protocol_prepared.json")
    assert prepared["data_manifest_sha256"] == file_sha256(BASE / "data_manifest.json")
    assert prepared["tokenizer_training_sha256"] == file_sha256(BASE / "tokenizer_training.json")
    excluded = set()
    for name, digest in manifest["previous_manifests"].items():
        assert file_sha256(ROOT / name) == digest
        old = read_json(ROOT / name)
        excluded.update(set().union(*(set(v["ids"]) for v in old["roles"].values())))
    assert not set(manifest["roles"]["audit"]["ids"]) & excluded
    assert manifest["roles"]["audit"]["label_counts"] == {"0": 1024, "1": 1024}
    assert file_sha256(ROOT / "results/sequences/vocabulary.json") == manifest["word_vocabulary_sha256"]
    for r in ["train", "validation"]:
        role = manifest["roles"][r]
        assert shared.array_hash(np.asarray(role["labels"], dtype=np.int64)) == role["labels_sha256"]
    tokenizer = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    assert tokenizer.get_vocab_size() == cfg["tokenizer"]["vocab_size"]
    assert tokenizer.token_to_id("<pad>") == 0 and tokenizer.token_to_id("<unk>") == 1
    audit = read_json(BASE / "audit_metrics.json"); pred = np.load(BASE / "audit_predictions.npz")
    assert audit["selection_frozen_sha256"] == file_sha256(BASE / "selection_frozen.json")
    assert file_sha256(BASE / "audit_predictions.npz") == audit["predictions_sha256"]
    assert shared.array_hash(pred["labels"]) == manifest["roles"]["audit"]["labels_sha256"]
    max_difference = 0.; count = 0
    for row in audit["metrics"]:
        computed = metrics(pred[row["run_id"]], pred["labels"])
        for name, value in computed.items(): max_difference = max(max_difference, abs(value - row[name]))
        assert all(abs(computed[k] - row[k]) < 1e-12 for k in computed)
        count += 1
    for seed in cfg["seeds"]:
        w = word_record(seed); b = read_json(BASE / "runs" / f"cnn_bpe20k__s{seed}.json")
        for field in ["initial_state_sha256", "initial_head_sha256", "batch_order_sha256", "parameters"]: assert w[field] == b[field]
        assert b["parameters"] == 2714502
        assert b["best_epoch"] == min(b["history"][1:], key=lambda h: h["validation_ce"])["epoch"]
        for kind, record in [("word", w), ("bpe", b)]:
            path = ROOT / record["predictions"]["path"]; assert file_sha256(path) == record["predictions"]["sha256"]
            z = np.load(path)
            expected = next(h for h in record["history"] if h["epoch"] == record["best_epoch"])
            for role in ["train", "validation"]:
                m = metrics(z[role], manifest["roles"][role]["labels"])
                for key in ["ce", "accuracy"]:
                    d = abs(m[key] - expected[f"{role}_{key}"]); max_difference = max(max_difference, d)
                    assert d < 2e-6
                count += 1
    replay = read_json(BASE / "replay_verification.json")
    assert replay["status"] == "passed" and len(replay["checks"]) == 6
    assert replay["selection_frozen_sha256"] == file_sha256(BASE / "selection_frozen.json")
    assert replay["predictions_sha256"] == file_sha256(BASE / "audit_predictions.npz")
    for r, check in zip(frozen["records"], replay["checks"]):
        assert r["sha256"] == check["checkpoint_sha256"] and check["max_absolute_logit_difference"] == 0
    save_json(BASE / "verification.json", {"status": "passed", "metric_sets_recomputed": count,
        "maximum_metric_difference": max_difference, "same_numeric_initialization": True, "same_batch_orders": True,
        "same_parameter_count": 2714502, "six_checkpoint_replay": "passed", "fresh_internal_audit_disjoint": True,
        "replay_certificate_sha256": file_sha256(BASE / "replay_verification.json"), "official_test_opened": False})
    print("Verified BPE protocol, six paired checkpoints and", count, "metric sets", flush=True)
