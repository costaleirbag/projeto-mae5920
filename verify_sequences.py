#!/usr/bin/env python3
"""Independently reconstruct CNN/encoder metrics and validation choices."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from imdb_lab.common import file_sha256, read_json, save_json
from imdb_lab.sequences import fingerprint, array_hash, orders


def metrics(logits, labels):
    z = np.asarray(logits, dtype=np.float64); y = np.asarray(labels, dtype=np.int64)
    assert z.shape == (len(y), 2) and np.isfinite(z).all()
    p = np.exp(z - logsumexp(z, axis=1, keepdims=True))[:, 1]
    pred = z.argmax(1)
    return {"ce": float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y])),
            "accuracy": float(np.mean(pred == y)), "f1_macro": float(f1_score(y, pred, average="macro")),
            "brier": float(np.mean((p - y) ** 2)), "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def verify(checkpoints=False):
    base = ROOT / "results/sequences"; cfg = read_json(ROOT / "sequence_config.json")
    manifest = read_json(base / "data_manifest.json"); frozen = read_json(base / "selection_frozen.json")
    assert fingerprint() == frozen["code"]
    assert file_sha256(base / "data_manifest.json") == frozen["data_manifest_sha256"]
    assert manifest["config_sha256"] == file_sha256(ROOT / "sequence_config.json")
    assert not manifest["official_test_opened"] and not frozen["official_test_opened"] and not frozen["audit_opened"]
    previous = read_json(ROOT / "results/didactic/data_manifest.json")
    assert manifest["source_sha256"] == previous["source_sha256"]
    assert file_sha256(ROOT / "results/didactic/data_manifest.json") == manifest["previous_didactic_manifest_sha256"]
    roles = manifest["roles"]
    for r, n in [("train", 6144), ("validation", 1536), ("audit", 2048)]:
        assert roles[r]["n"] == n and len(roles[r]["ids"]) == n
        assert roles[r]["label_counts"] == {"0": n // 2, "1": n // 2}
    for a, b in [("train", "validation"), ("train", "audit"), ("validation", "audit")]:
        assert not set(roles[a]["ids"]) & set(roles[b]["ids"])
        assert not set(roles[a]["text_sha256"]) & set(roles[b]["text_sha256"])
    for r in ["train", "validation"]: assert roles[r]["ids"] == previous["roles"][r]["ids"]
    old_ids = set().union(*(set(r["ids"]) for r in previous["roles"].values()))
    assert not old_ids & set(roles["audit"]["ids"])
    assert "labels" not in roles["audit"]  # Audit labels were not part of the training input manifest.
    assert file_sha256(base / "vocabulary.json") == manifest["vocabulary_sha256"]
    assert read_json(base / "models_pinned.json") == frozen["models"]
    vocab_n = len(read_json(base / "vocabulary.json"))
    records = []; max_ce_delta = 0.0
    assert len(frozen["runs"]) == 87
    for info in frozen["runs"]:
        path = base / "runs" / f"{info['run_id']}.json"
        assert file_sha256(path) == info["sha256"]
        r = read_json(path); spec = r["spec"]; records.append(r)
        assert r["checkpoint"] == info["checkpoint"]
        if checkpoints:
            cp = ROOT / r["checkpoint"]["path"]
            assert file_sha256(cp) == r["checkpoint"]["sha256"]
        assert r["seed"] in cfg["seeds"]
        assert len(r["history"]) == spec["epochs"] + 1
        assert [h["epoch"] for h in r["history"]] == list(range(spec["epochs"] + 1))
        assert r["history"][-1]["updates"] == spec["epochs"] * int(np.ceil(6144 / spec["batch_size"]))
        assert r["batch_order_sha256"] == array_hash(np.concatenate(orders(6144, spec["epochs"], r["seed"])))
        best = min(r["history"][1:], key=lambda h: h["validation_ce"])
        assert r["best_epoch"] == best["epoch"] and r["best_validation_ce"] == best["validation_ce"]
        pred = ROOT / r["predictions"]["path"]; assert file_sha256(pred) == r["predictions"]["sha256"]
        with np.load(pred) as npz:
            for role in ["train", "validation"]:
                m = metrics(npz[role], roles[role]["labels"])
                delta = abs(m["ce"] - best[role + "_ce"]); max_ce_delta = max(max_ce_delta, delta)
                assert delta < 2e-6
                assert abs(m["accuracy"] - best[role + "_accuracy"]) < 1e-7
        if spec["family"] == "cnn":
            expected = vocab_n * spec["embedding_dim"] + sum(spec["filters"] * (spec["embedding_dim"] * k + 1) for k in spec["kernels"]) + 2 * spec["filters"] * len(spec["kernels"]) + 2
            assert r["parameters"] == expected == r["trainable_parameters"]
        elif spec["group"] == "probe": assert r["parameters"] == 1538 == r["trainable_parameters"]
        else:
            assert r["parameters"] == r["trainable_parameters"] > 60_000_000
            assert r["parameters"] == read_json(base / f"{spec['family']}_architecture.json")["parameters"]
    assert sum(r["spec"]["family"] == "cnn" for r in records) == 42
    assert sum(r["spec"]["group"] == "probe" for r in records) == 27
    assert sum(r["spec"]["group"] == "full" for r in records) == 18
    choices = {}; candidate_means = {}
    for key in frozen["selections"]:
        family, method = key.split("_", 1)
        group = [r for r in records if r["spec"]["family"] == family and (r["spec"]["group"] == method if family != "cnn" else r["spec"]["group"] == "optimizer" and r["spec"]["optimizer"] == method)]
        rates = sorted({r["spec"]["lr"] for r in group})
        prescribed = cfg["cnn"]["learning_rates"][method] if family == "cnn" else cfg["encoders"][method]["learning_rates"]
        assert rates == sorted(prescribed)
        assert len(group) == len(rates) * 3
        means = {str(lr): float(np.mean([r["best_validation_ce"] for r in group if r["spec"]["lr"] == lr])) for lr in rates}
        rate = min(rates, key=lambda lr: (means[str(lr)], lr))
        assert rate == frozen["selections"][key] and means == frozen["validation_candidate_means"][key]
        choices[key] = rate; candidate_means[key] = means
    for seed in cfg["seeds"]:
        cnn = [r for r in records if r["seed"] == seed and r["spec"]["family"] == "cnn" and r["spec"]["group"] in {"optimizer", "dropout"}]
        assert len({r["initial_state_sha256"] for r in cnn}) == 1
        assert len({r["batch_order_sha256"] for r in cnn}) == 1
        assert len({r["initial_head_sha256"] for r in records if r["seed"] == seed and r["spec"]["family"] != "cnn"}) == 1
        for entry in cfg["encoders"]["models"]:
            family = entry["name"]
            enc = [r for r in records if r["seed"] == seed and r["spec"]["family"] == family]
            assert len({r["initial_head_sha256"] for r in enc}) == 1
            full = [r for r in enc if r["spec"]["group"] == "full"]
            assert len({r["initial_state_sha256"] for r in full}) == 1
    expected_audit_ids = []
    for r in records:
        s = r["spec"]; key = f"cnn_{s['optimizer']}" if s["family"] == "cnn" else f"{s['family']}_{s['group']}"
        if s["family"] == "cnn" and s["group"] != "optimizer" or s["lr"] == choices[key]: expected_audit_ids.append(r["run_id"])
    assert expected_audit_ids == frozen["audit_run_ids"] and len(expected_audit_ids) == 42
    baseline = frozen["baseline"]
    best_baseline = min(baseline["validation_candidates"], key=lambda x: (x["ce"], x["c"]))
    assert best_baseline["c"] == baseline["selected_c"] and best_baseline["ce"] == baseline["validation_ce"]
    if checkpoints:
        assert file_sha256(ROOT / baseline["checkpoint"]["path"]) == baseline["checkpoint"]["sha256"]
        save_json(base / "checkpoint_verification.json", {"status": "passed", "checkpoints": 88,
                  "selection_frozen_sha256": file_sha256(base / "selection_frozen.json"),
                  "sha256": {**{r["run_id"]: r["checkpoint"]["sha256"] for r in records}, "tfidf": baseline["checkpoint"]["sha256"]}})
    certificate = read_json(base / "checkpoint_verification.json")
    assert certificate["status"] == "passed" and certificate["checkpoints"] == 88
    assert certificate["selection_frozen_sha256"] == file_sha256(base / "selection_frozen.json")
    assert certificate["sha256"] == {**{r["run_id"]: r["checkpoint"]["sha256"] for r in records}, "tfidf": baseline["checkpoint"]["sha256"]}
    transfer = read_json(base / "audit_transfer_record.json")
    assert transfer["selection_existed_before_transfer"] and not transfer["audit_input_present_before_transfer"]
    assert transfer["selection_frozen_sha256"] == file_sha256(base / "selection_frozen.json")
    assert transfer["audit_input_sha256"] == roles["audit"]["input_sha256"]
    audit = read_json(base / "audit_metrics.json")
    assert audit["selection_frozen_sha256"] == file_sha256(base / "selection_frozen.json")
    assert not audit["tf32"] and not audit["official_test_opened"]
    assert file_sha256(base / "audit_predictions.npz") == audit["predictions_sha256"]
    amap = {m["run_id"]: m for m in audit["metrics"]}
    assert set(amap) == set(expected_audit_ids) | {"tfidf"}
    rebuilt = []
    with np.load(base / "audit_predictions.npz") as data:
        assert data["ids"].tolist() == roles["audit"]["ids"]
        assert array_hash(data["y"].astype(np.int64)) == roles["audit"]["labels_sha256"]
        for run_id in expected_audit_ids + ["tfidf"]:
            m = metrics(data[run_id], data["y"])
            delta = abs(m["ce"] - amap[run_id]["ce"]); max_ce_delta = max(max_ce_delta, delta)
            assert delta < 2e-6 and abs(m["accuracy"] - amap[run_id]["accuracy"]) < 1e-7
            rebuilt.append({"run_id": run_id, **m})
    replay = read_json(base / "replay_verification.json")
    assert replay["status"] == "passed" and replay["checkpoints"] == 43 and replay["max_absolute_logit_difference"] <= 1e-5
    assert replay["audit_predictions_sha256"] == file_sha256(base / "audit_predictions.npz")
    assert replay["selection_frozen_sha256"] == file_sha256(base / "selection_frozen.json")
    assert replay["code"] == frozen["code"]
    with np.load(base / "audit_predictions.npz") as data:
        for key, record in replay["content_sha256"].items():
            assert array_hash(data[key]) == record["first"]
            if replay["per_checkpoint_max_difference"][key] == 0: assert record["first"] == record["replay"]
    result = {"status": "passed", "training_runs": 87, "cnn_runs": 42, "frozen_head_runs": 27, "full_fine_tuning_runs": 18,
              "audit_checkpoints": 43, "recomputed_metrics": 87 * 2 + 43, "max_cross_entropy_delta": max_ce_delta,
              "validation_choices": choices, "initialization_and_batch_pairing": "passed", "head_initialization_policy_pairing": "passed",
              "new_audit_disjoint_from_all_prior_didactic_roles": True, "official_test_opened": False,
              "code": fingerprint(), "checkpoints_checked_locally": checkpoints, "audit_metrics": rebuilt}
    save_json(base / "verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"audit_metrics", "code"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args(); verify(args.checkpoints)
