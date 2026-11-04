"""Controlled teaching experiments. The old IMDb/Qwen experiment is unchanged."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, file_sha256, read_json, save_json

DROOT = ROOT / "results/didactic"


def settings():
    return read_json(ROOT / "didactic_config.json")


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def code_fingerprints():
    names = ["didactic_config.json", "run_didactic.py", "src/imdb_lab/didactic.py", "src/imdb_lab/didactic_toys.py"]
    return {name: file_sha256(ROOT / name) for name in names}


def device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.device(name)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return target


def balanced_partition(frame, counts, seed):
    """Draw disjoint balanced roles using only the original cleaned training pool."""
    assert not frame.id.duplicated().any() and not frame.text_sha256.duplicated().any()
    assert all(n % 2 == 0 for n in counts.values())
    rng = np.random.default_rng(seed)
    groups = {}
    for label in [0, 1]:
        ids = np.flatnonzero(frame.label.to_numpy() == label)
        assert len(ids) >= sum(counts.values()) // 2
        groups[label] = rng.permutation(ids)
    parts, offset = {}, 0
    for role, count in counts.items():
        take = np.concatenate([groups[label][offset:offset + count // 2] for label in [0, 1]])
        parts[role] = frame.iloc[take].sort_values("id").reset_index(drop=True)
        offset += count // 2
    return parts


def prepare():
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    cfg = settings(); spec = cfg["representation"]
    source = ROOT / "data/train.parquet"
    if not source.exists():
        raise FileNotFoundError("Restore data/train.parquet with the original preparation pipeline first")
    frame = pd.read_parquet(source)
    assert frame.official_split.eq("train").all()
    parts = balanced_partition(frame, {k: spec[k + "_n"] for k in ["train", "validation", "audit"]}, cfg["split_seed"])
    vectorizer = TfidfVectorizer(max_features=spec["max_features"], min_df=spec["min_df"],
                                ngram_range=tuple(spec["ngram_range"]), sublinear_tf=True)
    counts = vectorizer.fit_transform(parts["train"].clean_text)
    svd = TruncatedSVD(n_components=spec["svd_components"], n_iter=7, random_state=cfg["split_seed"])
    train_projection = svd.fit_transform(counts)
    scaler = StandardScaler().fit(train_projection)
    DROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DROOT / "representation.npz", components=svd.components_, mean=scaler.mean_, scale=scaler.scale_,
                        idf=vectorizer.idf_, vocabulary=vectorizer.get_feature_names_out())
    manifest = {"source": "data/train.parquet", "source_sha256": file_sha256(source),
                "dataset_revision": read_json(ROOT / "config.json")["dataset_revision"],
                "config_sha256": file_sha256(ROOT / "didactic_config.json"),
                "representation_fit_role": "train only; vocabulary, IDF, SVD and standardization never see validation/audit",
                "representation_sha256": file_sha256(DROOT / "representation.npz"),
                "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()), "roles": {}}
    for role, rows in parts.items():
        z = train_projection if role == "train" else svd.transform(vectorizer.transform(rows.clean_text))
        x = scaler.transform(z).astype(np.float32)
        assert np.isfinite(x).all()
        out = DROOT / f"{role}_features.npz"
        np.savez_compressed(out, x=x, y=rows.label.to_numpy(dtype=np.int64), ids=rows.id.to_numpy(dtype=str))
        manifest["roles"][role] = {"n": len(rows), "label_counts": {str(k): int(v) for k, v in rows.label.value_counts().items()},
                                  "ids": rows.id.tolist(), "text_sha256": rows.text_sha256.tolist(), "sha256": file_sha256(out)}
    reg = parts["train"].groupby("label", group_keys=False).sample(n=cfg["regularization"]["train_n"] // 2,
                                                                               random_state=cfg["split_seed"] + 1)
    manifest["regularization_train_ids"] = sorted(reg.id.tolist())
    manifest["official_test_opened"] = False
    manifest["caveat"] = "Internal educational holdout carved from original training data; not the official 25,000-review benchmark."
    save_json(DROOT / "data_manifest.json", manifest)
    print("Prepared training-only didactic roles", {k: len(v) for k, v in parts.items()}, flush=True)


def load_role(role):
    path = DROOT / f"{role}_features.npz"
    manifest = read_json(DROOT / "data_manifest.json")
    assert file_sha256(path) == manifest["roles"][role]["sha256"]
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


class TeachingMLP(nn.Module):
    """Same learned maps for plain/residual controls; no normalization confound."""
    def __init__(self, input_dim, width, depth, residual=False, residual_scale=1.0, dropout=0.0):
        super().__init__()
        self.spec = {"input_dim": input_dim, "width": width, "depth": depth, "residual": residual,
                     "residual_scale": residual_scale, "dropout": dropout}
        self.stem = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.head = nn.Linear(width, 2)
        self.drop = nn.Dropout(dropout)
        self.residual, self.residual_scale = residual, float(residual_scale)
        for layer in [self.stem, *self.blocks]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu"); nn.init.zeros_(layer.bias)
        nn.init.normal_(self.head.weight, std=1 / math.sqrt(width)); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.stem(x)
        for layer in self.blocks:
            correction = layer(self.drop(F.relu(h)))
            h = h + self.residual_scale * correction if self.residual else correction
        return self.head(self.drop(F.relu(h)))


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def explicit_l2(model):
    return sum(layer.weight.square().sum() for layer in model.modules() if isinstance(layer, nn.Linear))


def early_stop_trace(losses, patience, min_delta):
    best, best_epoch, stale = math.inf, None, 0
    for epoch, loss in enumerate(losses, 1):
        if loss < best - min_delta:
            best, best_epoch, stale = loss, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            return best_epoch, epoch
    return best_epoch, len(losses)


def epoch_orders(n, epochs, seed):
    return [np.random.default_rng(seed + 1009 * epoch).permutation(n) for epoch in range(epochs)]


def make_optimizer(model, name, lr):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    if name in {"sgd", "momentum"}:
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9 if name == "momentum" else 0.0, weight_decay=0)
    raise ValueError(name)


@torch.inference_mode()
def assess(model, x, y, batch_size=2048):
    model.eval(); logits = torch.cat([model(part) for part in x.split(batch_size)])
    return {"ce": float(F.cross_entropy(logits, y)), "accuracy": float((logits.argmax(1) == y).float().mean()),
            "p": logits.softmax(1)[:, 1].cpu().numpy()}


def train_one(train, validation, spec, optimizer, lr, seed, epochs, batch_size, destination,
              l2=0.0, patience=None, min_delta=0.0):
    """Never accepts audit data. Fixed epoch budget and paired order across conditions."""
    torch.manual_seed(seed)
    model = TeachingMLP(**spec).to(destination)
    initial = state_hash(model)
    tx = torch.as_tensor(train["x"], device=destination); ty = torch.as_tensor(train["y"], device=destination)
    vx = torch.as_tensor(validation["x"], device=destination); vy = torch.as_tensor(validation["y"], device=destination)
    orders = epoch_orders(len(tx), epochs, seed)
    opt = make_optimizer(model, optimizer, lr)
    history = []; started = time.perf_counter(); train_seconds = 0.0
    initial_metric = assess(model, tx, ty); vm = assess(model, vx, vy)
    history.append({"epoch": 0, "updates": 0, "train_ce": initial_metric["ce"], "train_accuracy": initial_metric["accuracy"],
                    "validation_ce": vm["ce"], "validation_accuracy": vm["accuracy"], "training_seconds": 0.0,
                    "objective": initial_metric["ce"] + 0.5 * l2 * float(explicit_l2(model).detach()),
                    "weight_norm": float(explicit_l2(model).detach().sqrt())})
    best_loss, best_state, best_epoch = math.inf, None, None
    early_state, stop_epoch, stale = None, None, 0
    for epoch, order in enumerate(orders, 1):
        model.train()
        if destination.type == "cuda": torch.cuda.synchronize()
        tick = time.perf_counter()
        for indices in np.array_split(order, math.ceil(len(order) / batch_size)):
            index = torch.as_tensor(indices, device=destination)
            opt.zero_grad(set_to_none=True)
            objective = F.cross_entropy(model(tx[index]), ty[index]) + 0.5 * l2 * explicit_l2(model)
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError(f"Nonfinite objective for {optimizer}, lr={lr}, seed={seed}")
            objective.backward(); opt.step()
        if destination.type == "cuda": torch.cuda.synchronize()
        train_seconds += time.perf_counter() - tick
        tm, vm = assess(model, tx, ty), assess(model, vx, vy)
        penalty = float(explicit_l2(model).detach())
        history.append({"epoch": epoch, "updates": epoch * math.ceil(len(order) / batch_size),
                        "train_ce": tm["ce"], "train_accuracy": tm["accuracy"],
                        "validation_ce": vm["ce"], "validation_accuracy": vm["accuracy"],
                        "training_seconds": train_seconds, "objective": tm["ce"] + 0.5 * l2 * penalty,
                        "weight_norm": math.sqrt(penalty)})
        if patience is not None and stop_epoch is None:
            if vm["ce"] < best_loss - min_delta:
                best_loss, best_epoch, best_state, stale = vm["ce"], epoch, snapshot(model), 0
            else:
                stale += 1
            if stale >= patience:
                stop_epoch, early_state = epoch, copy.deepcopy(best_state)
    if patience is not None and stop_epoch is None:
        stop_epoch, early_state = epochs, best_state
    return {"history": history, "initial_state_sha256": initial,
            "minibatch_order_sha256": array_hash(np.concatenate(orders)),
            "training_seconds": train_seconds, "total_seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()), "final_state": snapshot(model),
            "early_state": early_state, "early_epoch": best_epoch, "virtual_stop_epoch": stop_epoch,
            "spec": spec, "seed": seed, "optimizer": optimizer, "lr": lr, "l2": l2}


def save_run(group, name, result):
    directory = DROOT / "runs"; directory.mkdir(exist_ok=True)
    prefix = f"{group}__{name}__s{result['seed']}"
    weights = {}
    for label in ["final", "early"]:
        state = result[label + "_state"]
        if state is not None:
            path = directory / f"{prefix}__{label}.npz"
            np.savez_compressed(path, **{k: v.numpy() for k, v in state.items()})
            weights[label] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha256(path)}
    public = {k: v for k, v in result.items() if not k.endswith("_state")}
    public.update(group=group, name=name, weights=weights)
    save_json(directory / f"{prefix}.json", public)
    return public


def train_all(target_name="auto"):
    cfg = settings(); torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    target = device(target_name)
    if target.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    train, validation = load_role("train"), load_role("validation")
    assert not (DROOT / "selection_frozen.json").exists(), "Use a separate output copy for a new study"
    save_json(DROOT / "protocol_started.json", {"code": code_fingerprints(), "data_manifest_sha256": file_sha256(DROOT / "data_manifest.json"),
              "audit_opened": False, "device": target.type, "torch": torch.__version__, "numpy": np.__version__,
              "python": platform.python_version(), "threads": cfg["threads"],
              "float32_policy": "TF32 disabled in training and audit; no mixed precision",
              "accelerator": torch.cuda.get_device_name(0) if target.type == "cuda" else "CPU"})
    records = []; opt_cfg = cfg["optimizers"]
    spec = {"input_dim": train["x"].shape[1], "width": opt_cfg["width"], "depth": opt_cfg["depth"]}
    for name, rates in opt_cfg["learning_rates"].items():
        for lr in rates:
            for seed in cfg["seeds"]:
                out = train_one(train, validation, spec, name, lr, seed, opt_cfg["epochs"], opt_cfg["batch_size"], target)
                rec = save_run("optimizers", f"{name}_lr{lr:g}", out); records.append(rec)
                print("Completed", rec["group"], rec["name"], seed, "val CE", round(rec["history"][-1]["validation_ce"], 4), flush=True)
    choices = {}
    for name, rates in opt_cfg["learning_rates"].items():
        scores = {str(lr): float(np.mean([r["history"][-1]["validation_ce"] for r in records if r["optimizer"] == name and r["lr"] == lr])) for lr in rates}
        choices[name] = {"lr": float(min(scores, key=lambda lr: (scores[lr], float(lr)))), "mean_final_validation_ce_by_lr": scores}
    adam_lr = choices["adam"]["lr"]
    arch = cfg["architectures"]
    for variant in arch["variants"]:
        scale = 1 / math.sqrt(variant["depth"] - 1) if variant["residual_scale"] == "inverse_sqrt_blocks" else variant["residual_scale"]
        spec = {"input_dim": train["x"].shape[1], "width": arch["width"], "depth": variant["depth"],
                "residual": variant["residual"], "residual_scale": scale}
        for seed in cfg["seeds"]:
            rec = save_run("architectures", variant["name"], train_one(train, validation, spec, "adam", adam_lr, seed, arch["epochs"], arch["batch_size"], target))
            records.append(rec); print("Completed", rec["group"], rec["name"], seed, flush=True)
    reg = cfg["regularization"]
    ids = set(read_json(DROOT / "data_manifest.json")["regularization_train_ids"])
    keep = np.array([v in ids for v in train["ids"]])
    small = {k: v[keep] for k, v in train.items()}
    assert len(small["x"]) == reg["train_n"]
    for variant in reg["variants"]:
        spec = {"input_dim": train["x"].shape[1], "width": reg["width"], "depth": reg["depth"], "dropout": variant["dropout"]}
        for seed in cfg["seeds"]:
            rec = save_run("regularization", variant["name"], train_one(small, validation, spec, "adam", adam_lr, seed,
                           reg["epochs"], reg["batch_size"], target, l2=variant["l2"], patience=reg["patience"], min_delta=reg["min_delta"]))
            records.append(rec); print("Completed", rec["group"], rec["name"], seed, "stop", rec["virtual_stop_epoch"], flush=True)
    run_paths = sorted((DROOT / "runs").glob("*"))
    save_json(DROOT / "selection_frozen.json", {"code": code_fingerprints(), "data_manifest_sha256": file_sha256(DROOT / "data_manifest.json"),
              "optimizer_choices": choices, "shared_architecture_and_regularization_adam_lr": adam_lr,
              "audit_opened": False, "n_runs": len(records), "all_architectures_and_regularizers_evaluated_without_audit_selection": True,
              "run_sha256": {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in run_paths}})
    print("Frozen before audit:", len(records), "training runs", flush=True)


def audit(target_name="auto"):
    frozen = read_json(DROOT / "selection_frozen.json")
    assert frozen["code"] == code_fingerprints()
    assert frozen["data_manifest_sha256"] == file_sha256(DROOT / "data_manifest.json")
    for path, digest in frozen["run_sha256"].items(): assert file_sha256(ROOT / path) == digest
    target = device(target_name)
    if target.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    data = load_role("audit")
    x = torch.as_tensor(data["x"], device=target); y = torch.as_tensor(data["y"], device=target)
    rows, predictions = [], {}
    for path in sorted((DROOT / "runs").glob("*.json")):
        rec = read_json(path)
        if rec["group"] == "optimizers" and rec["lr"] != frozen["optimizer_choices"][rec["optimizer"]]["lr"]: continue
        for label, checkpoint in rec["weights"].items():
            model = TeachingMLP(**rec["spec"]).to(target)
            with np.load(ROOT / checkpoint["path"]) as arrays:
                model.load_state_dict({k: torch.as_tensor(arrays[k], device=target) for k in arrays.files})
            result = assess(model, x, y)
            key = path.stem + "__" + label; predictions[key] = result.pop("p")
            epoch = len(rec["history"]) - 1 if label == "final" else rec["early_epoch"]
            rows.append({"key": key, "group": rec["group"], "name": rec["name"], "seed": rec["seed"],
                         "checkpoint": label, "epoch": epoch, "parameters": rec["parameters"],
                         "lr": rec["lr"], "n": len(y), "accuracy": result["accuracy"], "ce": result["ce"],
                         "checkpoint_path": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"]})
    np.savez_compressed(DROOT / "audit_predictions.npz", **predictions)
    pd.DataFrame(rows).to_csv(DROOT / "audit_metrics.csv", index=False)
    save_json(DROOT / "audit_metadata.json", {"selection_frozen_sha256": file_sha256(DROOT / "selection_frozen.json"),
              "audit_features_sha256": file_sha256(DROOT / "audit_features.npz"), "code": code_fingerprints(),
              "rows": len(rows), "predictions_sha256": file_sha256(DROOT / "audit_predictions.npz"),
              "float32_policy": "TF32 disabled; no mixed precision; same policy as training validation",
              "official_test_opened": False, "role": "single internal audit after freezing; no refit or selection afterwards"})
    print("Internal audit complete:", len(rows), "checkpoints; official IMDb test unused", flush=True)
