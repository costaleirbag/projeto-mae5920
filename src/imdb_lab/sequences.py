"""TextCNN, frozen encoder probes and full fine-tuning on common IMDb excerpts.

Training functions never load the audit role. Model revisions, checkpoints,
validation predictions and choices are frozen before audit evaluation.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import math
import platform
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, file_sha256, read_json, save_json
from .models import TextCNN, Vocabulary

SROOT = ROOT / "results/sequences"
INPUT = ROOT / ".cache/sequence_inputs"
WEIGHTS = ROOT / "artifacts/sequences"


def settings():
    return read_json(ROOT / "sequence_config.json")


def excerpt(text, words):
    return " ".join(text.split()[:words])


def fingerprint():
    return {p: file_sha256(ROOT / p) for p in ["sequence_config.json", "run_sequences.py", "src/imdb_lab/sequences.py", "src/imdb_lab/models.py"]}


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def configure(device):
    torch.set_num_threads(settings()["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    return target


def sync(target):
    if target.type == "cuda": torch.cuda.synchronize()


def autocast(target, enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled and target.type == "cuda" else nullcontext()


def prepare():
    """Prepare common text roles. Only train/validation are needed by train_all."""
    cfg = settings(); source = ROOT / "data/train.parquet"
    frame = pd.read_parquet(source)
    assert frame.official_split.eq("train").all()
    assert frame.id.is_unique and frame.text_sha256.is_unique
    old = read_json(ROOT / "results/didactic/data_manifest.json")
    parts = {}
    for role in ["train", "validation"]:
        ids = old["roles"][role]["ids"]
        parts[role] = frame.set_index("id").loc[ids].reset_index()
        assert len(parts[role]) == cfg["data"][role + "_n"]
    excluded = set().union(*(set(old["roles"][r]["ids"]) for r in old["roles"]))
    pool = frame[~frame.id.isin(excluded)]
    rng = np.random.default_rng(cfg["split_seed"])
    chosen = np.concatenate([rng.choice(pool[pool.label.eq(y)].index, cfg["data"]["audit_n"] // 2, replace=False) for y in [0, 1]])
    parts["audit"] = frame.loc[chosen].sort_values("id").reset_index(drop=True)
    assert not set(parts["audit"].id) & excluded
    INPUT.mkdir(parents=True, exist_ok=True); SROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"source_sha256": file_sha256(source), "source_role": "original clean official train only", "config_sha256": file_sha256(ROOT / "sequence_config.json"),
                "previous_didactic_manifest_sha256": file_sha256(ROOT / "results/didactic/data_manifest.json"),
                "new_audit_disjoint_from_all_previous_didactic_roles": True, "official_test_opened": False,
                "common_input": cfg["common_input"], "roles": {}}
    for role, rows in parts.items():
        texts = rows.clean_text.map(lambda t: excerpt(t, cfg["common_input"]["whitespace_words"]))
        public_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        local = pd.DataFrame({"id": rows.id, "label": rows.label, "text": texts})
        local.to_parquet(INPUT / f"{role}.parquet", index=False)
        manifest["roles"][role] = {"n": len(rows), "ids": rows.id.tolist(), "text_sha256": rows.text_sha256.tolist(),
                                    "excerpt_sha256": public_hashes,
                                    "labels_sha256": array_hash(rows.label.to_numpy(dtype=np.int64)),
                                    "label_counts": {str(k): int(v) for k, v in rows.label.value_counts().items()},
                                    "input_sha256": file_sha256(INPUT / f"{role}.parquet")}
        if role != "audit": manifest["roles"][role]["labels"] = rows.label.astype(int).tolist()
    vocab = Vocabulary.fit(parts["train"].clean_text.map(lambda t: excerpt(t, cfg["common_input"]["whitespace_words"])), cfg["cnn"]["max_vocabulary"])
    save_json(SROOT / "vocabulary.json", vocab.token_to_id)
    manifest["vocabulary_sha256"] = file_sha256(SROOT / "vocabulary.json")
    save_json(SROOT / "data_manifest.json", manifest)
    print("Prepared common excerpts", {k: len(v) for k, v in parts.items()}, flush=True)


def load_role(role):
    path = INPUT / f"{role}.parquet"; manifest = read_json(SROOT / "data_manifest.json")
    assert file_sha256(path) == manifest["roles"][role]["input_sha256"]
    df = pd.read_parquet(path)
    assert df.id.tolist() == manifest["roles"][role]["ids"]
    return df


def orders(n, epochs, seed):
    return [np.random.default_rng(seed + 1009 * e).permutation(n) for e in range(epochs)]


def batches(order, size):
    return [order[i:i + size] for i in range(0, len(order), size)]


def cnn_inputs(frame, target):
    vocab = Vocabulary(read_json(SROOT / "vocabulary.json")); cap = settings()["common_input"]["token_cap"]
    seq = [vocab.encode(t, cap) for t in frame.text]
    width = max(5, max(map(len, seq)))
    x = torch.zeros((len(seq), width), dtype=torch.long, device=target)
    lengths = torch.tensor(list(map(len, seq)), device=target)
    for i, row in enumerate(seq): x[i, :len(row)] = torch.tensor(row, device=target)
    return {"input_ids": x, "lengths": lengths}, {"token_cap": cap, "truncated_n": sum(len(vocab.encode(t, cap + 1)) > cap for t in frame.text),
                                                 "mean_tokens": float(lengths.float().mean().cpu())}


def encoder_inputs(tokenizer, frame, target):
    cap = settings()["common_input"]["token_cap"]
    raw = tokenizer(frame.text.tolist(), truncation=False, padding=False, return_special_tokens_mask=True)
    encoded = tokenizer(frame.text.tolist(), truncation=True, max_length=cap, padding="max_length", return_special_tokens_mask=True, return_tensors="pt")
    inputs = {k: v.to(target) for k, v in encoded.items() if k != "special_tokens_mask"}
    inputs["pool_mask"] = (encoded.attention_mask.bool() & ~encoded.special_tokens_mask.bool()).to(target)
    return inputs, {"token_cap": cap, "truncated_n": sum(len(row) > cap for row in raw["input_ids"]),
                    "mean_tokens": float(encoded.attention_mask.sum(1).float().mean())}


def indexed(inputs, indices):
    return {k: v[indices] for k, v in inputs.items()}


def masked_mean(hidden, mask):
    weight = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weight).sum(1) / weight.sum(1).clamp_min(1)


class MeanEncoder(nn.Module):
    def __init__(self, backbone, hidden_size, dropout=0.1):
        super().__init__(); self.backbone = backbone; self.dropout = nn.Dropout(dropout); self.classifier = nn.Linear(hidden_size, 2)

    def features(self, input_ids, attention_mask, pool_mask, **kwargs):
        result = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return masked_mean(result.last_hidden_state, pool_mask)

    def forward(self, **inputs):
        return self.classifier(self.dropout(self.features(**inputs)))


def load_encoder(name, pretrained, dropout=0.1):
    from transformers import AutoConfig, AutoModel
    cfg = settings(); model = next(x for x in cfg["encoders"]["models"] if x["name"] == name)
    path = pretrained / model["model_id"].split("/")[-1]
    conf = AutoConfig.from_pretrained(path, local_files_only=True)
    if hasattr(conf, "reference_compile"): conf.reference_compile = False
    kwargs = {"add_pooling_layer": False} if conf.model_type == "bert" else {}
    backbone = AutoModel.from_pretrained(path, config=conf, local_files_only=True, attn_implementation="sdpa", **kwargs)
    hidden = getattr(conf, "hidden_size", getattr(conf, "dim", None))
    return MeanEncoder(backbone, hidden, dropout), conf


def initialize_cnn(spec, seed, target):
    torch.manual_seed(seed)
    vocab_n = len(read_json(SROOT / "vocabulary.json"))
    model = TextCNN(vocab_n, spec["embedding_dim"], spec["filters"], spec["kernels"], spec["dropout"])
    nn.init.normal_(model.embedding.weight, std=0.1)
    with torch.no_grad(): model.embedding.weight[0].zero_()
    for conv in model.convs:
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu"); nn.init.zeros_(conv.bias)
    nn.init.xavier_uniform_(model.classifier.weight); nn.init.zeros_(model.classifier.bias)
    return model.to(target)


def optimizer_for(model, name, lr):
    if name == "adam": return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    if name == "adamw":
        # Exclude biases and normalization parameters from decoupled decay.
        decay, no_decay = [], []
        for key, p in model.named_parameters():
            (no_decay if p.ndim <= 1 or key.endswith(".bias") else decay).append(p)
        return torch.optim.AdamW([{"params": decay, "weight_decay": settings()["encoders"]["full"]["weight_decay"]},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9 if name == "momentum" else 0.0, weight_decay=0)


@torch.inference_mode()
def predict(model, inputs, labels, batch_size, target, mixed=False, features=False):
    model.eval(); outputs = []
    n = len(labels)
    for indices in batches(np.arange(n), batch_size):
        with autocast(target, mixed):
            part = indexed(inputs, indices)
            z = model.features(**part) if features else model(**part)
        outputs.append(z.float().cpu())
    z = torch.cat(outputs)
    if features: return z.numpy()
    y = labels.cpu()
    return {"ce": float(F.cross_entropy(z, y)), "accuracy": float((z.argmax(1) == y).float().mean()), "logits": z.numpy()}


class Probe(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__(); self.dropout = nn.Dropout(dropout); self.classifier = nn.Linear(hidden, 2)
    def forward(self, x): return self.classifier(self.dropout(x))


def train_one(model, train, validation, spec, seed, target):
    """Shared trainer; no audit argument, no patience-based budget changes."""
    torch.manual_seed(seed + 40000)  # Paired dropout stream, independent of parameter construction.
    tx, ty = train; vx, vy = validation
    initial = state_hash(model); initial_head = state_hash(model.classifier)
    sequence = orders(len(ty), spec["epochs"], seed)
    opt = optimizer_for(model, spec["optimizer"], spec["lr"])
    updates = spec["epochs"] * math.ceil(len(ty) / spec["batch_size"])
    warmup = max(1, round(updates * spec.get("warmup_fraction", 0))) if spec.get("warmup_fraction") else 0
    scheduler = None
    if warmup:
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: k / warmup if k < warmup else max(0.0, (updates - k) / max(1, updates - warmup)))
    batch_eval = spec.get("eval_batch_size", 64)
    history, best_loss, best_epoch, best_state, best_predictions = [], math.inf, None, None, None
    started = time.perf_counter(); train_seconds = 0.0; completed_updates = 0
    if target.type == "cuda": torch.cuda.reset_peak_memory_stats()
    for epoch in range(spec["epochs"] + 1):
        if epoch:
            model.train(); sync(target); tick = time.perf_counter(); summed = 0.0; seen = 0
            for step, indices in enumerate(batches(sequence[epoch - 1], spec["batch_size"])):
                opt.zero_grad(set_to_none=True)
                with autocast(target, spec.get("mixed", False)):
                    logits = model(**indexed(tx, indices)); loss = F.cross_entropy(logits.float(), ty[indices])
                if not bool(torch.isfinite(loss)): raise FloatingPointError(f"Nonfinite loss: {spec}, epoch={epoch}, step={step}")
                loss.backward()
                if spec.get("gradient_clip_norm"):
                    norm = nn.utils.clip_grad_norm_(model.parameters(), spec["gradient_clip_norm"], error_if_nonfinite=True)
                opt.step()
                if scheduler: scheduler.step()
                summed += float(loss.detach().cpu()) * len(indices); seen += len(indices); completed_updates += 1
            sync(target); train_seconds += time.perf_counter() - tick
        tm = predict(model, tx, ty, batch_eval, target, spec.get("mixed", False))
        vm = predict(model, vx, vy, batch_eval, target, spec.get("mixed", False))
        row = {"epoch": epoch, "updates": completed_updates, "train_ce": tm["ce"], "train_accuracy": tm["accuracy"],
               "validation_ce": vm["ce"], "validation_accuracy": vm["accuracy"], "training_seconds": train_seconds,
               "learning_rate": opt.param_groups[0]["lr"]}
        if epoch: row["online_training_ce"] = summed / seen
        history.append(row)
        # The untrained epoch zero is diagnostic only, never a selected checkpoint.
        if epoch and vm["ce"] < best_loss:
            best_loss, best_epoch, best_state = vm["ce"], epoch, snapshot(model)
            best_predictions = {"train": tm["logits"], "validation": vm["logits"]}
        print("EPOCH", spec["run_id"], seed, row, flush=True)
    return {"history": history, "seed": seed, "spec": spec, "initial_state_sha256": initial,
            "initial_head_sha256": initial_head,
            "batch_order_sha256": array_hash(np.concatenate(sequence)), "best_epoch": best_epoch, "best_validation_ce": best_loss,
            "training_seconds": train_seconds, "total_seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if target.type == "cuda" else None,
            "best_state": best_state, "best_predictions": best_predictions}


def save_run(result):
    spec = result["spec"]; run_id = f"{spec['run_id']}__s{result['seed']}"
    WEIGHTS.mkdir(parents=True, exist_ok=True); (SROOT / "runs").mkdir(exist_ok=True)
    path = WEIGHTS / f"{run_id}.pt"; torch.save(result.pop("best_state"), path)
    pred = SROOT / "runs" / f"{run_id}__validation.npz"
    np.savez_compressed(pred, **result.pop("best_predictions"))
    result["run_id"] = run_id
    result["checkpoint"] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    result["predictions"] = {"path": pred.relative_to(ROOT).as_posix(), "sha256": file_sha256(pred)}
    save_json(SROOT / "runs" / f"{run_id}.json", result)
    return result


def cached_run(run_id, seed):
    path = SROOT / "runs" / f"{run_id}__s{seed}.json"
    if not path.exists(): return None
    obj = read_json(path)
    assert file_sha256(ROOT / obj["checkpoint"]["path"]) == obj["checkpoint"]["sha256"]
    assert file_sha256(ROOT / obj["predictions"]["path"]) == obj["predictions"]["sha256"]
    return obj


def select_rate(records):
    rates = sorted({r["spec"]["lr"] for r in records})
    means = {str(lr): float(np.mean([r["best_validation_ce"] for r in records if r["spec"]["lr"] == lr])) for lr in rates}
    return min(rates, key=lambda lr: (means[str(lr)], lr)), means


def model_metadata(conf):
    names = ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "vocab_size", "max_position_embeddings", "local_attention", "global_attn_every_n_layers", "position_embedding_type", "hidden_activation", "hidden_act", "reference_compile"]
    return {k: getattr(conf, k) for k in names if hasattr(conf, k)}


def validate_pretrained(pretrained):
    pinned = read_json(SROOT / "models_pinned.json")
    for model_id, record in pinned.items():
        for filename, info in record["files"].items():
            path = pretrained / model_id.split("/")[-1] / filename
            assert path.stat().st_size == info["bytes"]
            assert file_sha256(path) == info["sha256"], "Pretrained file changed"


def pilot(pretrained, device="cuda"):
    """Compatibility and timing only, first training minibatch; not model selection."""
    from transformers import AutoTokenizer
    target = configure(device); train = load_role("train").iloc[:32]
    y = torch.tensor(train.label.tolist(), device=target)
    output = []
    for entry in settings()["encoders"]["models"]:
        torch.manual_seed(17)
        tokenizer = AutoTokenizer.from_pretrained(pretrained / entry["model_id"].split("/")[-1], local_files_only=True)
        inputs, meta = encoder_inputs(tokenizer, train, target)
        model, conf = load_encoder(entry["name"], pretrained); model.to(target).train()
        opt = optimizer_for(model, "adamw", 0.00001)
        tick = time.perf_counter()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with autocast(target, True): z = model(**inputs); loss = F.cross_entropy(z.float(), y)
            loss.backward(); opt.step()
        sync(target)
        output.append({"name": entry["name"], "seconds_per_update": (time.perf_counter() - tick) / 3,
                       "loss_finite": bool(torch.isfinite(loss)), "parameters": sum(p.numel() for p in model.parameters()),
                       "architecture": model_metadata(conf)})
        print("PILOT", output[-1], flush=True)
        del model, opt; gc.collect(); torch.cuda.empty_cache()
    save_json(SROOT / "pilot.json", {"purpose": "Compatibility and timing on 32 training examples only, no validation or audit selection", "results": output})


def train_all(pretrained, device="cuda"):
    from transformers import AutoTokenizer
    cfg = settings(); target = configure(device)
    validate_pretrained(pretrained)
    assert not (SROOT / "selection_frozen.json").exists(), "Preserve frozen study; use a new study directory"
    code = fingerprint(); started_path = SROOT / "protocol_started.json"
    if started_path.exists(): assert read_json(started_path)["code"] == code, "Code changed during a partially completed protocol"
    else:
        save_json(started_path, {"code": code, "data_manifest_sha256": file_sha256(SROOT / "data_manifest.json"),
                "audit_opened": False, "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                "numpy": np.__version__, "python": platform.python_version(), "accelerator": torch.cuda.get_device_name(0) if target.type == "cuda" else "CPU",
                "tf32": False, "deterministic_algorithms": True, "encoder_attention": "sdpa", "encoder_training": "BF16 autocast, FP32 parameters and optimizer states"})
    train, validation = load_role("train"), load_role("validation")
    ty = torch.tensor(train.label.tolist(), device=target); vy = torch.tensor(validation.label.tolist(), device=target)
    tx, tmeta = cnn_inputs(train, target); vx, vmeta = cnn_inputs(validation, target)
    save_json(SROOT / "cnn_tokenization.json", {"train": tmeta, "validation": vmeta})
    records, selections, candidate_means = [], {}, {}
    base = {k: cfg["cnn"][k] for k in ["embedding_dim", "filters", "kernels", "dropout"]}
    for opt, rates in cfg["cnn"]["learning_rates"].items():
        group = []
        for lr in rates:
            run_id = f"cnn_optimizer__{opt}__lr{lr:g}"
            spec = dict(base, family="cnn", group="optimizer", variant=opt, optimizer=opt, lr=lr, run_id=run_id,
                        epochs=cfg["cnn"]["epochs"], batch_size=cfg["cnn"]["batch_size"], gradient_clip_norm=cfg["cnn"]["gradient_clip_norm"])
            for seed in cfg["seeds"]:
                obj = cached_run(run_id, seed)
                if obj is None:
                    model = initialize_cnn(base, seed, target)
                    obj = save_run(train_one(model, (tx, ty), (vx, vy), spec, seed, target)); del model
                group.append(obj); records.append(obj)
        lr, means = select_rate(group); selections[f"cnn_{opt}"] = lr; candidate_means[f"cnn_{opt}"] = means
    for variant in cfg["cnn"]["variants"] + [{"name": f"dropout_{drop:g}", "dropout": drop} for drop in cfg["cnn"]["dropout_alternatives"]]:
        name = variant["name"]; model_spec = dict(base, **{k: v for k, v in variant.items() if k != "name"})
        run_id = f"cnn_variant__{name}"
        spec = dict(model_spec, family="cnn", group="dropout" if name.startswith("dropout") else "architecture", variant=name,
                    optimizer="adam", lr=selections["cnn_adam"], run_id=run_id,
                    epochs=cfg["cnn"]["epochs"], batch_size=cfg["cnn"]["batch_size"], gradient_clip_norm=cfg["cnn"]["gradient_clip_norm"])
        for seed in cfg["seeds"]:
            obj = cached_run(run_id, seed)
            if obj is None:
                model = initialize_cnn(model_spec, seed, target)
                obj = save_run(train_one(model, (tx, ty), (vx, vy), spec, seed, target)); del model
            records.append(obj)
    del tx, vx; gc.collect()
    metadata = read_json(SROOT / "models_pinned.json")
    for entry in cfg["encoders"]["models"]:
        name = entry["name"]; tokenizer = AutoTokenizer.from_pretrained(pretrained / entry["model_id"].split("/")[-1], local_files_only=True)
        tokenizer.save_pretrained(SROOT / "tokenizers" / name)
        tx, tm = encoder_inputs(tokenizer, train, target); vx, vm = encoder_inputs(tokenizer, validation, target)
        save_json(SROOT / f"{name}_tokenization.json", {"train": tm, "validation": vm})
        # Stock encoder features: encoder dropout off; no fitted representation step.
        cache = INPUT / f"{name}_frozen_features.npz"
        if not cache.exists():
            torch.manual_seed(17); frozen, conf = load_encoder(name, pretrained); frozen.to(target)
            sync(target); extraction_start = time.perf_counter()
            a = predict(frozen, tx, ty, 64, target, True, features=True)
            b = predict(frozen, vx, vy, 64, target, True, features=True)
            sync(target); extraction_seconds = time.perf_counter() - extraction_start
            np.savez_compressed(cache, train=a, validation=b)
            save_json(SROOT / f"{name}_feature_extraction.json", {"train_validation_seconds": extraction_seconds,
                        "cache_sha256": file_sha256(cache), "roles": ["train", "validation"],
                        "precision": "BF16 encoder, FP32 stored features", "encoder_in_eval_mode": True})
            save_json(SROOT / f"{name}_architecture.json", dict(model_metadata(conf), parameters=sum(p.numel() for p in frozen.parameters()),
                        backbone_parameters=sum(p.numel() for p in frozen.backbone.parameters()), hidden_size=a.shape[1]))
            del frozen; gc.collect(); torch.cuda.empty_cache()
        assert file_sha256(cache) == read_json(SROOT / f"{name}_feature_extraction.json")["cache_sha256"]
        with np.load(cache) as data:
            px = {"x": torch.tensor(data["train"], device=target)}; qx = {"x": torch.tensor(data["validation"], device=target)}
        for policy in ["probe", "full"]:
            group = []; policy_cfg = cfg["encoders"][policy]
            for lr in policy_cfg["learning_rates"]:
                run_id = f"{name}_{policy}__lr{lr:g}"
                spec = {"family": name, "group": policy, "variant": policy, "optimizer": "adam" if policy == "probe" else "adamw", "lr": lr,
                        "run_id": run_id, "epochs": policy_cfg["epochs"], "batch_size": policy_cfg["batch_size"],
                        "mixed": policy == "full", "eval_batch_size": 64}
                if policy == "full": spec.update(gradient_clip_norm=policy_cfg["gradient_clip_norm"], warmup_fraction=policy_cfg["warmup_fraction"])
                for seed in cfg["seeds"]:
                    obj = cached_run(run_id, seed)
                    if obj is None:
                        torch.manual_seed(seed)
                        if policy == "probe":
                            model = Probe(px["x"].shape[1]).to(target); pair = ((px, ty), (qx, vy))
                        else:
                            model, _ = load_encoder(name, pretrained, policy_cfg["head_dropout"])
                            # Same starting linear head as the frozen probe for this seed.
                            torch.manual_seed(seed)
                            model.classifier = nn.Linear(model.classifier.in_features, 2)
                            model.to(target); pair = ((tx, ty), (vx, vy))
                        obj = save_run(train_one(model, *pair, spec, seed, target)); del model; gc.collect(); torch.cuda.empty_cache()
                    group.append(obj); records.append(obj)
            lr, means = select_rate(group); selections[f"{name}_{policy}"] = lr; candidate_means[f"{name}_{policy}"] = means
        del tx, vx, px, qx; gc.collect(); torch.cuda.empty_cache()
    # Sparse comparator uses the same excerpt and training/validation roles.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib
    bc = cfg["baseline"]; vectorizer = TfidfVectorizer(max_features=bc["max_features"], min_df=bc["min_df"], ngram_range=tuple(bc["ngram_range"]), sublinear_tf=True)
    bx = vectorizer.fit_transform(train.text); bv = vectorizer.transform(validation.text)
    candidates = []
    for c in bc["c_grid"]:
        model = LogisticRegression(C=c, max_iter=1000, solver="liblinear", random_state=17).fit(bx, train.label)
        v = model.decision_function(bv); ce = float(np.logaddexp(0, v).mean() - np.mean(validation.label.to_numpy() * v))
        candidates.append((ce, c, model))
    ce, c, chosen = min(candidates, key=lambda x: x[:2])
    WEIGHTS.mkdir(exist_ok=True); bp = WEIGHTS / "tfidf.joblib"; joblib.dump((vectorizer, chosen), bp)
    baseline = {"selected_c": c, "validation_ce": ce, "validation_candidates": [{"c": x[1], "ce": x[0]} for x in candidates],
                "checkpoint": {"path": bp.relative_to(ROOT).as_posix(), "sha256": file_sha256(bp)}, "features": len(vectorizer.vocabulary_)}
    selected = []
    for r in records:
        spec = r["spec"]
        key = f"cnn_{spec['optimizer']}" if spec["family"] == "cnn" else f"{spec['family']}_{spec['group']}"
        if spec["family"] == "cnn" and spec["group"] != "optimizer" or spec["lr"] == selections[key]: selected.append(r["run_id"])
    save_json(SROOT / "selection_frozen.json", {"code": code, "data_manifest_sha256": file_sha256(SROOT / "data_manifest.json"),
                "models": metadata, "selections": selections, "validation_candidate_means": candidate_means, "baseline": baseline,
                "runs": [{"run_id": r["run_id"], "sha256": file_sha256(SROOT / "runs" / f"{r['run_id']}.json"), "checkpoint": r["checkpoint"]} for r in records],
                "audit_run_ids": selected, "audit_opened": False, "official_test_opened": False})
    print("SELECTION_FROZEN", len(records), "training runs; audit checkpoints", len(selected), selections, flush=True)


def audit(pretrained, device="cuda"):
    from transformers import AutoTokenizer
    import joblib
    target = configure(device); frozen = read_json(SROOT / "selection_frozen.json")
    validate_pretrained(pretrained)
    assert frozen["code"] == fingerprint()
    assert frozen["data_manifest_sha256"] == file_sha256(SROOT / "data_manifest.json")
    frame = load_role("audit"); y = torch.tensor(frame.label.tolist(), device=target)
    all_predictions = {}; metrics = []; tokenization = {}; extraction = {}
    cnn_x, tokenization["cnn"] = cnn_inputs(frame, target)
    # Each selected checkpoint is evaluated once. No choice depends on these results.
    for family in ["cnn"] + [m["name"] for m in settings()["encoders"]["models"]]:
        if family != "cnn":
            entry = next(x for x in settings()["encoders"]["models"] if x["name"] == family)
            tokenizer = AutoTokenizer.from_pretrained(pretrained / entry["model_id"].split("/")[-1], local_files_only=True)
            x, tokenization[family] = encoder_inputs(tokenizer, frame, target)
            stock, _ = load_encoder(family, pretrained); stock.to(target)
            sync(target); extraction_start = time.perf_counter()
            features = predict(stock, x, y, 64, target, True, features=True)
            sync(target); extraction[family] = time.perf_counter() - extraction_start
            fx = {"x": torch.tensor(features, device=target)}
            np.savez_compressed(INPUT / f"{family}_audit_features.npz", x=features)
            del stock; gc.collect(); torch.cuda.empty_cache()
        for run_id in frozen["audit_run_ids"]:
            r = read_json(SROOT / "runs" / f"{run_id}.json"); spec = r["spec"]
            if spec["family"] != family: continue
            cp = ROOT / r["checkpoint"]["path"]; assert file_sha256(cp) == r["checkpoint"]["sha256"]
            if family == "cnn":
                model = initialize_cnn(spec, r["seed"], target); xx = cnn_x
            elif spec["group"] == "probe": model = Probe(features.shape[1]).to(target); xx = fx
            else: model, _ = load_encoder(family, pretrained); model.to(target); xx = x
            model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
            sync(target); tick = time.perf_counter()
            result = predict(model, xx, y, 64, target, spec.get("mixed", False)); sync(target)
            inference_seconds = time.perf_counter() - tick
            all_predictions[run_id] = result.pop("logits")
            metrics.append({"run_id": run_id, **result, "inference_seconds": inference_seconds,
                            "inference_scope": "linear head only; frozen encoder extraction separate" if spec["group"] == "probe" else "end-to-end, batching64, tokenization excluded"})
            print("AUDIT", metrics[-1], flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()
        if family != "cnn": del x, fx
    vectorizer, model = joblib.load(ROOT / frozen["baseline"]["checkpoint"]["path"])
    z = model.decision_function(vectorizer.transform(frame.text)); logits = np.stack([np.zeros_like(z), z], axis=1)
    all_predictions["tfidf"] = logits
    metrics.append({"run_id": "tfidf", "ce": float(np.logaddexp(0, z).mean() - np.mean(frame.label.to_numpy() * z)), "accuracy": float(np.mean((z >= 0) == frame.label.to_numpy()))})
    path = SROOT / "audit_predictions.npz"; np.savez_compressed(path, y=frame.label.to_numpy(), ids=frame.id.to_numpy(dtype=str), **all_predictions)
    save_json(SROOT / "audit_metrics.json", {"selection_frozen_sha256": file_sha256(SROOT / "selection_frozen.json"),
                "predictions_sha256": file_sha256(path), "metrics": metrics, "tokenization": tokenization,
                "frozen_encoder_extraction_seconds": extraction,
                "tf32": False, "encoder_precision": "BF16 autocast; FP32 head-only probe", "official_test_opened": False})
    print("AUDIT_COMPLETE", len(metrics), flush=True)
