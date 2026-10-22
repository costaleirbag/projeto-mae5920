from __future__ import annotations

import math
import platform
import resource
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .common import (ROOT, config, device, environment, read_json, release_memory,
                     save_json, synchronize)
from .data import load_splits
from .metrics import evaluate, save_predictions
from .models import EncodedReviews, PadBatch, TextCNN, Vocabulary


def model_logits(model, inputs):
    result = model(**inputs)
    return result if isinstance(result, torch.Tensor) else result.logits


@torch.inference_mode()
def predict(model, loader, destination):
    model.eval()
    outputs = []
    for inputs, _ in loader:
        inputs = {k: v.to(destination) for k, v in inputs.items()}
        z = model_logits(model, inputs)
        outputs.append((z[:, 1] - z[:, 0]).cpu().numpy())
    return np.concatenate(outputs)


def make_bert(model_id, revision, train_last_n_layers, gradient_checkpointing=False):
    from transformers import BertForSequenceClassification
    # MPS SDPA does not support training-time attention dropout in this runtime.
    # Eager attention preserves dropout and has the same model definition.
    model = BertForSequenceClassification.from_pretrained(model_id, revision=revision, num_labels=2, attn_implementation="eager")
    if train_last_n_layers >= 0:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if train_last_n_layers:
            for layer in model.bert.encoder.layer[-train_last_n_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        for parameter in model.bert.pooler.parameters():
            parameter.requires_grad_(True)
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(True)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def prepare_inputs(kind, splits, cfg, artifact):
    if kind == "cnn":
        vocab = Vocabulary.fit(splits["train"].clean_text, cfg["max_tokens"])
        save_json(artifact / "vocabulary.json", vocab.token_to_id)
        sequences = {name: [vocab.encode(t, cfg["max_length"]) for t in df.clean_text] for name, df in splits.items()}
        model = TextCNN(len(vocab.token_to_id), cfg["embedding_dim"], cfg["filters"], cfg["kernels"], cfg["dropout"])
        return model, sequences, 0, {"vocab_size": len(vocab.token_to_id)}
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    revision = cfg.get("revision") or HfApi().model_info(cfg["model_id"]).sha
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=revision)
    tokenizer.save_pretrained(artifact)
    sequences = {name: tokenizer(df.clean_text.tolist(), truncation=True, max_length=cfg["max_length"],
                                 padding=False)["input_ids"] for name, df in splits.items()}
    model = make_bert(cfg["model_id"], revision, cfg["train_last_n_layers"], cfg["gradient_checkpointing"])
    return model, sequences, tokenizer.pad_token_id, {"pretrained_model_id": cfg["model_id"], "revision": revision,
                 "fine_tuning_policy": ("All encoder blocks, embeddings, pooler and classifier trained." if cfg['train_last_n_layers'] < 0 else f"Train last {cfg['train_last_n_layers']} encoder blocks, pooler and classifier. Other weights frozen.")}


def run(kind):
    full_cfg, splits = config(), load_splits()
    cfg, destination = full_cfg[kind], device()
    artifact = ROOT / "artifacts" / kind
    artifact.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model, sequences, pad_id, extra = prepare_inputs(kind, splits, cfg, artifact)
    collate = PadBatch(kind, pad_id)
    datasets = {name: EncodedReviews(seqs, splits[name].label) for name, seqs in sequences.items()}
    rng = torch.Generator().manual_seed(full_cfg["seed"])
    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=collate, generator=rng, num_workers=0)
    eval_loaders = {name: DataLoader(ds, batch_size=cfg["batch_size"] * 2, collate_fn=collate, num_workers=0)
                    for name, ds in datasets.items()}
    model.to(destination)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["lr"], weight_decay=.01)
    accumulation = cfg.get("accumulation_steps", 1)
    total_updates = math.ceil(len(train_loader) / accumulation) * cfg["epochs"]
    warmup = max(1, int(.1 * total_updates))
    if kind == "bert":
        def schedule(step):
            return step / warmup if step < warmup else max(0, (total_updates - step) / max(1, total_updates - warmup))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    else:
        scheduler = None
    loss_fn = nn.CrossEntropyLoss()
    best_score, best_epoch, patience = -1., None, 0
    history = []
    print(kind, "device", destination, "trainable", sum(p.numel() for p in parameters), flush=True)
    training_start = time.perf_counter()
    for epoch in range(cfg["epochs"]):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        summed_loss, seen = 0., 0
        for step, (inputs, labels) in enumerate(train_loader):
            inputs = {k: v.to(destination) for k, v in inputs.items()}
            labels = labels.to(destination)
            logits = model_logits(model, inputs)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {kind} loss at epoch {epoch}, batch {step}")
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            (loss / group_size).backward()
            summed_loss += float(loss.detach().cpu()) * len(labels)
            seen += len(labels)
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 100 == 0:
                print(kind, "epoch", epoch + 1, "batch", step + 1, "/", len(train_loader),
                      "loss", round(summed_loss / seen, 4), "elapsed_s", round(time.perf_counter() - epoch_start, 1), flush=True)
        val_logits = predict(model, eval_loaders["validation"], destination)
        score = f1_score(splits["validation"].label, val_logits >= 0, average="macro")
        synchronize()
        row = {"epoch": epoch + 1, "training_loss": summed_loss / seen,
               "validation_f1_macro": float(score), "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        print(kind, row, flush=True)
        save_json(ROOT / "results" / kind / "history.json", history)
        if score > best_score:
            best_score, best_epoch, patience = score, epoch + 1, 0
            if kind == "bert":
                model.save_pretrained(artifact, safe_serialization=True)
            else:
                torch.save(model.state_dict(), artifact / "model.pt")
        else:
            patience += 1
        if patience >= cfg["patience"]:
            break
    training_seconds = time.perf_counter() - training_start
    del optimizer, parameters
    model.to("cpu")
    del model
    release_memory()
    if kind == "bert":
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(artifact, local_files_only=True, attn_implementation="eager")
    else:
        vocab = read_json(artifact / "vocabulary.json")
        model = TextCNN(len(vocab), cfg["embedding_dim"], cfg["filters"], cfg["kernels"], cfg["dropout"])
        model.load_state_dict(torch.load(artifact / "model.pt", map_location="cpu", weights_only=True))
    model.to(destination)
    inference_start = time.perf_counter()
    logits_by_split = {}
    for name, loader in eval_loaders.items():
        split_start = time.perf_counter()
        logits_by_split[name] = predict(model, loader, destination)
        print(kind, name, "evaluated", len(logits_by_split[name]), "seconds", round(time.perf_counter() - split_start, 1), flush=True)
    metadata = {"environment": environment(), "config": cfg, **extra,
                "best_epoch": best_epoch, "history": history, "selection_split": "validation",
                "training_seconds": training_seconds, "total_seconds": time.perf_counter() - start,
                "all_splits_inference_seconds": time.perf_counter() - inference_start,
                "host_peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024) / 1e9,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "input_scope": f"Prefix truncated to {cfg['max_length']} {'word tokens' if kind == 'cnn' else 'WordPiece tokens including specials'}"}
    # Count training parameters from the training policy without allocating another model.
    metadata["trainable_parameters"] = _trainable_count(kind, model, cfg)
    save_json(artifact / "experiment_config.json", cfg)
    save_predictions(kind, splits, logits_by_split, metadata)


def _trainable_count(kind, model, cfg):
    if kind == "cnn" or cfg.get("train_last_n_layers", -1) < 0:
        return sum(p.numel() for p in model.parameters())
    count = sum(p.numel() for p in model.bert.pooler.parameters()) + sum(p.numel() for p in model.classifier.parameters())
    n = cfg["train_last_n_layers"]
    if n:
        count += sum(p.numel() for block in model.bert.encoder.layer[-n:] for p in block.parameters())
    return count


def load_model(kind, destination=None):
    cfg = read_json(ROOT / "artifacts" / kind / "experiment_config.json")
    if kind == "bert":
        from transformers import AutoTokenizer, BertForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "artifacts" / kind, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(ROOT / "artifacts" / kind, local_files_only=True, attn_implementation="eager")
    else:
        tokenizer = Vocabulary(read_json(ROOT / "artifacts" / kind / "vocabulary.json"))
        model = TextCNN(len(tokenizer.token_to_id), cfg["embedding_dim"], cfg["filters"], cfg["kernels"], cfg["dropout"])
        model.load_state_dict(torch.load(ROOT / "artifacts" / kind / "model.pt", weights_only=True, map_location="cpu"))
    model.to(destination or device()).eval()
    return model, tokenizer, cfg


def predict_texts(kind, texts):
    model, tokenizer, cfg = load_model(kind)
    if kind == "cnn":
        sequences = [tokenizer.encode(text, cfg["max_length"]) for text in texts]
        pad = 0
    else:
        sequences = tokenizer(texts, truncation=True, max_length=cfg["max_length"])["input_ids"]
        pad = tokenizer.pad_token_id
    loader = DataLoader(EncodedReviews(sequences, [0] * len(texts)), batch_size=cfg["batch_size"] * 2,
                        collate_fn=PadBatch(kind, pad))
    logits = predict(model, loader, next(model.parameters()).device)
    model.to("cpu")
    del model
    release_memory()
    return logits
