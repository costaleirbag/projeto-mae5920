"""Ablações de comprimento, arquitetura e regularização nas divisões originais."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch import nn
from torch.nn import functional as F

from imdb_lab import sequences
from imdb_lab.common import ROOT, file_sha256, read_json, save_json
from imdb_lab.metrics import evaluate, fit_temperature
from imdb_lab.models import TextCNN, Vocabulary, tokenize
from medir_posicao import CONTRAST_FILES, CONTRAST_REV, contrast_data, synthetic_data
from treinar_pooling import masked_max, masked_mean, pair_results, pool


SEEDS = (17, 29, 42)
BATCH_SIZE = 64
EPOCHS = 8
REFERENCE = "max_400"
RECIPES = {
    "max_256": {"cap": 256},
    "max_400": {"cap": 400},
    "max_768": {"cap": 768},
    "max_integral": {"cap": None},
    "max_media_400": {"pooling": "max_media"},
    "max_media_integral": {"pooling": "max_media", "cap": None},
    "max_metades_400": {"pooling": "max_metades_posicao"},
    "largura_357_400": {"kernels": [3, 5, 7]},
    "dilatada_max_media_400": {"pooling": "max_media", "dilations": [1, 3]},
    "sem_decay_400": {"weight_decay": 0.0},
    "dropout_02_400": {"dropout": 0.2},
}
DEFAULTS = {
    "cap": 400, "kernels": [3, 4, 5], "dropout": 0.5, "pooling": "max_global",
    "dilations": [], "weight_decay": 0.01,
}


class ContextCNN(TextCNN):
    """TextCNN com as mesmas convoluções de saída e resumos alternativos."""

    def __init__(self, vocab_size, spec):
        super().__init__(vocab_size, kernels=spec["kernels"], dropout=spec["dropout"])
        self.summary = spec["pooling"]
        self.context = nn.ModuleList(
            nn.Conv1d(128, 128, 5, padding=2 * dilation, dilation=dilation)
            for dilation in spec["dilations"]
        )
        self.classifier = nn.Linear(2 * self.classifier.in_features, 2)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids).transpose(1, 2)
        if self.context:
            token_positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            valid_tokens = token_positions[None, None, :] < lengths[:, None, None]
            for layer in self.context:
                embedded = (embedded + 0.5 * layer(embedded).relu()).masked_fill(~valid_tokens, 0)
        summaries = []
        for conv, width in zip(self.convs, self.kernels, strict=True):
            features = conv(embedded).relu()
            valid_count = (lengths - width + 1).clamp_min(1)
            if self.summary == "max_media":
                positions = torch.arange(features.shape[-1], device=features.device)
                valid = positions[None, None, :] < valid_count[:, None, None]
                summary = torch.cat((masked_max(features, valid), masked_mean(features, valid)), 1)
            else:
                summary = pool(features, valid_count, self.summary)
            summaries.append(summary)
        return self.classifier(self.dropout(torch.cat(summaries, 1)))


def model_for(vocab_size, spec, seed, device):
    torch.manual_seed(seed)
    if spec["pooling"] == "max_global":
        model = TextCNN(vocab_size, kernels=spec["kernels"], dropout=spec["dropout"])
    else:
        model = ContextCNN(vocab_size, spec)
    return model.to(device)


def plan():
    sources = [
        "seminario/treinar_cnn_comprimento.py", "seminario/treinar_pooling.py",
        "seminario/medir_posicao.py", "seminario/casos.json",
        "src/imdb_lab/models.py", "src/imdb_lab/sequences.py", "src/imdb_lab/metrics.py",
        "data/manifest.json", "config.json", "sequence_config.json",
    ]
    return {
        "seeds": list(SEEDS), "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "recipes": {key: DEFAULTS | value for key, value in RECIPES.items()},
        "shared": {"vocabulary": 20000, "embedding": 128, "filters_per_width": 100,
                   "optimizer": "AdamW", "lr": 0.001, "clip_norm": 1.0, "precision": "FP32"},
        "selection": "Menor entropia cruzada bruta na validação, entre as épocas 1 a 8.",
        "evaluation": "Calibração da temperatura após congelar todos os checkpoints. Teste oficial e pares exploratórios.",
        "reference": REFERENCE, "contrast_revision": CONTRAST_REV, "contrast_sha256": CONTRAST_FILES,
        "code_sha256": {path: file_sha256(ROOT / path) for path in sources},
    }


def load_splits(folder, roles):
    manifest = read_json(ROOT / "data/manifest.json")["splits"]
    frames = {}
    for role in roles:
        path = folder / f"{role}.parquet"
        if file_sha256(path) != manifest[role]["sha256"]:
            raise ValueError(f"Hash inesperado da divisão {role}.")
        frames[role] = pd.read_parquet(path, columns=["id", "clean_text", "label", "text_sha256"])
        assert frames[role].id.is_unique and len(frames[role]) == manifest[role]["n"]
    return frames


def encode_frame(frame, vocab):
    mapping = vocab.token_to_id
    return [[mapping.get(token, 1) for token in tokenize(text)] or [1] for text in frame.clean_text]


class ReviewBatches:
    def __init__(self, encoded, labels, cap, min_width, device):
        self.device = device
        self.labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        clipped = [ids[:cap] for ids in encoded]
        self.lengths = np.fromiter(map(len, clipped), dtype=np.int64, count=len(clipped))
        self.width = max(min_width, int(self.lengths.max()))
        self.min_width = min_width
        # One staging matrix, but each convolution only sees the batch's own width.
        tokens = np.zeros((len(clipped), self.width), dtype=np.int32)
        for row, ids in zip(tokens, clipped, strict=True):
            row[:len(ids)] = ids
        self.tokens = torch.from_numpy(tokens).to(device)
        self.length_tensor = torch.from_numpy(self.lengths).to(device)

    def batch(self, indices):
        width = max(self.min_width, int(self.lengths[indices].max()))
        x = self.tokens[:, :width][indices]
        return (x, self.length_tensor[indices]), self.labels[indices]

    def inference_order(self):
        return np.argsort(self.lengths, kind="stable")


@torch.inference_mode()
def margins(model, data):
    model.eval()
    batches = sequences.batches(data.inference_order(), BATCH_SIZE)
    z = torch.empty(len(data.lengths), device=data.device)
    for indices in batches:
        inputs, _ = data.batch(indices)
        logits = model(*inputs)
        z[indices] = logits[:, 1] - logits[:, 0]
    values = z.cpu().numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError("Margens não finitas.")
    return values


def raw_scores(logits, labels):
    delta = logits.astype(np.float64)
    return {"ce": float(np.logaddexp(0, (1 - 2 * labels) * delta).mean()),
            "accuracy": float(((delta >= 0) == labels).mean())}


def train_one(model, data, labels, spec, seed, target):
    torch.manual_seed(seed + 40000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=spec["weight_decay"])
    orders = sequences.orders(len(labels["train"]), EPOCHS, seed)
    train, validation = data["train"], data["validation"]
    history, best_loss, best_state = [], float("inf"), None
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    for epoch, order in enumerate(orders, 1):
        model.train()
        sequences.sync(target)
        start = perf_counter()
        for indices in sequences.batches(order, BATCH_SIZE):
            inputs, targets = train.batch(indices)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(*inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        sequences.sync(target)
        elapsed = perf_counter() - start
        values = raw_scores(margins(model, validation), labels["validation"])
        history.append({"epoch": epoch, "train_seconds": elapsed, "validation": values})
        if values["ce"] < best_loss:
            best_loss, best_state = values["ce"], sequences.snapshot(model)
        print(epoch, f"val CE {values['ce']:.5f}", f"val acurácia {values['accuracy']:.5f}",
              f"treino {elapsed:.1f}s", flush=True)
    model.load_state_dict(best_state)
    return best_state, {
        "history": history, "best_epoch": min(history, key=lambda row: row["validation"]["ce"])["epoch"],
        "validation": raw_scores(margins(model, validation), labels["validation"]),
        "train": raw_scores(margins(model, train), labels["train"]),
        "batch_order_sha256": sequences.array_hash(np.concatenate(orders)),
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else None,
    }


def train(folder, output, target):
    protocol = plan()
    frames = load_splits(folder, ("train", "validation"))
    vocab = Vocabulary.fit(frames["train"].clean_text, 20000)
    original = read_json(ROOT / "seminario/dados/recorte_teste/resumo.json")
    output.mkdir(parents=True, exist_ok=False)
    weights = output / "weights"
    weights.mkdir()
    save_json(output / "protocol.json", protocol)
    vocab_file = output / "vocabulary.json"
    save_json(vocab_file, vocab.token_to_id)
    assert file_sha256(vocab_file) == original["artefatos_sha256"]["cnn/vocabulary.json"]
    encoded = {role: encode_frame(frame, vocab) for role, frame in frames.items()}
    labels = {role: frame.label.to_numpy(dtype=np.int64) for role, frame in frames.items()}
    records, initial, orders = [], {}, {}
    for name, spec in protocol["recipes"].items():
        data = {role: ReviewBatches(rows, labels[role], spec["cap"], max(spec["kernels"]), target)
                for role, rows in encoded.items()}
        for seed in SEEDS:
            print("INÍCIO", name, seed, flush=True)
            model = model_for(len(vocab.token_to_id), spec, seed, target)
            backbone = sequences.state_hash(nn.ModuleList([model.embedding, model.convs]))
            key = (seed, tuple(spec["kernels"]))
            assert initial.setdefault(key, backbone) == backbone
            state, record = train_one(model, data, labels, spec, seed, target)
            assert orders.setdefault(seed, record["batch_order_sha256"]) == record["batch_order_sha256"]
            run = f"{name}__s{seed}"
            checkpoint = weights / f"{run}.pt"
            torch.save(state, checkpoint)
            record.update(run=run, recipe=name, seed=seed, initial_backbone_sha256=backbone,
                          checkpoint_sha256=file_sha256(checkpoint))
            save_json(output / "runs" / f"{run}.json", record)
            records.append(record)
            print("CONCLUÍDO", run, "época", record["best_epoch"], flush=True)
        del data
    save_json(output / "selection.json", {
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "vocabulary_sha256": file_sha256(vocab_file),
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(target) if target.type == "cuda" else target.type,
        "records": records,
    })


def coverage(encoded, cap):
    original = np.fromiter(map(len, encoded), dtype=np.int64, count=len(encoded))
    retained = original if cap is None else np.minimum(original, cap)
    return {"n": len(encoded), "reviews_cut": int((retained < original).sum()),
            "tokens": int(original.sum()), "tokens_retained": int(retained.sum()),
            "max_tokens": int(original.max())}


def load_pairs(contrast_dir, seen_frames, vocab, caps):
    def encode(text):
        return vocab.encode(text, None)

    seen = set()
    for frame in seen_frames:
        seen.update(map(tuple, encode_frame(frame, vocab)))
    pairs, row_ids, _ = contrast_data(contrast_dir, seen, encode)
    all_tokens = [encode(text) for text in pairs.text]
    retain = np.ones(len(row_ids), dtype=bool)
    for cap in caps:
        seen_prefixes = {ids[:cap] for ids in seen}
        visible = [tuple(ids[:cap]) not in seen_prefixes for ids in all_tokens]
        retain &= np.asarray(visible).reshape(-1, 2).all(axis=1)
    pairs = pairs.iloc[np.repeat(retain, 2)].reset_index(drop=True)
    synthetics = synthetic_data(encode)
    assert all(tuple(encode(text)) not in seen for text in synthetics.text)
    return pairs, np.asarray(row_ids)[retain], synthetics


def subset_scores(labels, logits, temperature):
    values = evaluate(labels, expit(logits.astype(np.float64) / temperature))
    return {key: values[key] for key in ("n", "accuracy", "f1_macro", "log_loss")}


def pair_summary(logits, labels):
    return pair_results(np.column_stack((np.zeros_like(logits), logits)), labels)["fracao_pares"]


def audit(folder, contrast_dir, output, target):
    protocol_file = output / "protocol.json"
    selection_file = output / "selection.json"
    vocab_file = output / "vocabulary.json"
    protocol, selection = read_json(protocol_file), read_json(selection_file)
    assert protocol == plan() and file_sha256(protocol_file) == selection["protocol_sha256"]
    assert file_sha256(vocab_file) == selection["vocabulary_sha256"]
    records = selection["records"]
    assert [(row["recipe"], row["seed"]) for row in records] == [
        (name, seed) for name in protocol["recipes"] for seed in SEEDS
    ]
    for row in records:
        assert file_sha256(output / "weights" / f"{row['run']}.pt") == row["checkpoint_sha256"]
    frames = load_splits(folder, ("train", "validation", "calibration", "test"))
    vocab = Vocabulary(read_json(vocab_file))
    caps = sorted({spec["cap"] for spec in protocol["recipes"].values() if spec["cap"] is not None})
    contrast, row_ids, synthetic = load_pairs(contrast_dir, (frames[role] for role in
                                              ("train", "validation", "calibration")), vocab, caps)
    frames["contrast"] = contrast.rename(columns={"text": "clean_text"})
    frames["synthetic"] = synthetic.rename(columns={"text": "clean_text"})
    encoded = {role: encode_frame(frame, vocab) for role, frame in frames.items()}
    labels = {role: frame.label.to_numpy(dtype=np.int64) for role, frame in frames.items()}
    lengths = np.fromiter(map(len, encoded["test"]), dtype=np.int64)
    long_reviews = lengths > DEFAULTS["cap"]
    last = synthetic.family.str.endswith("_antes").to_numpy()
    arrays = {"test_ids": frames["test"].id.to_numpy(dtype=str),
              "test_word_tokens": lengths, "contrast_tsv_rows": row_ids,
              "synthetic_family": synthetic.family.to_numpy(dtype=str)}
    arrays.update({f"{role}_labels": labels[role] for role in ("calibration", "test", "contrast", "synthetic")})
    results = []
    for row in records:
        name, seed, run = row["recipe"], row["seed"], row["run"]
        spec = protocol["recipes"][name]
        model = model_for(len(vocab.token_to_id), spec, seed, target)
        model.load_state_dict(torch.load(output / "weights" / f"{run}.pt", map_location="cpu", weights_only=True))
        for role in ("calibration", "test", "contrast", "synthetic"):
            batches = ReviewBatches(encoded[role], labels[role], spec["cap"], max(spec["kernels"]), target)
            arrays[f"{role}_{run}"] = margins(model, batches)
        test = arrays[f"test_{run}"]
        temperature = fit_temperature(arrays[f"calibration_{run}"], labels["calibration"])
        pair_logits = arrays[f"synthetic_{run}"]
        result = {"recipe": name, "seed": seed, "temperature": temperature,
                  "validation": row["validation"], "train": row["train"],
                  "test": subset_scores(labels["test"], test, temperature),
                  "test_long": subset_scores(labels["test"][long_reviews], test[long_reviews], temperature),
                  "contrast": pair_summary(arrays[f"contrast_{run}"], labels["contrast"]),
                  "verdict_last": pair_summary(pair_logits[last], labels["synthetic"][last]),
                  "verdict_first": pair_summary(pair_logits[~last], labels["synthetic"][~last]),
                  "seconds": sum(epoch["train_seconds"] for epoch in row["history"])}
        results.append(result)
        print("TESTE", name, seed, result["test"], flush=True)
    predictions = output / "predictions.npz"
    np.savez_compressed(predictions, **arrays)
    means = {}
    for name in protocol["recipes"]:
        rows = [row for row in results if row["recipe"] == name]
        means[name] = {group: {metric: mean(row[group][metric] for row in rows)
                               for metric in rows[0][group] if metric != "n"}
                       for group in ("train", "validation", "test", "test_long")}
        means[name].update({key: mean(row[key] for row in rows)
                            for key in ("contrast", "verdict_last", "verdict_first", "seconds")})
    save_json(output / "summary.json", {
        "selection_sha256": file_sha256(selection_file), "predictions_sha256": file_sha256(predictions),
        "coverage": {str(cap) if cap else "integral": {role: coverage(encoded[role], cap)
                     for role in ("train", "validation", "test")} for cap in (*caps, None)},
        "contrast_pairs": len(row_ids), "synthetic_pairs_per_verdict_position": int(last.sum() // 2),
        "mean": means, "runs": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "audit"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contrast-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "audit" and args.contrast_dir is None:
        parser.error("audit requer --contrast-dir.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    target = sequences.configure(args.device)
    if args.stage == "train":
        train(args.data, args.output, target)
    else:
        audit(args.data, args.contrast_dir, args.output, target)


if __name__ == "__main__":
    main()
