"""Compara tamanho da CNN e duração do treino com resenhas inteiras."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from shutil import copyfile
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from scipy.special import expit
from torch import nn
from torch.nn import functional as F

from imdb_lab import sequences
from imdb_lab.common import ROOT, file_sha256, read_json, save_json
from imdb_lab.metrics import fit_temperature
from imdb_lab.models import TextCNN, Vocabulary
from treinar_cnn_comprimento import (
    BATCH_SIZE, ReviewBatches, encode_frame, load_splits, raw_scores, subset_scores,
)
from treinar_pooling import masked_max, masked_mean


SEEDS = (17, 29)
RECIPES = {
    "textcnn": {"architecture": "textcnn", "lr": 0.001},
    "residual": {"architecture": "residual", "lr": 0.001},
    "residual_lr_menor": {"architecture": "residual", "lr": 0.0003},
}
CHECKPOINTS = ("ate_5", "melhor")
MAX_EPOCHS = 30
RUN_SECONDS = 660
PATIENCE = 6
MIN_DELTA = 0.001


def token_mask(features, lengths):
    return torch.arange(features.size(-1), device=features.device)[None, None, :] < lengths[:, None, None]


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.first = nn.Conv1d(channels, channels, 3, padding=1)
        self.last = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x, keep):
        normalized = self.norm(x.transpose(1, 2)).transpose(1, 2).masked_fill(~keep, 0)
        local = self.first(normalized).relu().masked_fill(~keep, 0)
        return (x + 0.5 * self.last(local)).relu().masked_fill(~keep, 0)


class ResidualCNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim=256, channels=384, blocks=6, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.stem = nn.Conv1d(embedding_dim, channels, 5, padding=2)
        self.blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(blocks))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * channels, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 2),
        )

    def forward(self, input_ids, lengths):
        x = self.stem(self.embedding(input_ids).transpose(1, 2)).relu()
        keep = token_mask(x, lengths)
        x = x.masked_fill(~keep, 0)
        for block in self.blocks:
            # ReLU makes zeros beyond the valid end neutral in this local maximum.
            x = F.max_pool1d(x, 2, stride=2, ceil_mode=True)
            lengths = (lengths + 1) // 2
            keep = token_mask(x, lengths)
            x = block(x, keep)
        return self.classifier(torch.cat((masked_max(x, keep), masked_mean(x, keep)), dim=1))


def model_for(name, vocabulary_size, seed):
    torch.manual_seed(seed)
    model = (TextCNN(vocabulary_size, dropout=0.3) if name == "textcnn"
             else ResidualCNN(vocabulary_size))
    nn.init.normal_(model.embedding.weight, std=0.1)
    with torch.no_grad():
        model.embedding.weight[0].zero_()
    for layer in model.modules():
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model


def training_batches(lengths, seed, epoch):
    rng = np.random.default_rng(seed + 1009 * epoch)
    batches = []
    for part in sequences.batches(rng.permutation(len(lengths)), 16 * BATCH_SIZE):
        similar_lengths = part[np.argsort(lengths[part], kind="stable")]
        batches.extend(sequences.batches(similar_lengths, BATCH_SIZE))
    return [batches[i] for i in rng.permutation(len(batches))]


@torch.inference_mode()
def predict(model, data):
    model.eval()
    output = torch.empty(len(data.lengths), device=data.device)
    with sequences.autocast(data.device, True):
        for indices in sequences.batches(data.inference_order(), BATCH_SIZE):
            inputs, _ = data.batch(indices)
            logits = model(*inputs).float()
            output[indices] = logits[:, 1] - logits[:, 0]
    values = output.cpu().numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError("Margens não finitas na avaliação.")
    return values


def plan():
    paths = (
        "data/manifest.json", "sequence_config.json", "seminario/treinar_cnn_maior.py",
        "seminario/treinar_cnn_comprimento.py", "seminario/treinar_pooling.py",
        "src/imdb_lab/models.py", "src/imdb_lab/sequences.py", "src/imdb_lab/metrics.py",
    )
    return {
        "source_sha256": {name: file_sha256(ROOT / name) for name in paths},
        "seeds": list(SEEDS), "recipes": RECIPES, "vocabulary_size": 20000, "token_cap": None,
        "textcnn": {"embedding": 128, "kernels": [3, 4, 5], "filters_per_width": 100},
        "residual": {"embedding": 256, "channels": 384, "blocks": 6, "stem_kernel": 5,
                     "local_max_pool_stride": 2, "convs_per_block": 2, "block_kernel": 3,
                     "residual_scale": 0.5, "document_pool": ["max", "mean"], "head_hidden": 256},
        "batch_size": BATCH_SIZE, "batching": "Grupos de 16 lotes ordenados por comprimento e sorteados por época.",
        "optimizer": "AdamW", "weight_decay": 0.01, "clip_norm": 1,
        "dropout": 0.3, "max_epochs": MAX_EPOCHS, "max_seconds_per_run": RUN_SECONDS,
        "plateau": {"factor": 0.5, "patience": 2, "min_lr": 0.00005},
        "early_stop": {"patience": PATIENCE, "min_delta": MIN_DELTA, "min_epochs": 8},
        "selection": "Menor CE bruta de validação entre as primeiras 5 épocas e em todo o treino.",
        "precision": "BF16 autocast em CUDA, pesos e otimizador FP32, TF32 desligado.",
        "audit": "Calibração e teste lidos após congelar todos os checkpoints. Teste oficial já consultado.",
    }


def train_one(model, data, labels, seed, initial_lr, output, run):
    torch.manual_seed(seed + 40000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, threshold=MIN_DELTA, threshold_mode="abs", min_lr=0.00005,
    )
    train, validation = data["train"], data["validation"]
    device = train.device
    best_loss, best_progress, last_progress = float("inf"), float("inf"), 0
    selected, history = {}, []
    started = perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        statistics = torch.zeros(2, device=device)
        batches = training_batches(train.lengths, seed, epoch)
        sequences.sync(device)
        step_start = perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        for indices in batches:
            inputs, targets = train.batch(indices)
            optimizer.zero_grad(set_to_none=True)
            with sequences.autocast(device, True):
                logits = model(*inputs).float()
                loss = F.cross_entropy(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            statistics[0] += loss.detach() * len(indices)
            statistics[1] += (logits.detach().argmax(dim=1) == targets).sum()
        sequences.sync(device)
        train_seconds = perf_counter() - step_start
        online_ce, online_accuracy = (statistics / len(train.lengths)).tolist()
        values = raw_scores(predict(model, validation), labels["validation"])
        row = {"epoch": epoch, "lr": lr, "train_online": {"ce": online_ce, "accuracy": online_accuracy},
               "validation": values, "train_seconds": train_seconds, "elapsed_seconds": perf_counter() - started,
               "batches_sha256": sequences.array_hash(np.concatenate(batches))}
        history.append(row)
        if values["ce"] < best_loss:
            best_loss = values["ce"]
            checkpoint = output / "weights" / f"{run}_melhor.pt"
            torch.save(sequences.snapshot(model), checkpoint)
            selected["melhor"] = {"epoch": epoch, "sha256": file_sha256(checkpoint)}
            if epoch <= 5:
                short = output / "weights" / f"{run}_ate_5.pt"
                copyfile(checkpoint, short)
                selected["ate_5"] = selected["melhor"].copy()
        if values["ce"] < best_progress - MIN_DELTA:
            best_progress, last_progress = values["ce"], epoch
        scheduler.step(values["ce"])
        save_json(output / "curves" / f"{run}.json", history)
        print(f"{run} época {epoch:02d}: treino CE {online_ce:.4f}, ac {online_accuracy:.3f}, "
              f"val CE {values['ce']:.4f}, ac {values['accuracy']:.3f}, "
              f"LR {lr:g}, {row['elapsed_seconds']:.0f}s acumulados", flush=True)
        if epoch >= 8 and epoch - last_progress >= PATIENCE:
            stop = "patience"
            break
        if epoch >= 5 and perf_counter() - started + row["train_seconds"] >= RUN_SECONDS:
            stop = "budget"
            break
    else:
        stop = "max_epochs"
    return {"run": run, "history": history, "selected": selected, "stop": stop,
            "seconds": perf_counter() - started, "parameters": sum(p.numel() for p in model.parameters()),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def prepare_batches(frames, vocab, device):
    labels = {role: frame.label.to_numpy(dtype=np.int64) for role, frame in frames.items()}
    data = {role: ReviewBatches(encode_frame(frame, vocab), labels[role], None, 5, device)
            for role, frame in frames.items()}
    return data, labels


def train(folder, output, device):
    output.mkdir(parents=True, exist_ok=False)
    (output / "weights").mkdir()
    protocol = output / "protocol.json"
    save_json(protocol, plan())
    frames = load_splits(folder, ("train", "validation"))
    vocab = Vocabulary.fit(frames["train"].clean_text, 20000)
    vocab_file = output / "vocabulary.json"
    save_json(vocab_file, vocab.token_to_id)
    data, labels = prepare_batches(frames, vocab, device)
    records = []
    for seed in SEEDS:
        starts = {}
        for name, spec in RECIPES.items():
            run = f"{name}_s{seed}"
            architecture = spec["architecture"]
            model = model_for(architecture, len(vocab.token_to_id), seed)
            initial_hash = sequences.state_hash(model)
            assert starts.setdefault(architecture, initial_hash) == initial_hash
            model.to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            record = train_one(model, data, labels, seed, spec["lr"], output, run)
            records.append({"model": name, "seed": seed, "initial_sha256": initial_hash, **record})
            print(f"FIM {run} {record['stop']} melhores épocas "
                  f"{[item['epoch'] for item in record['selected'].values()]}", flush=True)
            del model
    for seed in SEEDS:
        batch_orders = {}
        for row in (r for r in records if r["seed"] == seed):
            for epoch in row["history"]:
                digest = epoch["batches_sha256"]
                assert batch_orders.setdefault(epoch["epoch"], digest) == digest
    save_json(output / "selection.json", {
        "protocol_sha256": file_sha256(protocol), "vocabulary_sha256": file_sha256(vocab_file),
        "runtime": {"torch": str(torch.__version__), "device": device.type,
                    "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "records": records,
    })


def audit(folder, output, device):
    selection_path = output / "selection.json"
    selection = read_json(selection_path)
    vocab_file = output / "vocabulary.json"
    assert file_sha256(vocab_file) == selection["vocabulary_sha256"]
    assert file_sha256(output / "protocol.json") == selection["protocol_sha256"]
    assert read_json(output / "protocol.json") == plan()
    vocab = Vocabulary(read_json(vocab_file))
    expected = [(name, seed) for seed in SEEDS for name in RECIPES]
    assert [(row["model"], row["seed"]) for row in selection["records"]] == expected
    for row in selection["records"]:
        for budget in CHECKPOINTS:
            assert file_sha256(output / "weights" / f"{row['run']}_{budget}.pt") == row["selected"][budget]["sha256"]
    frames = load_splits(folder, ("train", "calibration", "test"))
    data, labels = prepare_batches(frames, vocab, device)
    long = data["test"].lengths > 400
    arrays = {"test_ids": frames["test"].id.to_numpy(dtype=str), "test_labels": labels["test"],
              "test_tokens": data["test"].lengths, "calibration_labels": labels["calibration"]}
    results = []
    for row in selection["records"]:
        architecture = RECIPES[row["model"]]["architecture"]
        model = model_for(architecture, len(vocab.token_to_id), row["seed"]).to(device)
        cached_scores = {}
        for budget in CHECKPOINTS:
            run = f"{row['run']}_{budget}"
            digest = row["selected"][budget]["sha256"]
            if digest not in cached_scores:
                checkpoint = output / "weights" / f"{run}.pt"
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                zc, zt = (predict(model, data[role]) for role in ("calibration", "test"))
                temperature = fit_temperature(zc, labels["calibration"])
                cached_scores[digest] = zc, zt, {
                    "temperature": temperature,
                    "train": raw_scores(predict(model, data["train"]), labels["train"]),
                    "test": subset_scores(labels["test"], zt, temperature),
                    "test_long": subset_scores(labels["test"][long], zt[long], temperature),
                    "test_raw_ce": raw_scores(zt, labels["test"])["ce"],
                }
            zc, zt, scores = cached_scores[digest]
            metrics = {"model": row["model"], "seed": row["seed"], "budget": budget,
                       "epoch": row["selected"][budget]["epoch"], **scores}
            arrays[f"calibration_{run}"], arrays[f"test_{run}"] = zc, zt
            results.append(metrics)
            print(f"TESTE {run}: {metrics['test']}", flush=True)
        del model
    predictions = output / "predictions.npz"
    np.savez_compressed(predictions, **arrays)
    groups = {}
    for name in RECIPES:
        for budget in CHECKPOINTS:
            rows = [row for row in results if (row["model"], row["budget"]) == (name, budget)]
            groups[f"{name}_{budget}"] = {
                split: {key: mean(row[split][key] for row in rows) for key in rows[0][split]}
                for split in ("train", "test", "test_long")
            }
    save_json(output / "summary.json", {
        "selection_sha256": file_sha256(selection_path), "predictions_sha256": file_sha256(predictions),
        "groups": groups, "runs": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "audit"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = sequences.configure(args.device)
    (train if args.stage == "train" else audit)(args.data, args.output, device)


if __name__ == "__main__":
    main()
