"""Amplia filtros e embeddings da TextCNN com resenhas inteiras."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import nn

from imdb_lab import sequences
from imdb_lab.common import ROOT, file_sha256, read_json, save_json
from imdb_lab.metrics import fit_temperature
from imdb_lab.models import TextCNN, Vocabulary
from treinar_cnn_comprimento import load_splits, raw_scores, subset_scores
from treinar_cnn_maior import CHECKPOINTS, SEEDS, plan as previous_plan, predict, prepare_batches, train_one


RECIPES = {
    "textcnn": {"embedding_dim": 128, "filters": 100},
    "mais_filtros": {"embedding_dim": 128, "filters": 1024},
    "mais_filtros_embedding": {"embedding_dim": 384, "filters": 1024},
}
LEARNING_RATE = 0.001
KERNELS = (3, 4, 5)


def plan():
    recipe = previous_plan()
    del recipe["residual"], recipe["textcnn"]
    recipe["source_sha256"]["seminario/treinar_textcnn_larga.py"] = file_sha256(Path(__file__))
    recipe["recipes"] = {name: {**dimensions, "lr": LEARNING_RATE} for name, dimensions in RECIPES.items()}
    recipe["kernels"] = list(KERNELS)
    recipe["pooling"] = "Máximo global por filtro com máscara."
    recipe["initialization"] = "Embedding normal (std 0,1), convoluções He normal, camada linear Xavier uniforme."
    return recipe


def model_for(name, vocabulary_size, seed):
    torch.manual_seed(seed)
    model = TextCNN(vocabulary_size, kernels=KERNELS, dropout=0.3, **RECIPES[name])
    nn.init.normal_(model.embedding.weight, std=0.1)
    with torch.no_grad():
        model.embedding.weight[0].zero_()
    for convolution in model.convs:
        nn.init.kaiming_normal_(convolution.weight, nonlinearity="relu")
        nn.init.zeros_(convolution.bias)
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
    return model


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
        batch_orders = {}
        for name in RECIPES:
            run = f"{name}_s{seed}"
            model = model_for(name, len(vocab.token_to_id), seed)
            initial_hash = sequences.state_hash(model)
            model.to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            record = train_one(model, data, labels, seed, LEARNING_RATE, output, run)
            records.append({"model": name, "seed": seed, "initial_sha256": initial_hash, **record})
            for row in record["history"]:
                digest = row["batches_sha256"]
                assert batch_orders.setdefault(row["epoch"], digest) == digest
            print(f"FIM {run}: {record['stop']}, melhor época {record['selected']['melhor']['epoch']}", flush=True)
            del model
    save_json(output / "selection.json", {
        "protocol_sha256": file_sha256(protocol), "vocabulary_sha256": file_sha256(vocab_file),
        "runtime": {"torch": str(torch.__version__), "device": device.type,
                    "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "records": records,
    })


def audit(folder, output, device):
    selection_path = output / "selection.json"
    selection = read_json(selection_path)
    assert read_json(output / "protocol.json") == plan()
    assert file_sha256(output / "protocol.json") == selection["protocol_sha256"]
    vocab_file = output / "vocabulary.json"
    assert file_sha256(vocab_file) == selection["vocabulary_sha256"]
    records = selection["records"]
    assert [(row["model"], row["seed"]) for row in records] == [(name, seed) for seed in SEEDS for name in RECIPES]
    for row in records:
        for budget in CHECKPOINTS:
            path = output / "weights" / f"{row['run']}_{budget}.pt"
            assert file_sha256(path) == row["selected"][budget]["sha256"]

    vocab = Vocabulary(read_json(vocab_file))
    frames = load_splits(folder, ("calibration", "test"))
    data, labels = prepare_batches(frames, vocab, device)
    long = data["test"].lengths > 400
    arrays = {"test_ids": frames["test"].id.to_numpy(dtype=str), "test_labels": labels["test"],
              "test_tokens": data["test"].lengths, "calibration_labels": labels["calibration"]}
    results = []
    for row in records:
        model = model_for(row["model"], len(vocab.token_to_id), row["seed"]).to(device)
        evaluated = {}
        for budget in CHECKPOINTS:
            run = f"{row['run']}_{budget}"
            checkpoint = row["selected"][budget]
            digest = checkpoint["sha256"]
            if digest not in evaluated:
                path = output / "weights" / f"{run}.pt"
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                zc, zt = (predict(model, data[role]) for role in ("calibration", "test"))
                temperature = fit_temperature(zc, labels["calibration"])
                evaluated[digest] = zc, zt, {
                    "temperature": temperature, "test_raw_ce": raw_scores(zt, labels["test"])["ce"],
                    "test": subset_scores(labels["test"], zt, temperature),
                    "test_long": subset_scores(labels["test"][long], zt[long], temperature),
                }
            zc, zt, scores = evaluated[digest]
            arrays[f"calibration_{run}"], arrays[f"test_{run}"] = zc, zt
            result = {"model": row["model"], "seed": row["seed"], "budget": budget,
                      "epoch": checkpoint["epoch"], **scores}
            results.append(result)
            print(f"TESTE {run}: {result['test']}", flush=True)
        del model
    predictions = output / "predictions.npz"
    np.savez_compressed(predictions, **arrays)
    groups = {}
    for name in RECIPES:
        for budget in CHECKPOINTS:
            rows = [row for row in results if (row["model"], row["budget"]) == (name, budget)]
            groups[f"{name}_{budget}"] = {
                split: {key: mean(row[split][key] for row in rows) for key in rows[0][split]}
                for split in ("test", "test_long")
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
