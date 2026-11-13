"""Controla a inicialização dos embeddings de uma TextCNN com GloVe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

import numpy as np
import torch

from imdb_lab import sequences
from imdb_lab.common import ROOT, file_sha256, save_json
from imdb_lab.metrics import fit_temperature
from imdb_lab.models import TextCNN, Vocabulary
from treinar_cnn_comprimento import (
    DEFAULTS, EPOCHS, SEEDS, ReviewBatches, encode_frame, load_splits,
    margins, subset_scores, train_one,
)


GLOVE_SHA256 = "6471382cdd837544bf3ac72497a38715e845897d265b2b424b4761832009c837"
GLOVE_FILE = "glove.6B.100d.txt"
RECIPES = ("aleatorio", "glove_embaralhado", "glove")


def load_glove(archive, vocab):
    if file_sha256(archive) != GLOVE_SHA256:
        raise ValueError("Use o arquivo GloVe 6B publicado por Stanford.")
    mapping = vocab.token_to_id
    vectors = {}
    with ZipFile(archive) as bundle, bundle.open(GLOVE_FILE) as handle:
        for line in handle:
            word, _, values = line.partition(b" ")
            token = word.decode("utf-8")
            if token in mapping and mapping[token] > 1:
                vector = np.fromstring(values.decode("ascii"), sep=" ", dtype=np.float32)
                assert vector.shape == (100,) and np.isfinite(vector).all()
                vectors[mapping[token]] = vector
    ids = np.array(sorted(vectors))
    matrix = torch.tensor(np.stack([vectors[i] for i in ids]))
    assert 0 < len(ids) <= len(mapping) - 2
    return ids, matrix


def model_for(name, vocabulary_size, ids, vectors, seed, target):
    torch.manual_seed(seed)
    model = TextCNN(vocabulary_size, embedding_dim=100)
    with torch.no_grad():
        table = model.embedding.weight
        if name == "glove":
            table[ids] = vectors
        elif name == "glove_embaralhado":
            rng = np.random.default_rng(seed)
            table[ids] = vectors[rng.permutation(len(ids))]
    return model.to(target)


def run(folder, archive, output, target):
    frames = load_splits(folder, ("train", "validation"))
    vocab = Vocabulary.fit(frames["train"].clean_text, 20000)
    ids, vectors = load_glove(archive, vocab)
    output.mkdir(parents=True, exist_ok=False)
    weights = output / "weights"
    weights.mkdir()
    protocol = output / "protocol.json"
    save_json(protocol, {
        "recipes": RECIPES, "seeds": SEEDS, "epochs": EPOCHS,
        "shared": DEFAULTS | {"cap": None, "embedding_dim": 100, "optimizer": "AdamW",
                              "lr": 0.001, "trainable_embeddings": True},
        "glove_archive_sha256": GLOVE_SHA256, "glove_file": GLOVE_FILE,
        "glove_rows": len(ids), "vocabulary_entries": len(vocab.token_to_id),
        "selection": "Menor entropia cruzada bruta na validação. Checkpoints congelados antes da calibração e teste.",
        "sources_sha256": {p: file_sha256(ROOT / p) for p in (
            "seminario/treinar_cnn_embeddings.py", "seminario/treinar_cnn_comprimento.py",
            "src/imdb_lab/models.py", "src/imdb_lab/sequences.py", "data/manifest.json",
        )},
        "runtime": {"torch": str(torch.__version__), "target": target.type},
    })
    labels = {role: frame.label.to_numpy(dtype=np.int64) for role, frame in frames.items()}
    data = {role: ReviewBatches(encode_frame(frame, vocab), labels[role], None, 5, target)
            for role, frame in frames.items()}
    records = []
    for seed in SEEDS:
        other_weights = None
        for name in RECIPES:
            run_id = f"{name}__s{seed}"
            print("INÍCIO", run_id, flush=True)
            model = model_for(name, len(vocab.token_to_id), ids, vectors, seed, target)
            initial = {key: sequences.state_hash(getattr(model, key)) for key in ("convs", "classifier")}
            if other_weights is None:
                other_weights = initial
            assert initial == other_weights
            state, record = train_one(model, data, labels, DEFAULTS, seed, target)
            checkpoint = weights / f"{run_id}.pt"
            torch.save(state, checkpoint)
            record.update(recipe=name, seed=seed, run=run_id, initial=initial,
                          checkpoint_sha256=file_sha256(checkpoint))
            records.append(record)
    selection = output / "selection.json"
    save_json(selection, {"records": records, "protocol_sha256": file_sha256(protocol)})

    frames = load_splits(folder, ("calibration", "test"))
    labels = {role: frame.label.to_numpy(dtype=np.int64) for role, frame in frames.items()}
    data = {role: ReviewBatches(encode_frame(frame, vocab), labels[role], None, 5, target)
            for role, frame in frames.items()}
    arrays = {role + "_labels": values for role, values in labels.items()}
    arrays["test_ids"] = frames["test"].id.to_numpy(dtype=str)
    arrays["test_word_tokens"] = data["test"].lengths
    long = data["test"].lengths > 400
    results = []
    for record in records:
        run_id = record["run"]
        checkpoint = weights / f"{run_id}.pt"
        assert file_sha256(checkpoint) == record["checkpoint_sha256"]
        model = TextCNN(len(vocab.token_to_id), embedding_dim=100).to(target)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        calibration, test = (margins(model, data[role]) for role in ("calibration", "test"))
        tau = fit_temperature(calibration, labels["calibration"])
        arrays[f"calibration_{run_id}"] = calibration
        arrays[f"test_{run_id}"] = test
        result = {"recipe": record["recipe"], "seed": record["seed"],
                  "validation": record["validation"], "train": record["train"], "temperature": tau,
                  "test": subset_scores(labels["test"], test, tau),
                  "test_long": subset_scores(labels["test"][long], test[long], tau)}
        results.append(result)
        print("TESTE", run_id, result["test"], flush=True)
    means = {}
    for name in RECIPES:
        rows = [row for row in results if row["recipe"] == name]
        means[name] = {group: {key: mean(row[group][key] for row in rows)
                               for key in rows[0][group] if key != "n"}
                       for group in ("train", "validation", "test", "test_long")}
    predictions = output / "predictions.npz"
    np.savez_compressed(predictions, **arrays)
    save_json(output / "summary.json", {"selection_sha256": file_sha256(selection),
              "predictions_sha256": file_sha256(predictions), "mean": means, "runs": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(args.data, args.glove, args.output, sequences.configure(args.device))


if __name__ == "__main__":
    main()
