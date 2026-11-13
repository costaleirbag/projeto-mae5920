"""Testa se exemplos contrastivos no treino ensinam a identificar o veredito."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch

from imdb_lab import sequences
from imdb_lab.common import ROOT, file_sha256, save_json
from medir_posicao import CONTRAST_FILES, NEGATIVE, POSITIVE
from treinar_cnn_comprimento import (
    SEEDS, EPOCHS, ReviewBatches, encode_frame, margins, model_for, pair_summary, raw_scores, train_one,
)
from treinar_cnn_pares import RECIPES, prepare_data


POSITIVE_TRAIN = "engaging joyful splendid beautiful hilarious riveting gripping touching".split()
NEGATIVE_TRAIN = "atrocious tedious lousy dismal annoying unbearable unwatchable obnoxious".split()
TRAIN_TEMPLATES = (
    "I expected a {setup} film. Having seen it, my verdict is {verdict}.",
    "My final verdict is {verdict}. Before watching, I expected a {setup} film.",
    "I went in thinking it would be {setup}, but the movie turned out to be {verdict}.",
    "The movie turned out to be {verdict}, although I went in thinking it would be {setup}.",
    "The film looked {setup} from the trailer. In fact, it was {verdict}.",
    "In fact, the film was {verdict}, despite looking {setup} in the trailer.",
)


def training_pairs(vocab):
    assert set(POSITIVE_TRAIN + NEGATIVE_TRAIN).isdisjoint(POSITIVE + NEGATIVE)
    texts, labels = [], []
    for template in TRAIN_TEMPLATES:
        for positive in POSITIVE_TRAIN:
            for negative in NEGATIVE_TRAIN:
                a = template.format(setup=negative, verdict=positive)
                b = template.format(setup=positive, verdict=negative)
                first, second = vocab.encode(a, None), vocab.encode(b, None)
                assert first != second and Counter(first) == Counter(second)
                texts.extend((a, b))
                labels.extend((1, 0))
    return pd.DataFrame({"clean_text": texts, "label": labels})


def describe_sources():
    paths = [
        "seminario/treinar_cnn_ordem_supervisionada.py", "seminario/treinar_cnn_pares.py",
        "seminario/treinar_cnn_comprimento.py", "seminario/treinar_pooling.py", "seminario/medir_posicao.py",
        "src/imdb_lab/sequences.py", "src/imdb_lab/models.py", "results/bpe/data_manifest.json",
        "results/sequences/vocabulary.json",
    ]
    return {p: file_sha256(ROOT / p) for p in paths}


def run(source, folder, output, target):
    roles, vocab, line_ids = prepare_data(source, folder)
    train = roles["train"]
    extra = training_pairs(vocab)
    original = {role: encode_frame(frame, vocab) for role, frame in roles.items()}
    synthetic = encode_frame(extra, vocab)
    seen = set(map(tuple, synthetic))
    assert len(seen) == len(extra)
    assert all(seen.isdisjoint(map(tuple, encoded)) for encoded in original.values())
    labels = {role: frame.label.to_numpy(dtype=np.int64) for role, frame in roles.items()}
    rng = np.random.default_rng(20261113)
    duplicate = np.concatenate([
        rng.choice(np.flatnonzero(labels["train"] == label), len(extra) // 2, replace=False)
        for label in (0, 1)
    ])
    conditions = {
        "repeticao": pd.concat([train, train.iloc[duplicate]], ignore_index=True),
        "contrastes": pd.concat([train, extra], ignore_index=True),
    }
    assert all(len(frame) == len(train) + len(extra) for frame in conditions.values())
    assert conditions["repeticao"].label.value_counts().to_dict() == conditions["contrastes"].label.value_counts().to_dict()
    specs = {name: RECIPES[name] for name in ("max_integral", "dilatada_integral")}

    output.mkdir(parents=True, exist_ok=False)
    protocol = output / "protocol.json"
    weights = output / "weights"
    weights.mkdir()
    save_json(protocol, {
        "architectures": specs, "conditions": list(conditions), "seeds": SEEDS, "epochs": EPOCHS,
        "original_train": len(train), "extra_train": len(extra), "control_sampling_seed": 20261113,
        "training_templates": TRAIN_TEMPLATES, "positive_train": POSITIVE_TRAIN, "negative_train": NEGATIVE_TRAIN,
        "selection": "Menor CE em 1.536 resenhas de validação originais. Pares não participam da escolha.",
        "scope": "Auditoria interna, pares editados e novos moldes de veredito em relação aos usados no treino. Sem teste oficial.",
        "contrast_sha256": CONTRAST_FILES, "sources_sha256": describe_sources(), "torch": str(torch.__version__),
    })
    records = []
    for name, spec in specs.items():
        validation = ReviewBatches(original["validation"], labels["validation"], None, 5, target)
        initial = {}
        for condition, frame in conditions.items():
            target_labels = labels | {"train": frame.label.to_numpy(dtype=np.int64)}
            data = {"validation": validation, "train": ReviewBatches(
                encode_frame(frame, vocab), target_labels["train"], None, 5, target)}
            for seed in SEEDS:
                run_id = f"{name}__{condition}__s{seed}"
                print("INÍCIO", run_id, flush=True)
                model = model_for(len(vocab.token_to_id), spec, seed, target)
                signature = sequences.state_hash(model)
                assert initial.setdefault(seed, signature) == signature
                state, record = train_one(model, data, target_labels, spec, seed, target)
                checkpoint = weights / f"{run_id}.pt"
                torch.save(state, checkpoint)
                record.update(architecture=name, condition=condition, seed=seed, run=run_id,
                              initial_sha256=signature, checkpoint_sha256=file_sha256(checkpoint))
                records.append(record)
    selection = output / "selection.json"
    save_json(selection, {"records": records, "protocol_sha256": file_sha256(protocol)})

    last = roles["synthetic"].family.str.endswith("_antes").to_numpy()
    evaluations = ("audit", "contrast", "synthetic")
    audit = {role: ReviewBatches(original[role], labels[role], None, 5, target) for role in evaluations}
    arrays = {role + "_labels": labels[role] for role in evaluations}
    arrays.update(contrast_tsv_rows=np.asarray(line_ids), audit_ids=roles["audit"].index.to_numpy(dtype=str),
                  synthetic_family=roles["synthetic"].family.to_numpy(dtype=str))
    results = []
    for row in records:
        name, condition, seed, run_id = (row[k] for k in ("architecture", "condition", "seed", "run"))
        checkpoint = weights / f"{run_id}.pt"
        assert file_sha256(checkpoint) == row["checkpoint_sha256"]
        model = model_for(len(vocab.token_to_id), specs[name], seed, target)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        scores = {role: margins(model, values) for role, values in audit.items()}
        arrays.update({f"{role}_{run_id}": value for role, value in scores.items()})
        pairs = scores["synthetic"]
        result = {
            "architecture": name, "condition": condition, "seed": seed,
            "audit_accuracy": raw_scores(scores["audit"], labels["audit"])["accuracy"],
            "human_pair_accuracy": pair_summary(scores["contrast"], labels["contrast"]),
            "verdict_last": pair_summary(pairs[last], labels["synthetic"][last]),
            "verdict_first": pair_summary(pairs[~last], labels["synthetic"][~last]),
        }
        results.append(result)
        print("AVALIAÇÃO", result, flush=True)
    means = {}
    for name in specs:
        for condition in conditions:
            matching = [r for r in results if r["architecture"] == name and r["condition"] == condition]
            means[f"{name}__{condition}"] = {
                key: mean(row[key] for row in matching)
                for key in ("audit_accuracy", "human_pair_accuracy", "verdict_last", "verdict_first")
            }
    predictions = output / "predictions.npz"
    np.savez_compressed(predictions, **arrays)
    save_json(output / "summary.json", {"mean": means, "runs": results, "human_pairs": len(line_ids),
              "selection_sha256": file_sha256(selection), "predictions_sha256": file_sha256(predictions)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--contrast-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(args.data, args.contrast_dir, args.output, sequences.configure(args.device))


if __name__ == "__main__":
    main()
