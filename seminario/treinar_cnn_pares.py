"""Repete o ensaio de ordem reservando os pares humanos desde o início."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from imdb_lab import sequences
from imdb_lab.common import ROOT, file_sha256, read_json, save_json
from imdb_lab.models import Vocabulary
from medir_posicao import CONTRAST_FILES, contrast_data, synthetic_data
from treinar_cnn_comprimento import (
    DEFAULTS, EPOCHS, SEEDS, ReviewBatches, encode_frame, load_splits,
    margins, model_for, pair_summary, raw_scores, train_one,
)


RECIPES = {
    "max_128_palavras": DEFAULTS,
    "max_integral": DEFAULTS | {"cap": None},
    "max_media_integral": DEFAULTS | {"cap": None, "pooling": "max_media"},
    "max_metades_integral": DEFAULTS | {"cap": None, "pooling": "max_metades_posicao"},
    "dilatada_integral": DEFAULTS | {"cap": None, "pooling": "max_media", "dilations": [1, 3]},
}


def crop(text, recipe):
    return sequences.excerpt(text, 128) if recipe == "max_128_palavras" else text


def prepare_data(source, folder):
    reference = read_json(ROOT / "results/pooling_order/protocol.json")
    manifest = read_json(ROOT / "results/bpe/data_manifest.json")
    train = load_splits(source, ("train",))["train"].set_index("id", verify_integrity=True)
    roles = {role: train.loc[declared["ids"]].copy() for role, declared in manifest["roles"].items()}
    assert set(roles) == {"train", "validation", "audit"}
    assert sum(len(frame) for frame in roles.values()) == len(set().union(*(set(f.index) for f in roles.values())))
    vocab_path = ROOT / "results/sequences/vocabulary.json"
    assert file_sha256(vocab_path) == reference["vocabulary_sha256"]
    vocab = Vocabulary(read_json(vocab_path))
    def encode_prefix(text):
        return vocab.encode(crop(text, "max_128_palavras"), 256)

    seen_prefixes = {
        tuple(encode_prefix(text)) for role in ("train", "validation") for text in roles[role].clean_text
    }
    contrast, line_ids, identical = contrast_data(folder, seen_prefixes, encode_prefix)
    assert len(line_ids) == 344 and identical == 8
    synth = synthetic_data(lambda text: vocab.encode(text, None))
    for role, frame in roles.items():
        declared = manifest["roles"][role]
        assert frame.text_sha256.tolist() == declared["text_sha256"]
        assert sequences.array_hash(frame.label.to_numpy(dtype=np.int64)) == declared["labels_sha256"]
    roles["contrast"] = contrast.rename(columns={"text": "clean_text"})
    roles["synthetic"] = synth.rename(columns={"text": "clean_text"})
    return roles, vocab, line_ids


def run(source, folder, output, target):
    roles, vocab, line_ids = prepare_data(source, folder)
    output.mkdir(parents=True, exist_ok=False)
    weights = output / "weights"
    weights.mkdir()
    hashes = {p: file_sha256(ROOT / p) for p in (
        "seminario/treinar_cnn_pares.py", "seminario/treinar_cnn_comprimento.py",
        "seminario/medir_posicao.py", "seminario/treinar_pooling.py",
        "src/imdb_lab/models.py", "src/imdb_lab/sequences.py",
        "results/bpe/data_manifest.json", "results/sequences/vocabulary.json",
    )}
    save_json(output / "protocol.json", {
        "recipes": RECIPES, "seeds": list(SEEDS), "epochs": EPOCHS,
        "selection": "Menor CE de validação por semente. Contrastes avaliados após a seleção.",
        "scope": "Divisões internas do BPE e as mesmas 344 edições do ensaio de posição. Teste oficial fora deste ensaio.",
        "sources_sha256": hashes, "contrast_sha256": CONTRAST_FILES,
        "torch": str(torch.__version__),
    })
    labels = {role: frame.label.to_numpy(dtype=np.int64) for role, frame in roles.items()}
    last = roles["synthetic"].family.str.endswith("_antes").to_numpy()
    arrays = {role + "_labels": labels[role] for role in ("audit", "contrast", "synthetic")}
    arrays.update(contrast_tsv_rows=np.asarray(line_ids), audit_ids=roles["audit"].index.to_numpy(dtype=str),
                  synthetic_family=roles["synthetic"].family.to_numpy(dtype=str))
    records, aggregate, initialization = [], {}, {}
    for name, spec in RECIPES.items():
        encoded = {}
        for role, frame in roles.items():
            text = frame.clean_text.map(lambda value: crop(value, name))
            encoded[role] = encode_frame(frame.assign(clean_text=text), vocab)
        seen = {tuple(ids) for role in ("train", "validation") for ids in encoded[role]}
        assert all(tuple(ids) not in seen for role in ("contrast", "synthetic") for ids in encoded[role])
        data = {role: ReviewBatches(ids, labels[role], spec["cap"], max(spec["kernels"]), target)
                for role, ids in encoded.items() if role in ("train", "validation")}
        for seed in SEEDS:
            run_id = f"{name}__s{seed}"
            print("INÍCIO", run_id, flush=True)
            model = model_for(len(vocab.token_to_id), spec, seed, target)
            backbone = sequences.state_hash(torch.nn.ModuleList([model.embedding, model.convs]))
            assert initialization.setdefault(seed, backbone) == backbone
            state, result = train_one(model, data, labels, spec, seed, target)
            checkpoint = weights / f"{run_id}.pt"
            torch.save(state, checkpoint)
            result.update(recipe=name, seed=seed, initial_backbone_sha256=backbone,
                          checkpoint_sha256=file_sha256(checkpoint))
            records.append(result)
            save_json(output / "runs" / f"{run_id}.json", result)
    selection = output / "selection.json"
    save_json(selection, {"records": records, "protocol_sha256": file_sha256(output / "protocol.json")})
    results = []
    for name, spec in RECIPES.items():
        data = {}
        for role in ("audit", "contrast", "synthetic"):
            frame = roles[role]
            text = frame.clean_text.map(lambda value: crop(value, name))
            encoded = encode_frame(frame.assign(clean_text=text), vocab)
            data[role] = ReviewBatches(encoded, labels[role], spec["cap"], max(spec["kernels"]), target)
        for seed in SEEDS:
            run_id = f"{name}__s{seed}"
            checkpoint = weights / f"{run_id}.pt"
            expected = next(row["checkpoint_sha256"] for row in records if row["recipe"] == name and row["seed"] == seed)
            assert file_sha256(checkpoint) == expected
            model = model_for(len(vocab.token_to_id), spec, seed, target)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            scores = {role: margins(model, values) for role, values in data.items()}
            arrays.update({f"{role}_{run_id}": value for role, value in scores.items()})
            synthetics = scores["synthetic"]
            row = {"recipe": name, "seed": seed,
                   "audit_accuracy": raw_scores(scores["audit"], labels["audit"])["accuracy"],
                   "human_pair_accuracy": pair_summary(scores["contrast"], labels["contrast"]),
                   "verdict_last": pair_summary(synthetics[last], labels["synthetic"][last]),
                   "verdict_first": pair_summary(synthetics[~last], labels["synthetic"][~last])}
            results.append(row)
            print("AVALIAÇÃO", row, flush=True)
        subset = [row for row in results if row["recipe"] == name]
        aggregate[name] = {key: mean(row[key] for row in subset)
                           for key in ("audit_accuracy", "human_pair_accuracy", "verdict_last", "verdict_first")}
    predictions = output / "predictions.npz"
    np.savez_compressed(predictions, **arrays)
    save_json(output / "summary.json", {
        "selection_sha256": file_sha256(selection), "predictions_sha256": file_sha256(predictions),
        "human_pairs": len(line_ids), "synthetic_pairs_per_verdict_position": int(last.sum() // 2),
        "mean": aggregate, "runs": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--contrast-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(args.data, args.contrast_dir, args.output, sequences.configure(args.device))


if __name__ == "__main__":
    main()
