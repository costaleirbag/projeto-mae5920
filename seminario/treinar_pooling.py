"""Compara resumos da sequência com treino de ponta a ponta da TextCNN."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from medir_posicao import (
    CONTRAST_FILES, CONTRAST_REV, HERE, ROOT, SEEDS, contrast_data,
    exact_control, pair_metrics, synthetic_data,
)
from imdb_lab import sequences
from imdb_lab.bpe_study import metrics
from imdb_lab.common import file_sha256, read_json, save_json
from imdb_lab.models import Vocabulary


POOLINGS = {
    "max_global": 1,
    "max_metades_valor": 2,
    "max_metades_posicao": 2,
    "media_global": 1,
    "media_metades_posicao": 2,
    "kmax2_ordem": 2,
}
MANIFEST = ROOT / "results/bpe/data_manifest.json"
VOCABULARY = ROOT / "results/sequences/vocabulary.json"


def masked_mean(features, keep):
    return features.masked_fill(~keep, 0).sum(2) / keep.sum(2).clamp_min(1)


def masked_max(features, keep):
    return features.masked_fill(~keep, -torch.inf).max(2).values.clamp_min(0)


def pool(features, valid_count, name):
    """Resume ativações após a ReLU. Metade vazia e pico ausente valem zero."""
    positions = torch.arange(features.shape[2], device=features.device)[None, None, :]
    count = valid_count[:, None, None]
    valid = positions < count
    if name == "media_global":
        return masked_mean(features, valid)
    if name == "kmax2_ordem":
        scores = features.masked_fill(~valid, -torch.inf)
        # Em empates, seleciona primeiro a posição mais à esquerda.
        indices = scores.argsort(dim=2, descending=True, stable=True)[:, :, :2]
        ordered = scores.gather(2, indices.sort(dim=2).values).clamp_min(0)
        return F.pad(ordered, (0, 2 - ordered.shape[2])).transpose(1, 2).flatten(1)

    midpoint = (count + 1) // 2
    early = positions < midpoint
    late = (positions >= midpoint) & valid
    reduce = masked_mean if name == "media_metades_posicao" else masked_max
    first, second = reduce(features, early), reduce(features, late)
    if name == "max_metades_valor":
        first, second = torch.minimum(first, second), torch.maximum(first, second)
    return torch.cat((first, second), dim=1)


class PoolingCNN(nn.Module):
    def __init__(self, cnn, name, seed, target):
        super().__init__()
        self.embedding = cnn.embedding
        self.convs = cnn.convs
        self.kernels = cnn.kernels
        self.dropout = cnn.dropout
        self.name = name
        self.classifier = cnn.classifier
        if POOLINGS[name] == 2:
            torch.manual_seed(seed + 50000)
            self.classifier = nn.Linear(2 * cnn.classifier.in_features, 2)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
            self.classifier.to(target)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids).transpose(1, 2)
        summary = []
        for conv, width in zip(self.convs, self.kernels):
            valid_count = (lengths - width + 1).clamp_min(1)
            summary.append(pool(conv(embedded).relu(), valid_count, self.name))
        return self.classifier(self.dropout(torch.cat(summary, dim=1)))


def plan(source):
    config = read_json(ROOT / "bpe_config.json")
    manifest = read_json(MANIFEST)
    source_sha = file_sha256(source)
    if source_sha != manifest["source_sha256"]:
        raise ValueError("Use o train.parquet original, com o hash do manifesto BPE.")
    assert file_sha256(VOCABULARY) == manifest["word_vocabulary_sha256"]
    assert tuple(config["seeds"]) == SEEDS
    common = sequences.settings()["common_input"]
    assert all(config[key] == common[key] for key in ("whitespace_words", "token_cap"))
    sources = [Path(__file__).resolve().relative_to(ROOT).as_posix(),
               "seminario/medir_posicao.py", "seminario/casos.json", "bpe_config.json", "sequence_config.json",
               "src/imdb_lab/sequences.py", "src/imdb_lab/models.py"]
    return {"source_sha256": source_sha, "manifest_sha256": file_sha256(MANIFEST),
            "vocabulary_sha256": file_sha256(VOCABULARY),
            "code_sha256": {name: file_sha256(ROOT / name) for name in sources},
            "poolings": POOLINGS, "seeds": list(SEEDS), "cnn": config["cnn"],
            "whitespace_words": config["whitespace_words"], "token_cap": config["token_cap"],
            "primary": "Acerto nas duas versões dos pares sintéticos com veredito no fim",
            "selection": "Menor CE de validação entre as épocas 1 a 12, por treino"}


def load_roles(source, roles, words):
    manifest = read_json(MANIFEST)["roles"]
    ids = [item for role in roles for item in manifest[role]["ids"]]
    assert len(ids) == len(set(ids))
    dataset = pd.read_parquet(source, filters=[("id", "in", ids)],
                              columns=["id", "clean_text", "label", "text_sha256", "official_split"])
    assert dataset.official_split.eq("train").all() and dataset.id.is_unique
    indexed = dataset.set_index("id")
    frames = {}
    for role in roles:
        expected = manifest[role]
        frame = indexed.loc[expected["ids"]].copy()
        assert frame.text_sha256.tolist() == expected["text_sha256"]
        assert sequences.array_hash(frame.label.to_numpy(dtype=np.int64)) == expected["labels_sha256"]
        frame["text"] = frame.clean_text.map(lambda text: sequences.excerpt(text, words))
        assert frame.text.map(lambda text: hashlib.sha256(text.encode()).hexdigest()).tolist() == expected["excerpt_sha256"]
        frames[role] = frame
    return frames


def inputs(frame, target):
    arrays, _ = sequences.cnn_inputs(frame, torch.device("cpu"))
    return {key: value.to(target) for key, value in arrays.items()}


def model_for(name, recipe, seed, target):
    cnn = sequences.initialize_cnn(recipe, seed, target)
    return cnn if name == "max_global" else PoolingCNN(cnn, name, seed, target)


def train(source, output, target):
    protocol = plan(source)
    frames = load_roles(source, ("train", "validation"), protocol["whitespace_words"])
    sets = {role: (inputs(frame, target), torch.as_tensor(frame.label.to_numpy(), dtype=torch.long, device=target))
            for role, frame in frames.items()}
    output.mkdir(parents=True, exist_ok=False)
    weights = output / "weights"
    weights.mkdir()
    save_json(output / "protocol.json", protocol)
    records = []
    for seed in SEEDS:
        reference = None
        heads = {}
        for name, multiplier in POOLINGS.items():
            run_id = f"{name}__s{seed}"
            spec = dict(protocol["cnn"], run_id=run_id, pooling=name)
            model = model_for(name, spec, seed, target)
            backbone = sequences.state_hash(nn.ModuleList([model.embedding, model.convs]))
            if reference is None:
                reference = backbone
            assert backbone == reference
            head = sequences.state_hash(model.classifier)
            assert head == heads.setdefault(multiplier, head)
            record = sequences.train_one(model, sets["train"], sets["validation"], spec, seed, target)
            state = record.pop("best_state")
            record.pop("best_predictions")
            checkpoint = weights / f"{run_id}.pt"
            torch.save(state, checkpoint)
            record.update(backbone_initial_sha256=backbone, checkpoint_sha256=file_sha256(checkpoint))
            records.append(record)
            save_json(output / "runs" / f"{run_id}.json", record)
            print("CONCLUÍDO", run_id, "época", record["best_epoch"], "segundos", round(record["total_seconds"], 1), flush=True)
    save_json(output / "selection.json", {"protocol_sha256": file_sha256(output / "protocol.json"),
              "runtime": {"torch": str(torch.__version__), "precision": "FP32, TF32 desativado",
                          "device": torch.cuda.get_device_name(target) if target.type == "cuda" else target.type},
              "records": records})


def pair_results(logits, labels):
    result = pair_metrics(logits, labels)
    guesses = logits.argmax(1).reshape(-1, 2)
    result["fracao_pares_mesma_predicao"] = float((guesses[:, 0] == guesses[:, 1]).mean())
    return result


def summarize(arrays, records):
    families = arrays["synthetic_family"]
    verdict_last = np.isin(families, ("expectativa_antes", "impressao_antes"))
    # Cada família percorre os mesmos pares de adjetivos e rótulos na mesma ordem.
    moved = np.concatenate([np.column_stack((np.flatnonzero(families == kind + "_antes"),
                                             np.flatnonzero(families == kind + "_depois"))).ravel()
                            for kind in ("expectativa", "impressao")])
    synth_labels = arrays["synthetic_labels"]
    np.testing.assert_array_equal(synth_labels[moved][::2], synth_labels[moved][1::2])
    visible = arrays["contrast_distinct"]
    runs = []
    for record in records:
        spec = record["spec"]
        key = spec["run_id"]
        synthetic = arrays[f"synthetic_{key}"]
        contrast = arrays[f"contrast_{key}"]
        contrast_labels = arrays["contrast_labels"]
        control = arrays[f"control_{key}"]
        margin = control[:, 1] - control[:, 0]
        if spec["pooling"] in ("max_global", "media_global", "max_metades_valor"):
            np.testing.assert_allclose(control[0], control[1], atol=2e-5, rtol=0)
        runs.append({"pooling": spec["pooling"], "seed": record["seed"], "epoch": record["best_epoch"],
                     "audit": metrics(arrays[f"audit_{key}"], arrays["audit_labels"]),
                     "verdict_last": pair_results(synthetic[verdict_last], synth_labels[verdict_last]),
                     "verdict_first": pair_results(synthetic[~verdict_last], synth_labels[~verdict_last]),
                     "same_verdict_moved": pair_results(synthetic[moved], synth_labels[moved]),
                     "contrast": pair_results(contrast, contrast_labels),
                     "contrast_distinct": pair_results(contrast[visible], contrast_labels[visible]),
                     "control_margin": margin.tolist()})
    scores = {}
    for name in POOLINGS:
        subset = [row for row in runs if row["pooling"] == name]
        scores[name] = {"audit_accuracy": mean(row["audit"]["accuracy"] for row in subset)}
        for group in ("verdict_last", "verdict_first", "same_verdict_moved", "contrast", "contrast_distinct"):
            scores[name][group] = mean(row[group]["fracao_pares"] for row in subset)
    return runs, scores


@torch.inference_mode()
def logits_for(model, tensors):
    model.eval()
    batches = range(0, len(tensors["lengths"]), 64)
    return torch.cat([model(**sequences.indexed(tensors, slice(start, start + 64))).cpu()
                      for start in batches]).numpy()


def audit(source, contrast_folder, output, target):
    protocol = read_json(output / "protocol.json")
    selection = read_json(output / "selection.json")
    assert protocol == plan(source)
    assert file_sha256(output / "protocol.json") == selection["protocol_sha256"]
    records = selection["records"]
    assert [(row["seed"], row["spec"]["pooling"]) for row in records] == [
        (seed, name) for seed in SEEDS for name in POOLINGS]
    for record in records:
        checkpoint = output / "weights" / f"{record['spec']['run_id']}.pt"
        assert file_sha256(checkpoint) == record["checkpoint_sha256"]

    words, cap = protocol["whitespace_words"], protocol["token_cap"]
    frames = load_roles(source, ("train", "validation", "audit"), words)
    vocabulary = Vocabulary(read_json(VOCABULARY))

    def encode(text):
        return vocabulary.encode(sequences.excerpt(text, words), cap)

    seen = {tuple(encode(text)) for role in ("train", "validation") for text in frames[role].text}
    del frames["train"], frames["validation"]
    frames["contrast"], row_ids, identical = contrast_data(contrast_folder, seen, encode)
    frames["synthetic"] = synthetic_data(encode)
    case = next(case for case in read_json(HERE / "casos.json")["pares"] if case["id"] == "janelas")
    exact_control(case, encode, tuple(protocol["cnn"]["kernels"]))
    frames["control"] = pd.DataFrame({"text": [case["a"], case["b"]]})
    for role in ("contrast", "synthetic", "control"):
        frames[role]["text"] = frames[role].text.map(lambda text: sequences.excerpt(text, words))
    left, right = frames["contrast"].text.iloc[::2], frames["contrast"].text.iloc[1::2]
    distinct = np.repeat([encode(a) != encode(b) for a, b in zip(left, right, strict=True)], 2)
    assert int((~distinct).sum()) == 2 * identical
    arrays = {f"{role}_labels": frames[role].label.to_numpy(dtype=np.int64)
              for role in ("audit", "synthetic", "contrast")}
    arrays.update(contrast_tsv_rows=np.asarray(row_ids), contrast_distinct=distinct,
                  synthetic_family=frames["synthetic"].family.to_numpy(dtype=str))
    tensors = {role: inputs(frame, target) for role, frame in frames.items()}
    for record in records:
        spec, seed = record["spec"], record["seed"]
        model = model_for(spec["pooling"], spec, seed, target)
        checkpoint = output / "weights" / f"{spec['run_id']}.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        for role, batch in tensors.items():
            arrays[f"{role}_{spec['run_id']}"] = logits_for(model, batch)
    runs, scores = summarize(arrays, records)
    prediction_path = output / "predictions.npz"
    np.savez_compressed(prediction_path, **arrays)
    save_json(output / "summary.json", {"selection_sha256": file_sha256(output / "selection.json"),
              "predictions_sha256": file_sha256(prediction_path), "contrast_revision": CONTRAST_REV,
              "contrast_files_sha256": CONTRAST_FILES, "contrast_pairs": len(row_ids),
              "contrast_pairs_identical_after_crop": identical, "control": case["id"],
              "runs": runs, "mean": scores})
    print("MÉDIAS", scores, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "audit"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contrast-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "audit" and args.contrast_dir is None:
        parser.error("audit requer --contrast-dir")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    target = sequences.configure(args.device)
    if args.stage == "train":
        train(args.source, args.output, target)
    else:
        audit(args.source, args.contrast_dir, args.output, target)


if __name__ == "__main__":
    main()
