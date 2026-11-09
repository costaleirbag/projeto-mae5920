"""Isola o efeito do passo na inferência das CNNs por palavras já treinadas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch

from demo import HERE, ROOT, load_models
from imdb_lab.bpe_study import metrics
from imdb_lab.common import file_sha256, read_json
from imdb_lab.sequences import cnn_inputs, excerpt, indexed


STRIDES = (1, 2, 3)
SEEDS = (17, 29, 43)


@torch.inference_mode()
def predict_strides(model, inputs: dict, n: int) -> dict[int, np.ndarray]:
    chunks = {stride: [] for stride in STRIDES}
    for start in range(0, n, 64):
        batch = indexed(inputs, slice(start, start + 64))
        embedding = model.embedding(batch["input_ids"]).transpose(1, 2)
        pooled = {stride: [] for stride in STRIDES}
        for conv, width in zip(model.convs, model.kernels):
            features = conv(embedding).relu()
            valid = (batch["lengths"] - width + 1).clamp_min(1)
            for stride in STRIDES:
                sampled = features[:, :, ::stride]
                positions = torch.arange(0, features.shape[-1], stride)
                keep = positions[None, None, :] < valid[:, None, None]
                activations = sampled.masked_fill(~keep, torch.finfo(sampled.dtype).min)
                pooled[stride].append(activations.amax(dim=2))
        for stride in STRIDES:
            logits = model.classifier(torch.cat(pooled[stride], dim=1))
            chunks[stride].append(logits.numpy())
    return {stride: np.concatenate(parts) for stride, parts in chunks.items()}


def evaluate(source: Path, artifacts: Path, output: Path) -> dict:
    torch.set_num_threads(4)
    manifest_path = ROOT / "results/bpe/data_manifest.json"
    manifest = read_json(manifest_path)
    if file_sha256(source) != manifest["source_sha256"]:
        raise ValueError("Arquivo de treino diferente daquele usado na auditoria de BPE.")
    role = manifest["roles"]["audit"]
    frame = pd.read_parquet(source).set_index("id").loc[role["ids"]]
    assert frame["text_sha256"].tolist() == role["text_sha256"]
    n = len(frame)
    labels = frame["label"].to_numpy(dtype=np.int64)
    config = read_json(ROOT / "bpe_config.json")
    frame["text"] = frame["clean_text"].map(lambda text: excerpt(text, config["whitespace_words"]))
    inputs, _ = cnn_inputs(frame, torch.device("cpu"))
    original_path = ROOT / "results/bpe/audit_predictions.npz"
    runs, checkpoints = [], {}
    arrays = {"labels": labels}
    with np.load(original_path, allow_pickle=False) as original:
        np.testing.assert_array_equal(labels, original["labels"])
        for seed in SEEDS:
            word, _ = load_models(artifacts, seed)
            checkpoints[str(seed)] = word.checkpoint_sha256
            logits = predict_strides(word.model, inputs, n)
            reference = original[f"word__s{seed}"]
            np.testing.assert_allclose(logits[1], reference, atol=5e-5, rtol=0)
            np.testing.assert_array_equal(logits[1].argmax(1), reference.argmax(1))
            for stride, values in logits.items():
                arrays[f"passo{stride}_s{seed}"] = values
                runs.append({"semente": seed, "passo": stride, **metrics(values, labels),
                             "mudancas_vs_passo_1": int(np.sum(values.argmax(1) != logits[1].argmax(1)))})
            print(f"Semente {seed}, maior diferença para a auditoria original: {np.max(np.abs(logits[1] - reference)):.2g}",
                  flush=True)

    lengths = inputs["lengths"].numpy()
    valid = [np.maximum(1, lengths - width + 1) for width in config["cnn"]["kernels"]]
    avg = {}
    for stride in STRIDES:
        group = [run for run in runs if run["passo"] == stride]
        avg[str(stride)] = {key: mean(run[key] for run in group)
                            for key in ("accuracy", "f1_macro", "ce", "mudancas_vs_passo_1")}
        avg[str(stride)]["fracao_de_janelas"] = sum(((length - 1) // stride + 1).sum() for length in valid) / sum(
            length.sum() for length in valid
        )

    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "passo_predicoes.npz"
    np.savez_compressed(predictions, **arrays)
    return {
        "protocolo": "Pesos congelados. Usamos janelas de início 0, s, 2s, ... com s igual a 1, 2 ou 3.",
        "n": n, "tokenizador": "Palavras", "checkpoint_sha256": checkpoints,
        "manifesto_sha256": file_sha256(manifest_path), "auditoria_original_sha256": file_sha256(original_path),
        "predicoes_sha256": file_sha256(predictions), "medias": avg, "execucoes": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Arquivo train.parquet original.")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=HERE / "dados")
    args = parser.parse_args()
    data = evaluate(args.train, args.artifacts, args.output_dir)
    (args.output_dir / "passo_resultados.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(data["medias"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
