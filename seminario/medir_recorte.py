"""Compara prefixo e texto inteiro no teste oficial com a mesma CNN treinada."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.utils.data import DataLoader

from demo import HERE, ROOT
from imdb_lab.common import file_sha256, read_json, save_json
from imdb_lab.metrics import evaluate
from imdb_lab.models import EncodedReviews, PadBatch, TextCNN, Vocabulary, tokenize
from imdb_lab.training import predict


def scores(labels, prefix, full, temperature):
    before, after = prefix >= 0, full >= 0
    before_correct, after_correct = before == labels, after == labels
    result = {"n": len(labels), "corrigiu": int((~before_correct & after_correct).sum()),
              "passou_a_errar": int((before_correct & ~after_correct).sum())}
    for name, logits in (("prefixo", prefix), ("inteiro", full)):
        metrics = evaluate(labels, expit(logits.astype(np.float64) / temperature))
        result[name] = {key: metrics[key] for key in ("accuracy", "f1_macro", "log_loss")}
        result[name]["acertos"] = int(((logits >= 0) == labels).sum())
    return result


def measure(test: Path, artifacts: Path, output: Path) -> dict:
    torch.set_num_threads(4)
    reference = read_json(HERE / "dados/discordancias.json")
    test_sha = file_sha256(test)
    assert test_sha == reference["teste_sha256"]
    prediction_path = ROOT / "results/cnn/predictions.parquet"
    prediction_sha = file_sha256(prediction_path)
    assert prediction_sha == reference["fontes_sha256"]["results/cnn/predictions.parquet"]
    checkpoint, vocabulary = artifacts / "cnn/model.pt", artifacts / "cnn/vocabulary.json"
    hashes = {path.relative_to(artifacts).as_posix(): file_sha256(path)
              for path in (checkpoint, vocabulary)}
    assert all(value == reference["artefatos_sha256"][key] for key, value in hashes.items())
    cfg = read_json(ROOT / "config.json")["cnn"]
    meta = read_json(ROOT / "results/cnn/metadata.json")
    cap = cfg["max_length"]
    assert cfg == meta["config"] and cap == reference["limite_cnn_tokens"]
    assert meta["threshold"] == .5 and meta["temperature"] > 0

    data = pd.read_parquet(test, columns=["id", "clean_text", "label"]).set_index("id")
    saved = pd.read_parquet(prediction_path).query('split == "test"').set_index("id").loc[data.index]
    labels = data.label.to_numpy()
    np.testing.assert_array_equal(saved.label, labels)
    original = saved.logit.to_numpy()
    np.testing.assert_array_equal(saved.prediction, original >= 0)
    lengths = data.clean_text.map(lambda text: len(tokenize(text))).to_numpy()
    cropped = np.flatnonzero(lengths > cap)
    assert len(cropped) == reference["com_recorte"]["n"]
    # Agrupa comprimentos próximos para limitar padding na inferência.
    order = cropped[np.argsort(lengths[cropped], kind="stable")]
    vocab = Vocabulary(read_json(vocabulary))
    full = [vocab.encode(text, int(length))
            for text, length in zip(data.clean_text.iloc[order], lengths[order], strict=True)]
    prefix = [ids[:cap] for ids in full]
    model = TextCNN(len(vocab.token_to_id), cfg["embedding_dim"], cfg["filters"], cfg["kernels"], cfg["dropout"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    prefix_loader = DataLoader(EncodedReviews(prefix, labels[order]), batch_size=32, collate_fn=PadBatch("cnn"))
    full_loader = DataLoader(EncodedReviews(full, labels[order]), batch_size=32, collate_fn=PadBatch("cnn"))
    replay = predict(model, prefix_loader, torch.device("cpu"))
    np.testing.assert_allclose(replay, original[order], atol=2e-4, rtol=0)
    np.testing.assert_array_equal(replay >= 0, original[order] >= 0)
    print(f"{len(order)} prefixos conferidos com as predições publicadas.", flush=True)
    full_logits = predict(model, full_loader, torch.device("cpu"))
    assert np.isfinite(full_logits).all()
    whole = original.copy()
    whole[order] = full_logits
    tau = meta["temperature"]

    output.mkdir(parents=True, exist_ok=False)
    paired = output / "predicoes.npz"
    np.savez_compressed(paired, ids=data.index.to_numpy(dtype=str)[order], labels=labels[order],
                        tokens=lengths[order], prefixo=original[order], inteiro=full_logits)
    return {"teste_sha256": test_sha, "predicoes_originais_sha256": prediction_sha,
            "artefatos_sha256": hashes, "predicoes_pareadas_sha256": file_sha256(paired),
            "limite_original": cap, "maior_resenha_em_tokens": int(lengths.max()),
            "temperatura_original": tau, "max_diferenca_replay_prefixo": float(np.max(np.abs(replay - original[order]))),
            "teste": scores(labels, original, whole, tau),
            "resenhas_recortadas": scores(labels[order], original[order], full_logits, tau)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.test, args.artifacts, args.output)
    save_json(args.output / "resumo.json", result)
    for name in ("resenhas_recortadas", "teste"):
        print(name, result[name], flush=True)


if __name__ == "__main__":
    main()
