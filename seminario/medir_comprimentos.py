"""Resume comprimentos e recalcula métricas nas mesmas faixas para todos os modelos."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from imdb_lab.common import ROOT, file_sha256, read_json, save_json
from imdb_lab.metrics import evaluate
from imdb_lab.models import tokenize


PREDICTIONS = {
    "tfidf": "results/tfidf/predictions.parquet",
    "cnn": "results/cnn/predictions.parquet",
    "bert": "results/bert/predictions.parquet",
    "qwen": "results/qwen/classification/predictions.parquet",
}
LENGTH_BANDS = [(0, 100), (101, 250), (251, 500), (501, None)]
SENTENCE_BREAK = re.compile(r"""[.!?]+(?:["'”’)\]]+)?(?=\s|$)""")
REPLAY_FOLDER = ROOT / "seminario/dados/recorte_teste"


def describe(lengths):
    p10, median, p90 = np.quantile(lengths, [.1, .5, .9])
    return {"n": len(lengths), "mean": float(np.mean(lengths)), "min": int(np.min(lengths)),
            "p10": float(p10), "median": float(median), "p90": float(p90), "max": int(np.max(lengths))}


def load_data(folder, manifest, role):
    source = folder / f"{role}.parquet"
    if file_sha256(source) != manifest["splits"][role]["sha256"]:
        raise ValueError(f"Divisão inesperada: {role}.")
    data = pd.read_parquet(source, columns=["id", "label", "clean_text", "word_count"])
    assert data.id.is_unique and len(data) == manifest["splits"][role]["n"]
    return data.set_index("id")


def lengths(frame):
    words = frame.word_count.to_numpy()
    np.testing.assert_array_equal(words, frame.clean_text.map(lambda text: len(text.split())))
    regex = frame.clean_text.map(lambda text: len(tokenize(text))).to_numpy()
    sentence_lengths = [len(part.split()) for text in frame.clean_text
                        for part in SENTENCE_BREAK.split(text) if part.strip()]
    return regex, {"review_words": describe(words), "review_regex_tokens": describe(regex),
                   "sentence_words": describe(sentence_lengths)}


def saved_predictions(data):
    predictions = {}
    for model, source in PREDICTIONS.items():
        frame = pd.read_parquet(ROOT / source, columns=["id", "split", "label", "p_positive"])
        frame = frame.loc[frame.split.eq("test")].set_index("id", verify_integrity=True)
        assert set(frame.index) == set(data.index)
        frame = frame.loc[data.index]
        np.testing.assert_array_equal(frame.label, data.label)
        predictions[model] = frame.p_positive.to_numpy()
    return predictions


def replay_integral(frame, predictions, token_counts):
    source = REPLAY_FOLDER / "predicoes.npz"
    summary = read_json(REPLAY_FOLDER / "resumo.json")
    assert file_sha256(source) == summary["predicoes_pareadas_sha256"]
    cap = summary["limite_original"]
    with np.load(source, allow_pickle=False) as arrays:
        positions = frame.index.get_indexer(arrays["ids"])
        assert np.array_equal(np.sort(positions), np.flatnonzero(token_counts > cap))
        np.testing.assert_array_equal(token_counts[positions], arrays["tokens"])
        np.testing.assert_array_equal(frame.label.to_numpy()[positions], arrays["labels"])
        temperature = summary["temperatura_original"]
        np.testing.assert_allclose(predictions["cnn"][positions], expit(arrays["prefixo"] / temperature))
        whole = predictions["cnn"].copy()
        whole[positions] = expit(arrays["inteiro"].astype(np.float64) / temperature)
    predictions["cnn_integral"] = whole
    return cap


def metrics_by_length(frame, predictions, regex_counts, cap):
    words = frame.word_count.to_numpy()
    labels = frame.label.to_numpy()
    groups = []
    for low, high in LENGTH_BANDS:
        group = (words >= low) & ((words <= high) if high is not None else True)
        stats = {}
        for name, scores in predictions.items():
            values = evaluate(labels[group], scores[group])
            stats[name] = {key: values[key] for key in ("accuracy", "f1_macro", "log_loss")}
        groups.append({"min_words": low, "max_words": high, "n": int(group.sum()),
                       "positive_count": int(labels[group].sum()),
                       "cnn_truncated": int((regex_counts[group] > cap).sum()), "metrics": stats})
    assert sum(group["n"] for group in groups) == len(frame)
    return groups


def run(folder, output):
    manifest = read_json(ROOT / "data/manifest.json")
    train, test = (load_data(folder, manifest, role) for role in ("train", "test"))
    _, train_stats = lengths(train)
    regex_counts, test_stats = lengths(test)
    predictions = saved_predictions(test)
    cap = replay_integral(test, predictions, regex_counts)
    groups = metrics_by_length(test, predictions, regex_counts, cap)
    sources = [*PREDICTIONS.values(), "seminario/dados/recorte_teste/predicoes.npz",
               "seminario/dados/recorte_teste/resumo.json", "src/imdb_lab/models.py",
               "seminario/medir_comprimentos.py"]
    save_json(output, {
        "units": {"words": "Segmentos separados por espaço no texto limpo e inteiro.",
                  "regex_tokens": "Tokens da TextCNN por palavras, antes do recorte.",
                  "sentences": "Segmentação aproximada por . ! ? seguidos de espaço ou fim. Abreviações podem separar frases."},
        "lengths": {"train": train_stats, "test": test_stats},
        "predictions": "Teste oficial por ID. Para CNN integral, mesmos pesos e temperatura do teste original.",
        "cnn_cap": cap, "bands": groups,
        "split_sha256": {role: manifest["splits"][role]["sha256"] for role in ("train", "test")},
        "source_sha256": {source: file_sha256(ROOT / source) for source in sources},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "seminario/dados/comprimentos.json")
    args = parser.parse_args()
    run(args.data, args.output)
