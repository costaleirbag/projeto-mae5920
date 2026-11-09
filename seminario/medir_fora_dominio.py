"""Consulta as CNNs congeladas em frases da UCI e em comentários do YouTube."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

import numpy as np
import torch

from demo import HERE, ROOT, load_models
from imdb_lab.bpe_study import metrics
from imdb_lab.common import file_sha256, read_json
from imdb_lab.data import clean_text, fingerprint


DATASET_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
FILES = {"Amazon": "amazon_cells_labelled.txt", "Yelp": "yelp_labelled.txt"}
COMMENTS = {
    "A": ("Baby", "Let's be honest, Justin Bieber owned the world for few days when this song was released"),
    "B": ("Baby", "why 7million dislike for this song?"),
    "C": ("Friday", "AI could never do this"),
    "D": ("The Room", "The trailer makes more sense than the actual film"),
    "E": ("The Room", '"The best movie of the year" - No one'),
}
SEEDS = (17, 29, 43)


def evaluate(dataset: Path, artifacts: Path, output: Path) -> dict:
    torch.set_num_threads(2)
    models = [model for seed in SEEDS for model in load_models(artifacts, seed)]
    manifests = [ROOT / f"results/{family}/data_manifest.json" for family in ("sequences", "bpe")]
    selection_hashes = {
        text_hash
        for path in manifests
        for role, part in read_json(path)["roles"].items()
        if role in {"train", "validation"}
        for text_hash in part["text_sha256"]
    }
    domains = {}
    arrays = {}
    with ZipFile(dataset) as archive:
        for domain, name in FILES.items():
            member = f"sentiment labelled sentences/{name}"
            contents = archive.read(member)
            parsed = [line.rsplit("\t", 1) for line in contents.decode("utf-8").splitlines()]
            rows = [(i, review, int(label)) for i, (review, label) in enumerate(parsed, 1)]
            if Counter(label for _, _, label in rows) != {0: 500, 1: 500}:
                raise ValueError(f"Número ou rótulos de linhas inesperados em {member}.")
            selected = [row for row in rows if fingerprint(row[1]) not in selection_hashes]
            row_ids, texts, labels = zip(*selected)
            key = domain.lower()
            arrays[f"{key}_linhas"] = np.array(row_ids, dtype=np.int64)
            arrays[f"{key}_rotulos"] = np.array(labels, dtype=np.int64)
            runs = []
            for model in models:
                logits = np.array([model.predict(review)["logits"] for review in texts], dtype=np.float32)
                arrays[f"{key}_{model.label.lower()}_s{model.seed}"] = logits
                runs.append({"modelo": model.label, "semente": model.seed, **metrics(logits, labels)})
            averages = {}
            for label in ("Palavras", "BPE"):
                group = [run for run in runs if run["modelo"] == label]
                averages[label] = {metric: mean(run[metric] for run in group)
                                   for metric in ("accuracy", "f1_macro", "ce")}
            domains[domain] = {
                "arquivo": member, "sha256": sha256(contents).hexdigest(),
                "n": len(selected), "positivos": sum(labels),
                "excluidos_por_igualdade_com_treino_ou_validacao": len(rows) - len(selected),
                "mediana_palavras": median(len(clean_text(review).split()) for review in texts),
                "medias": averages, "execucoes": runs,
            }

    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "fora_dominio_predicoes.npz"
    np.savez_compressed(predictions, **arrays)
    return {
        "protocolo": "Inferência em CPU com as CNNs congeladas, sem ajuste de pesos, limiar ou temperatura.",
        "dataset": {"nome": "Sentiment Labelled Sentences", "doi": "10.24432/C57604",
                    "download": DATASET_URL, "licenca": "CC BY 4.0", "zip_sha256": file_sha256(dataset)},
        "manifestos_treino_validacao": {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in manifests},
        "checkpoint_sha256": {f"{model.label}_s{model.seed}": model.checkpoint_sha256 for model in models},
        "predicoes_sha256": file_sha256(predictions),
        "dominios_rotulados": domains,
        "comentarios_youtube": [
            {"id": name, "origem": source, "texto": review, "rotulo_referencia": None,
             "predicoes": [model.predict(review) for model in models]}
            for name, (source, review) in COMMENTS.items()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="ZIP original da UCI.")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=HERE / "dados")
    args = parser.parse_args()
    result = evaluate(args.dataset, args.artifacts, args.output_dir)
    (args.output_dir / "fora_dominio_resultados.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    for name, domain in result["dominios_rotulados"].items():
        print(name, domain["n"], domain["medias"])
    for comment in result["comentarios_youtube"]:
        print(comment["id"], [(p["modelo"], p["semente"], round(p["p_positiva"], 4))
                              for p in comment["predicoes"]])


if __name__ == "__main__":
    main()
