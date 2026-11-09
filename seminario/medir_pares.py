"""Avalia os pares sintéticos fixados em casos.json com os seis pesos congelados."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

import torch

from demo import HERE, ROOT, classify, load_models
from imdb_lab.common import file_sha256, read_json
from imdb_lab.models import tokenize


def windows(sequence: list, width: int) -> Counter:
    return Counter(tuple(sequence[start:start + width]) for start in range(len(sequence) - width + 1))


def evaluate(artifacts: Path) -> dict:
    torch.set_num_threads(2)
    cases_path = HERE / "casos.json"
    cases = read_json(cases_path)
    seeds = [17, 29, 43]
    models_by_seed = {seed: load_models(artifacts, seed) for seed in seeds}
    evaluated = []
    for case in cases["pares"]:
        tokens_a, tokens_b = tokenize(case["a"]), tokenize(case["b"])
        assert Counter(tokens_a) == Counter(tokens_b) and tokens_a != tokens_b, case["id"]
        a = {seed: classify(models, case["a"]) for seed, models in models_by_seed.items()}
        b = {seed: classify(models, case["b"]) for seed, models in models_by_seed.items()}
        results = {}
        for index, model in enumerate(models_by_seed[17]):
            pred_a, pred_b = a[17][index], b[17][index]
            matches = {width: windows(pred_a["ids"], width) == windows(pred_b["ids"], width)
                       for width in model.model.kernels}
            comparisons = [
                {"semente": seed, "a": a[seed][index], "b": b[seed][index]}
                for seed in seeds
            ]
            difference = max(abs(x - y) for r in comparisons
                             for x, y in zip(r["a"]["logits"], r["b"]["logits"]))
            if case["id"] == "janelas":
                assert all(matches.values()) and difference < 1e-5, (model.label, difference)
            results[model.label] = {
                "janelas_iguais": matches,
                "media_p_a": mean(r["a"]["p_positiva"] for r in comparisons),
                "media_p_b": mean(r["b"]["p_positiva"] for r in comparisons),
                "mudancas_de_classe": sum(r["a"]["classe"] != r["b"]["classe"] for r in comparisons),
                "maior_diferenca_de_logit": difference,
                "predicoes": comparisons,
            }
        evaluated.append({
            **case, "unigramas_iguais": True,
            "bigramas_iguais": windows(tokens_a, 2) == windows(tokens_b, 2),
            "resultados": results,
        })
    return {
        "protocolo": "Pares sintéticos fixos, inferência CPU com pesos congelados e escores de softmax sem calibração.",
        "casos_sha256": file_sha256(cases_path),
        "sementes": seeds,
        "checkpoints": [
            {"modelo": model.label, "semente": model.seed, "epoca": model.epoch,
             "sha256": model.checkpoint_sha256}
            for models in models_by_seed.values() for model in models
        ],
        "pares": evaluated,
        "cobertura": {"texto": cases["cobertura"],
                      "predicoes": classify(models_by_seed[17], cases["cobertura"])},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=HERE / "dados/pares_resultados.json")
    args = parser.parse_args()
    result = evaluate(args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for pair in result["pares"]:
        print(pair["id"], "bigramas iguais:", pair["bigramas_iguais"])
        for name, record in pair["resultados"].items():
            print(f"  {name:<8} A {record['media_p_a']:.4f} B {record['media_p_b']:.4f}"
                  f" | mudanças {record['mudancas_de_classe']}/3 | janelas {record['janelas_iguais']}"
                  f" | Δ logit máx. {record['maior_diferenca_de_logit']:.6g}")
    print("Cobertura:", [p["tokens"] for p in result["cobertura"]["predicoes"]])


if __name__ == "__main__":
    main()
