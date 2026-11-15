"""Aplica o baseline original aos dois comentários do slide sobre o YouTube."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from imdb_lab.common import file_sha256, read_json, save_json
from imdb_lab.data import clean_text


DATA = Path(__file__).resolve().parent / "dados"


def measure(checkpoint: Path) -> dict:
    expected = read_json(DATA / "discordancias.json")["artefatos_sha256"]["tfidf/model.joblib"]
    if file_sha256(checkpoint) != expected:
        raise ValueError("Use o checkpoint original do TF-IDF.")

    youtube_cases = read_json(DATA / "fora_dominio_resultados.json")["comentarios_youtube"]
    cases = {case["id"]: case for case in youtube_cases}
    selected = [cases[case_id] for case_id in ("E", "F")]
    pipeline = joblib.load(checkpoint)
    model = pipeline["model"]
    features = pipeline["vectorizer"].transform([clean_text(case["texto"]) for case in selected])
    logits = model.decision_function(features)
    positive_scores = model.predict_proba(features)[:, list(model.classes_).index(1)]
    labels = {0: "negativa", 1: "positiva"}

    return {
        "protocolo": "Uma previsão por comentário com o TF-IDF e a regressão logística do estudo original.",
        "checkpoint_sha256": expected,
        "comentarios": [
            {
                **{key: case[key] for key in ("id", "origem", "texto", "rotulo_referencia")},
                "classe": labels[label],
                "logit": float(logit),
                "p_positiva_sem_calibracao": float(score),
            }
            for case, label, logit, score in zip(selected, model.predict(features), logits, positive_scores, strict=True)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DATA / "tfidf_youtube_resultados.json")
    args = parser.parse_args()
    save_json(args.output, measure(args.checkpoint))
