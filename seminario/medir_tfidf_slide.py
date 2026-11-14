"""Calcula todos os termos de uma entrada curta com o baseline original."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from imdb_lab.common import ROOT, file_sha256, read_json, save_json


HERE = Path(__file__).resolve().parent
TEXT = "Movie was awful."


def measure(checkpoint):
    expected = read_json(HERE / "dados/discordancias.json")["artefatos_sha256"]["tfidf/model.joblib"]
    if file_sha256(checkpoint) != expected:
        raise ValueError("Use o checkpoint original do TF-IDF.")
    pipeline = joblib.load(checkpoint)
    vectorizer, model = pipeline["vectorizer"], pipeline["model"]
    train_docs = read_json(ROOT / "data/manifest.json")["splits"]["train"]["n"]
    vector = vectorizer.transform([TEXT])
    counts = Counter(vectorizer.build_analyzer()(TEXT))
    assert len(counts) == vector.nnz == 5
    terms = []
    for term, count in counts.items():
        index = vectorizer.vocabulary_[term]
        idf = float(vectorizer.idf_[index])
        df = round((train_docs + 1) / np.exp(idf - 1) - 1)
        tf = 1 + np.log(count)
        weight = float(model.coef_[0, index])
        normalized = float(vector[0, index])
        terms.append({"term": term, "count": count, "document_frequency": df, "tf": float(tf),
                      "idf": idf, "tfidf": normalized, "coefficient": weight,
                      "contribution": normalized * weight})

    norm = float(np.linalg.norm([row["tf"] * row["idf"] for row in terms]))
    bias = float(model.intercept_[0])
    logit = float(model.decision_function(vector)[0])
    np.testing.assert_allclose([row["tf"] * row["idf"] / norm for row in terms],
                               [row["tfidf"] for row in terms])
    np.testing.assert_allclose(bias + sum(row["contribution"] for row in terms), logit)
    return {"text": TEXT, "train_documents": train_docs, "normalization": norm,
            "terms": terms, "bias": bias, "logit": logit,
            "positive_raw": float(expit(logit)), "class": "positiva" if logit >= 0 else "negativa",
            "checkpoint_sha256": expected, "source_sha256": file_sha256(Path(__file__))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "dados/tfidf_slide.json")
    args = parser.parse_args()
    save_json(args.output, measure(args.checkpoint))
