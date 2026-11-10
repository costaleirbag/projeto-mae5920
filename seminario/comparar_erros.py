"""Cruza previsões do teste oficial e reavalia dois casos com pesos congelados."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from demo import HERE, ROOT
from imdb_lab.common import file_sha256, read_json
from imdb_lab.models import PadBatch, TextCNN, TOKEN_PATTERN, Vocabulary, tokenize


EXAMPLES = {"imdb_test_03531": "tfidf", "imdb_test_22831": "cnn"}
EXCERPTS = {
    "prefixo": "This film is as crassly manipulative as it is vapid",
    "final": "I had to leave because it was so bad.",
}


def locate_excerpt(text: str, excerpt: str, tokens: list) -> dict:
    start = text.index(excerpt)
    end = start + len(excerpt)
    positions = [index for index, token in enumerate(tokens, 1) if token.start() < end and token.end() > start]
    return {"texto": excerpt, "primeiro_token": positions[0], "ultimo_token": positions[-1]}


def counts(frame: pd.DataFrame) -> dict:
    tfidf = frame["tfidf"].eq(frame["label"])
    cnn = frame["cnn"].eq(frame["label"])
    return {"n": len(frame), "ambos_acertam": int((tfidf & cnn).sum()),
            "so_tfidf_acerta": int((tfidf & ~cnn).sum()), "so_cnn_acerta": int((~tfidf & cnn).sum()),
            "ambos_erram": int((~tfidf & ~cnn).sum())}


def strongest(contributions: list[tuple[float, str]]) -> dict:
    ordered = sorted(contributions)
    return {
        "a_favor_da_classe_negativa": [{"termo": term, "contribuicao": value}
                                       for value, term in ordered[:3] if value < 0],
        "a_favor_da_classe_positiva": [{"termo": term, "contribuicao": value}
                                       for value, term in reversed(ordered[-3:]) if value > 0],
    }


def tfidf_score(text: str, pipeline: dict, temperature: float) -> dict:
    vectorizer, model = pipeline["vectorizer"], pipeline["model"]
    vector = vectorizer.transform([text])
    logit = float(model.decision_function(vector)[0])
    names = vectorizer.get_feature_names_out()
    contributions = vector.data * model.coef_[0, vector.indices]
    return {"logit": logit, "p_positiva": float(expit(logit / temperature)),
            "termos": strongest(list(zip(contributions.tolist(), names[vector.indices].tolist())))}


@torch.inference_mode()
def cnn_score(text: str, cap: int, model: TextCNN, vocab: Vocabulary, temperature: float) -> dict:
    ids = vocab.encode(text, cap)
    tokens = [word if index != 1 else "<unk>" for word, index in zip(tokenize(text), ids)]
    batch, _ = PadBatch("cnn")([(ids, 0)])
    logits = model(**batch)[0]
    margin = float(logits[1] - logits[0])
    weights = (model.classifier.weight[1] - model.classifier.weight[0]).split(model.convs[0].out_channels)
    embedding = model.embedding(batch["input_ids"]).transpose(1, 2)
    contributions = []
    for conv, width, coefficient in zip(model.convs, model.kernels, weights):
        values, positions = conv(embedding)[0, :, :max(1, len(ids) - width + 1)].relu().max(dim=1)
        for index, contribution in zip(positions.tolist(), (values * coefficient).tolist()):
            contributions.append((contribution, " ".join(tokens[index:index + width])))
    assert abs(sum(value for value, _ in contributions) + float(model.classifier.bias[1] - model.classifier.bias[0]) - margin) < 1e-5
    return {"logit": margin, "p_positiva": float(expit(margin / temperature)),
            "janelas": strongest(contributions)}


def evaluate(test: Path, artifacts: Path) -> dict:
    torch.set_num_threads(4)
    dataset_sha = read_json(ROOT / "data/manifest.json")["splits"]["test"]["sha256"]
    assert file_sha256(test) == dataset_sha
    data = pd.read_parquet(test).set_index("id")
    sources = {}
    for name in ("cnn", "tfidf"):
        path = ROOT / f"results/{name}/predictions.parquet"
        scores = pd.read_parquet(path).query('split == "test"').set_index("id").loc[data.index]
        assert np.array_equal(scores["label"], data["label"])
        data[name] = scores["prediction"]
        data[f"{name}_logit"] = scores["logit"]
        sources[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    cfg = read_json(ROOT / "config.json")["cnn"]
    cap = cfg["max_length"]
    token_count = data["clean_text"].map(lambda text: len(tokenize(text)))
    truncated = token_count.gt(cap)
    tfidf_path = artifacts / "tfidf/model.joblib"
    cnn_path = artifacts / "cnn/model.pt"
    vocab_path = artifacts / "cnn/vocabulary.json"
    # model.joblib is the trusted local checkpoint produced by this project.
    pipeline = joblib.load(tfidf_path)
    vocab = Vocabulary(read_json(vocab_path))
    model = TextCNN(len(vocab.token_to_id), cfg["embedding_dim"], cfg["filters"], cfg["kernels"], cfg["dropout"])
    model.load_state_dict(torch.load(cnn_path, map_location="cpu", weights_only=True))
    model.eval()
    temperature = {k: read_json(ROOT / f"results/{k}/metadata.json")["temperature"] for k in ("cnn", "tfidf")}

    examples = []
    for id_, correct in EXAMPLES.items():
        row = data.loc[id_]
        assert all((row[name] == row["label"]) == (correct == name) for name in ("cnn", "tfidf"))
        text = row["clean_text"]
        cnn_prefix = cnn_score(text, cap, model, vocab, temperature["cnn"])
        tfidf_full = tfidf_score(text, pipeline, temperature["tfidf"])
        assert np.isclose(cnn_prefix["logit"], row["cnn_logit"], atol=2e-4, rtol=0)
        assert np.isclose(tfidf_full["logit"], row["tfidf_logit"], atol=1e-8, rtol=0)
        result = {"id": id_, "rotulo": int(row["label"]), "tokens_cnn": int(token_count[id_]),
                  "cnn_prefixo": cnn_prefix, "tfidf_inteiro": tfidf_full}
        if truncated[id_]:
            lower = text.lower()
            tokens = list(TOKEN_PATTERN.finditer(lower))
            boundary = tokens[cap - 1].end()
            if id_ == "imdb_test_03531":
                excerpts = {key: locate_excerpt(text, excerpt, tokens) for key, excerpt in EXCERPTS.items()}
                assert excerpts["prefixo"]["ultimo_token"] <= cap < excerpts["final"]["primeiro_token"]
                result["trechos"] = excerpts
            result["cnn_inteiro"] = cnn_score(text, int(token_count[id_]), model, vocab, temperature["cnn"])
            result["tfidf_prefixo"] = tfidf_score(lower[:boundary], pipeline, temperature["tfidf"])
        examples.append(result)

    return {
        "protocolo": "Pareamento pelo ID do teste oficial. Casos escolhidos para ilustrar recorte e resenha mista.",
        "teste_sha256": dataset_sha, "fontes_sha256": sources,
        "artefatos_sha256": {path.relative_to(artifacts).as_posix(): file_sha256(path)
                            for path in (cnn_path, vocab_path, tfidf_path)},
        "limite_cnn_tokens": cap, "temperatura": temperature, "total": counts(data),
        "sem_recorte": counts(data.loc[~truncated]), "com_recorte": counts(data.loc[truncated]),
        "casos": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True, help="Arquivo test.parquet original.")
    parser.add_argument("--artifacts", type=Path, required=True, help="Artefatos originais de CNN e TF-IDF.")
    parser.add_argument("--output", type=Path, default=HERE / "dados/discordancias.json")
    args = parser.parse_args()
    result = evaluate(args.test, args.artifacts)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for key in ("total", "sem_recorte", "com_recorte"):
        print(key, result[key])
    for case in result["casos"]:
        print(case["id"], {k: round(v["p_positiva"], 4) for k, v in case.items()
                           if isinstance(v, dict) and "p_positiva" in v})


if __name__ == "__main__":
    main()
