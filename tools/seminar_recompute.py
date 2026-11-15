"""Refaz diagnósticos do seminário com checkpoints de uma nova execução."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "seminario")]

from imdb_lab import sequences
from imdb_lab.bpe_study import metrics
from imdb_lab.common import file_sha256, read_json, save_json
from imdb_lab.data import clean_text, fingerprint
from imdb_lab.metrics import evaluate, fit_temperature
from imdb_lab.models import EncodedReviews, PadBatch, TextCNN, TOKEN_PATTERN, Vocabulary
from imdb_lab.training import predict


REFERENCE = ROOT / "referencia"
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "seminar_retrain/diagnosticos"
DATA = ROOT / "data"


def original_predictions(test):
    """Alinha as predições novas pelo ID e confirma os rótulos."""
    predictions = {}
    for name, path in (
        ("tfidf", RESULTS / "tfidf/predictions.parquet"),
        ("cnn", RESULTS / "cnn/predictions.parquet"),
        ("bert", RESULTS / "bert/predictions.parquet"),
        ("qwen", RESULTS / "qwen/classification/predictions.parquet"),
    ):
        if not path.is_file() and name in {"bert", "qwen"}:
            continue
        frame = pd.read_parquet(path).query("split == 'test'").set_index("id", verify_integrity=True)
        if set(frame.index) != set(test.index):
            raise ValueError(f"IDs do teste diferentes nas predições de {name}.")
        rows = frame.loc[test.index]
        np.testing.assert_array_equal(rows.label, test.label)
        predictions[name] = rows
    return predictions


def tfidf_controls(template, cap):
    """Repete os dois tetos com unigramas e bigramas e o controle de rótulos."""
    from auditar_baseline import cut, load, make_logistic, scores

    cfg = read_json(ROOT / "config.json")
    train, val = (load(DATA, role) for role in ("train", "validation"))
    recipes = {}
    for limit in (cap, None):
        vectorizer = clone(template)
        x = vectorizer.fit_transform(train.clean_text.map(lambda text: cut(text, limit)))
        xv = vectorizer.transform(val.clean_text.map(lambda text: cut(text, limit)))
        candidates = [make_logistic(c, cfg["seed"]).fit(x, train.label) for c in cfg["baseline"]["c_grid"]]
        validation = [scores(val.label, model.decision_function(xv)) for model in candidates]
        choice = max(range(len(validation)), key=lambda i: validation[i]["f1_macro"])
        model = candidates[choice]
        row = {"C": model.C, "train": scores(train.label, model.decision_function(x)),
               "validation": validation[choice]}
        if limit is None:
            shuffled = np.random.default_rng(cfg["seed"]).permutation(train.label.to_numpy())
            control = make_logistic(model.C, cfg["seed"]).fit(x, shuffled)
            row["rotulos_embaralhados"] = {"train": scores(shuffled, control.decision_function(x)),
                                          "validation": scores(val.label, control.decision_function(xv))}
        recipes[str(limit) if limit is not None else "integral"] = (limit, vectorizer, model, row)

    cal, test = (load(DATA, role) for role in ("calibration", "test"))
    result = {}
    for key, (limit, vectorizer, model, row) in recipes.items():
        cal_z = model.decision_function(vectorizer.transform(cal.clean_text.map(lambda text: cut(text, limit))))
        z = model.decision_function(vectorizer.transform(test.clean_text.map(lambda text: cut(text, limit))))
        tau = fit_temperature(cal_z, cal.label)
        result[key] = row | {"test": scores(test.label, z, tau), "temperatura": tau}
    return result


def tfidf_example(pipeline, n_train):
    """A mesma decomposição da margem na frase ilustrativa do slide 7."""
    from medir_tfidf_slide import TEXT

    vectorizer, model = (pipeline[key] for key in ("vectorizer", "model"))
    vector = vectorizer.transform([TEXT])
    counts = Counter(vectorizer.build_analyzer()(TEXT))
    terms = []
    for term, count in counts.items():
        index = vectorizer.vocabulary_[term]
        idf = float(vectorizer.idf_[index])
        term_frequency = float(1 + np.log(count))
        normalized = float(vector[0, index])
        weight = float(model.coef_[0, index])
        terms.append({"term": term, "count": count, "tf": term_frequency, "idf": idf,
                      "document_frequency": round((n_train + 1) / np.exp(idf - 1) - 1),
                      "tfidf": normalized, "coefficient": weight, "contribution": normalized * weight})
    norm = float(np.linalg.norm([term["tf"] * term["idf"] for term in terms]))
    np.testing.assert_allclose([term["tf"] * term["idf"] / norm for term in terms],
                               [term["tfidf"] for term in terms])
    bias = float(model.intercept_[0])
    logit = float(model.decision_function(vector)[0])
    np.testing.assert_allclose(bias + sum(term["contribution"] for term in terms), logit)
    return {"text": TEXT, "train_documents": n_train, "normalization": norm, "terms": terms,
            "bias": bias, "logit": logit, "positive_raw": float(expit(logit))}


def cnn_on_full_text(test, saved, model, vocab, token_counts, cap, tau):
    from medir_recorte import scores

    clipped = np.flatnonzero(token_counts > cap)
    indices = clipped[np.argsort(token_counts[clipped], kind="stable")]
    y = test.label.to_numpy()
    full = [vocab.encode(text, int(count))
            for text, count in zip(test.clean_text.iloc[indices], token_counts[indices], strict=True)]
    margins = saved.logit.to_numpy(dtype=float)
    replay, evaluated_full = (
        predict(model, DataLoader(EncodedReviews(ids, y[indices]), batch_size=32, collate_fn=PadBatch("cnn")),
                torch.device("cpu"))
        for ids in ([sequence[:cap] for sequence in full], full)
    )
    np.testing.assert_allclose(replay, margins[indices], atol=2e-4, rtol=0)
    new = margins.copy()
    new[indices] = evaluated_full
    return expit(new / tau), {"cap": cap, "maior_resenha_em_tokens": int(token_counts.max()),
                              "teste": scores(y, margins, new, tau),
                              "resenhas_recortadas": scores(y[indices], margins[indices], new[indices], tau)}


def original():
    from comparar_erros import EXAMPLES, cnn_score, counts, tfidf_score
    from medir_comprimentos import lengths, load_data, metrics_by_length
    from medir_pooling_regularizacao import baseline

    manifest = read_json(DATA / "manifest.json")
    train, test = (load_data(DATA, manifest, role) for role in ("train", "test"))
    _, train_stats = lengths(train)
    token_counts, test_stats = lengths(test)
    predictions = original_predictions(test)
    cfg = read_json(ROOT / "config.json")["cnn"]
    cap = cfg["max_length"]
    artifacts = ROOT / "artifacts"
    pipeline = joblib.load(artifacts / "tfidf/model.joblib")  # produzido pelo treino desta execução
    vocab = Vocabulary(read_json(artifacts / "cnn/vocabulary.json"))
    model = TextCNN(len(vocab.token_to_id), cfg["embedding_dim"], cfg["filters"], cfg["kernels"], cfg["dropout"])
    model.load_state_dict(torch.load(artifacts / "cnn/model.pt", map_location="cpu", weights_only=True))
    model.eval()
    tau = {name: read_json(RESULTS / name / "metadata.json")["temperature"] for name in ("cnn", "tfidf")}

    recomputed, recut = cnn_on_full_text(test, predictions["cnn"], model, vocab, token_counts, cap, tau["cnn"])
    calibrated = {name: rows.p_positive.to_numpy() for name, rows in predictions.items()}
    calibrated["cnn_integral"] = recomputed
    examples = []
    for case in EXAMPLES:
        row = test.loc[case]
        text = row.clean_text
        spans = list(TOKEN_PATTERN.finditer(text.lower()))
        prefix = text.lower()[:spans[cap - 1].end()] if len(spans) > cap else text
        examples.append({"id": case, "rotulo": int(row.label), "tokens_cnn": len(spans),
                         "cnn_prefixo": cnn_score(text, cap, model, vocab, tau["cnn"]),
                         "cnn_inteiro": cnn_score(text, len(spans), model, vocab, tau["cnn"]),
                         "tfidf_prefixo": tfidf_score(prefix, pipeline, tau["tfidf"]),
                         "tfidf_inteiro": tfidf_score(text, pipeline, tau["tfidf"])})
    paired = test.assign(**{name: predictions[name].prediction for name in ("tfidf", "cnn")})
    save_json(OUTPUT / "original.json", {
        "comprimentos": {"treino": train_stats, "teste": test_stats},
        "faixas": metrics_by_length(test, calibrated, token_counts, cap),
        "recorte_cnn": recut,
        "casos": examples,
        "acertos_pareados": counts(paired),
        "tfidf_frase": tfidf_example(pipeline, len(train)),
        "tfidf_controles": tfidf_controls(pipeline["vectorizer"], cap),
    })
    baseline(DATA, OUTPUT / "regularizacao_tfidf")


def paired_models():
    from demo import FrozenCNN

    selection = read_json(RESULTS / "seminar_retrain/textcnn/selection.json")
    cfg = read_json(ROOT / "bpe_config.json")
    tokens = {
        "word": Vocabulary(read_json(RESULTS / "sequences/vocabulary.json")),
        "bpe": Tokenizer.from_file(str(RESULTS / "bpe/tokenizer.json")),
    }
    expected = {(kind, seed) for kind in tokens for seed in cfg["seeds"]}
    observed = {(record["kind"], record["seed"]) for record in selection["runs"]}
    if observed != expected or len(selection["runs"]) != len(expected):
        raise ValueError("A avaliação requer as duas CNNs por semente já selecionadas.")
    models = []
    for record in selection["runs"]:
        kind, seed = record["kind"], record["seed"]
        checkpoint = (ROOT / record["path"]).resolve()
        if not checkpoint.is_relative_to(ROOT) or file_sha256(checkpoint) != record["sha256"]:
            raise ValueError(f"Checkpoint não corresponde à seleção de {kind}, semente {seed}.")
        tokenizer = tokens[kind]
        params = cfg["cnn"]
        size = len(tokenizer.token_to_id) if kind == "word" else tokenizer.get_vocab_size()
        model = TextCNN(size, params["embedding_dim"], params["filters"], params["kernels"], params["dropout"])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        models.append(FrozenCNN({"word": "Palavras", "bpe": "BPE"}[kind], seed, record["best_epoch"],
                                record["sha256"], model, tokenizer, cfg["whitespace_words"], cfg["token_cap"]))
    return models


def paired_cases(models, pipeline):
    from comparar_erros import tfidf_score
    from medir_animacoes import bpe_steps, cnn_responses
    from medir_pares import windows

    cases = read_json(ROOT / "seminario/casos.json")
    contrast = next(pair for pair in cases["pares"] if pair["id"] == "contraste")
    word = next(model for model in models if model.label == "Palavras" and model.seed == 17)
    bpe = next(model for model in models if model.label == "BPE" and model.seed == 17)
    pairs = []
    for model in models:
        a, b = (model.predict(contrast[side]) for side in ("a", "b"))
        pairs.append({"modelo": model.label, "semente": model.seed, "a": a, "b": b,
                      "janelas_iguais": {width: windows(a["ids"], width) == windows(b["ids"], width)
                                        for width in model.model.kernels}})
    published = read_json(REFERENCE / "seminario/dados/animacoes.json")["cnn"]["casos"]["A"]["janelas"]
    trace = {}
    for side in ("a", "b"):
        prediction, responses = cnn_responses(word, contrast[side])
        trace[side] = {"tokens": prediction["tokens"], "janelas": {}}
        for width, channel in responses.items():
            index = published[str(width)]["filtro"]
            values = channel[index]
            trace[side]["janelas"][str(width)] = {"filtro": index, "pre_relu": values.tolist(),
                                                  "ativacoes": values.relu().tolist()}
    return {"frases": contrast, "cnn": pairs, "ativacoes": trace,
            "tfidf": {side: tfidf_score(contrast[side], pipeline, 1) for side in ("a", "b")},
            "bpe": bpe_steps(bpe.tokenizer, "pallbearer")}


def stride_control(models, frame, cfg):
    from medir_passo import predict_strides

    frame = frame.copy()
    frame["text"] = frame.clean_text.map(lambda text: sequences.excerpt(text, cfg["whitespace_words"]))
    inputs, _ = sequences.cnn_inputs(frame, torch.device("cpu"))
    labels = frame.label.to_numpy(dtype=np.int64)
    runs = []
    for model in models:
        if model.label == "Palavras":
            logits = predict_strides(model.model, inputs, len(labels))
            runs.extend({"semente": model.seed, "passo": stride, **metrics(values, labels),
                         "mudancas_vs_passo_1": int((values.argmax(1) != logits[1].argmax(1)).sum())}
                        for stride, values in logits.items())
    return runs


def outside_imdb(models, pipeline, source, unseen_against):
    from comparar_erros import tfidf_score
    from medir_fora_dominio import FILES, youtube_comments

    domains = {}
    with ZipFile(source) as archive:
        for domain, name in FILES.items():
            lines = archive.read(f"sentiment labelled sentences/{name}").decode("utf-8").splitlines()
            records = [line.rsplit("\t", 1) for line in lines]
            if Counter(int(label) for _, label in records) != {0: 500, 1: 500}:
                raise ValueError(f"Rótulos inesperados na fonte {domain}.")
            records = [(review, int(label)) for review, label in records if fingerprint(review) not in unseen_against]
            texts, labels = zip(*records, strict=True)
            runs = []
            for model in models:
                logits = np.array([model.predict(text)["logits"] for text in texts], dtype=np.float32)
                runs.append({"modelo": model.label, "semente": model.seed, **metrics(logits, labels)})
            averages = {}
            for kind in ("Palavras", "BPE"):
                group = [run for run in runs if run["modelo"] == kind]
                averages[kind] = {key: mean(run[key] for run in group) for key in ("accuracy", "f1_macro", "ce")}
            domains[domain] = {"n": len(labels), "positivos": sum(labels),
                               "mediana_palavras": median(len(clean_text(text).split()) for text in texts),
                               "medias": averages, "execucoes": runs}
    examples = [case for case in youtube_comments(models) if case["id"] in {"E", "F"}]
    for case in examples:
        case["tfidf"] = tfidf_score(clean_text(case["texto"]), pipeline, 1)
    return {"dominios_rotulados": domains, "comentarios_youtube": examples}


def textcnn(dataset):
    from medir_fora_dominio import DATASET_URL

    models = paired_models()
    pipeline = joblib.load(ROOT / "artifacts/tfidf/model.joblib")
    manifest = read_json(REFERENCE / "results/bpe/data_manifest.json")["roles"]
    cfg = read_json(ROOT / "bpe_config.json")
    frame = pd.read_parquet(DATA / "train.parquet").set_index("id", verify_integrity=True)
    audit = frame.loc[manifest["audit"]["ids"]]
    if audit.text_sha256.tolist() != manifest["audit"]["text_sha256"]:
        raise ValueError("Os textos ordenados da auditoria diferem da referência.")
    fit_ids = manifest["train"]["ids"] + manifest["validation"]["ids"]
    seen = set(frame.loc[fit_ids].text_sha256)
    reference = read_json(REFERENCE / "seminario/dados/fora_dominio_resultados.json")
    expected = reference["dataset"]["zip_sha256"]
    save_json(OUTPUT / "pares_e_ativacoes.json", paired_cases(models, pipeline))
    save_json(OUTPUT / "passo.json", stride_control(models, audit, cfg))
    with TemporaryDirectory(prefix="uci-") as directory:
        source = dataset or Path(directory) / "sentiment-labelled.zip"
        if dataset is None:
            urlretrieve(DATASET_URL, source)
        if file_sha256(source) != expected:
            raise ValueError("ZIP da UCI diferente da versão usada no relatório.")
        save_json(OUTPUT / "fora_dominio.json", outside_imdb(models, pipeline, source, seen))


def bert_cap(device):
    from transformers import AutoTokenizer, BertForSequenceClassification
    from medir_comprimentos import LENGTH_BANDS, load_data

    path = ROOT / "artifacts/bert"
    target = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(path, local_files_only=True,
                                                          attn_implementation="eager").to(target).eval()
    meta = read_json(RESULTS / "bert/metadata.json")
    caps = (meta["config"]["max_length"], model.config.max_position_embeddings)
    n_special = tokenizer.num_special_tokens_to_add(pair=False)
    test = load_data(DATA, read_json(DATA / "manifest.json"), "test")
    words = test.word_count.to_numpy()
    texts = test.clean_text.tolist()
    sizes = []
    for start in range(0, len(texts), 256):
        sizes.extend(tokenizer(texts[start:start + 256], add_special_tokens=False, truncation=False,
                               verbose=False, return_length=True)["length"])
    sizes = np.asarray(sizes)
    bands = []
    for low, high in [*LENGTH_BANDS, (0, None)]:
        group = (words >= low) & (True if high is None else words <= high)
        values = sizes[group]
        bands.append({"min_words": low, "max_words": high, "n": int(group.sum()),
                      "truncated": {cap: int((values > cap - n_special).sum()) for cap in caps},
                      "retained": {cap: float(np.minimum(values, cap - n_special).sum() / values.sum())
                                   for cap in caps}})
    selected = test.loc[words > 500]
    y = selected.label.to_numpy()
    margins = {}
    for cap in caps:
        encoded = tokenizer(selected.clean_text.tolist(), truncation=True, max_length=cap, padding=False)
        loader = DataLoader(EncodedReviews(encoded["input_ids"], y), batch_size=16,
                            collate_fn=PadBatch("bert", tokenizer.pad_token_id))
        margins[cap] = predict(model, loader, target)
    saved = pd.read_parquet(RESULTS / "bert/predictions.parquet").query("split == 'test'").set_index("id")
    saved = saved.loc[selected.index]
    np.testing.assert_array_equal(y, saved.label)
    replay_error = np.max(np.abs(margins[caps[0]] - saved.logit.to_numpy()))
    np.testing.assert_allclose(margins[caps[0]], saved.logit, atol=.01, rtol=0)
    scores = {cap: evaluate(y, expit(values.astype(np.float64) / meta["temperature"]))
              for cap, values in margins.items()}
    save_json(OUTPUT / "recorte_bert.json", {"caps_including_specials": list(caps), "coverage": bands,
                                           "scores": scores, "max_replay_error": float(replay_error)})


def qwen_youtube():
    from medir_qwen_youtube import evaluate as measure

    source = REFERENCE / "seminario/dados/fora_dominio_resultados.json"
    save_json(OUTPUT / "qwen_youtube.json", measure(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("original", "textcnn", "bert-cap", "qwen-youtube"))
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--uci-zip", type=Path, help="Cópia local do ZIP da UCI, verificada por SHA-256.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    stages = {"original": original, "textcnn": lambda: textcnn(args.uci_zip),
              "bert-cap": lambda: bert_cap(args.device), "qwen-youtube": qwen_youtube}
    stages[args.stage]()
    print(f"Diagnósticos da nova execução: {OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
