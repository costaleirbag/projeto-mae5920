"""Mede o efeito de bigramas e do recorte no baseline original."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from imdb_lab.common import ROOT, file_sha256, read_json, save_json
from imdb_lab.metrics import evaluate, fit_temperature
from imdb_lab.models import TOKEN_PATTERN, tokenize


SCOPES = {"256": 256, "400": 400, "768": 768, "integral": None}
NGRAMS = {"unigramas": (1, 1), "unigramas_bigramas": (1, 2)}


def cut(text, limit):
    if limit is None:
        return text
    normalized = text.lower()
    for i, match in enumerate(TOKEN_PATTERN.finditer(normalized)):
        if i == limit:
            return normalized[:match.start()]
    return text


def load(folder, role):
    entry = read_json(ROOT / "data/manifest.json")["splits"][role]
    path = folder / f"{role}.parquet"
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"Hash inesperado da divisão {role}.")
    return pd.read_parquet(path, columns=["id", "label", "clean_text", "text_sha256"])


def scores(labels, margin, temperature=1):
    stats = evaluate(labels, expit(margin / temperature))
    return {key: stats[key] for key in ("n", "accuracy", "f1_macro", "log_loss")}


def make_logistic(c, seed):
    return LogisticRegression(C=c, solver="liblinear", max_iter=1000, random_state=seed)


def train(folder):
    cfg = read_json(ROOT / "config.json")
    baseline = cfg["baseline"]
    train = load(folder, "train")
    validation = load(folder, "validation")
    fits = {}
    for scope, limit in SCOPES.items():
        tx = train.clean_text.map(lambda text: cut(text, limit))
        vx = validation.clean_text.map(lambda text: cut(text, limit))
        for name, gram_range in NGRAMS.items():
            vectorizer = TfidfVectorizer(ngram_range=gram_range, min_df=2, max_df=.99,
                                         max_features=baseline["max_features"], sublinear_tf=True)
            xt = vectorizer.fit_transform(tx)
            xv = vectorizer.transform(vx)
            candidates, records = [], []
            for c in baseline["c_grid"]:
                model = make_logistic(c, cfg["seed"]).fit(xt, train.label)
                train_margin = model.decision_function(xt)
                val_margin = model.decision_function(xv)
                validation_metrics = scores(validation.label, val_margin)
                if model.n_iter_.max() >= model.max_iter:
                    raise RuntimeError("Regressão atingiu o limite de iterações.")
                records.append({"C": c, "train": scores(train.label, train_margin),
                                "validation": validation_metrics, "iterations": int(model.n_iter_.max())})
                candidates.append(model)
            choice = max(range(len(records)), key=lambda i: records[i]["validation"]["f1_macro"])
            model = candidates[choice]
            stats = {
                "features": int(xt.shape[1]), "nonzero_median": float(np.median(xt.getnnz(axis=1))),
                "selected_C": float(model.C), "candidates": records,
            }
            if scope == "integral" and name == "unigramas_bigramas":
                shuffled = np.random.default_rng(cfg["seed"]).permutation(train.label.to_numpy())
                control = make_logistic(model.C, cfg["seed"]).fit(xt, shuffled)
                stats["shuffled_labels"] = {
                    "seed": cfg["seed"], "C": float(control.C),
                    "train_against_shuffled_labels": scores(shuffled, control.decision_function(xt)),
                    "validation_against_real_labels": scores(validation.label, control.decision_function(xv)),
                }
            fits[scope, name] = (vectorizer, model, stats)
            print("AJUSTE", scope, name, "C", model.C, "features", xt.shape[1], flush=True)
    return fits, train, validation


def compare_saved(test):
    paths = {
        "tfidf": "results/tfidf/predictions.parquet",
        "cnn": "results/cnn/predictions.parquet",
        "bert": "results/bert/predictions.parquet",
        "qwen": "results/qwen/classification/predictions.parquet",
    }
    right = {}
    for model, path in paths.items():
        rows = pd.read_parquet(ROOT / path, columns=["id", "split", "label", "prediction"])
        rows = rows.loc[rows.split.eq("test")].set_index("id", verify_integrity=True)
        assert len(rows) == len(test)
        rows = rows.loc[test.id]
        np.testing.assert_array_equal(rows.label, test.label)
        right[model] = rows.prediction.to_numpy() == test.label.to_numpy()
    qwen = right["qwen"]
    return {
        "predictions_sha256": {key: file_sha256(ROOT / path) for key, path in paths.items()},
        "qwen": {name: {"only_qwen": int((qwen & ~correct).sum()),
                        "only_comparator": int((~qwen & correct).sum()),
                        "both_right": int((qwen & correct).sum()),
                        "both_wrong": int((~qwen & ~correct).sum())}
                 for name, correct in right.items() if name != "qwen"},
    }, right["tfidf"]


def audit(folder, fits, train, validation, output):
    calibration, test = (load(folder, role) for role in ("calibration", "test"))
    for i, left in enumerate((train, validation, calibration, test)):
        for right in (train, validation, calibration, test)[i + 1:]:
            assert set(left.text_sha256).isdisjoint(right.text_sha256)
    paired, tfidf_right = compare_saved(test)
    y = test.label.to_numpy()
    long = test.clean_text.map(lambda text: len(tokenize(text)) > 400).to_numpy()
    assert int(long.sum()) == read_json(ROOT / "seminario/dados/recorte_teste/resumo.json")["resenhas_recortadas"]["n"]
    rows = []
    margins = {"test_ids": test.id.to_numpy(dtype=str), "test_labels": y, "above_400": long,
               "calibration_labels": calibration.label.to_numpy()}
    for (scope, name), (vectorizer, model, stats) in fits.items():
        limit = SCOPES[scope]
        cal_x = vectorizer.transform(calibration.clean_text.map(lambda text: cut(text, limit)))
        test_x = vectorizer.transform(test.clean_text.map(lambda text: cut(text, limit)))
        cal_z, test_z = model.decision_function(cal_x), model.decision_function(test_x)
        tau = fit_temperature(cal_z, calibration.label)
        margins[f"test_{scope}_{name}"] = test_z
        margins[f"calibration_{scope}_{name}"] = cal_z
        rows.append({"scope": scope, "ngrams": name, **stats, "temperature": tau,
                     "test": scores(y, test_z, tau), "test_long": scores(y[long], test_z[long], tau)})
        if scope == "integral" and name == "unigramas_bigramas":
            stats["different_correctness_from_published"] = int((((test_z >= 0) == y) != tfidf_right).sum())
            rows[-1].update(stats)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "predictions.npz", **margins)
    save_json(output / "summary.json", {
        "scikit_learn": sklearn.__version__, "recipes": rows, "official_overlap": paired,
        "cross_role_equal_text_hashes": 0,
        "predictions_sha256": file_sha256(output / "predictions.npz"),
        "source_sha256": file_sha256(Path(__file__)),
        "data_manifest_sha256": file_sha256(ROOT / "data/manifest.json"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output deve apontar para um diretório novo.")
    audit(args.data, *train(args.data), args.output)


if __name__ == "__main__":
    main()
