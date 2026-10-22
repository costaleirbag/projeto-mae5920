from __future__ import annotations

import re
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from .common import ROOT, config, device, release_memory, save_json
from .data import load_splits
from .metrics import comparison, evaluate, wilson

POSITIVE_WORDS = set("good great excellent wonderful amazing enjoyable brilliant love loved beautiful superb perfect best entertaining outstanding".split())
NEGATIVE_WORDS = set("bad terrible awful boring dull worst poor disappointing waste horrible ridiculous unwatchable hated".split())


def textual_features(texts):
    rows = []
    for text in texts:
        lower = text.lower()
        words = set(re.findall(r"[a-z]+", lower))
        rows.append({
            "negation": bool(re.search(r"\b(?:not|no|never|hardly|without)\b|n't\b", lower)),
            "contrast": bool(re.search(r"\b(?:but|however|although|yet|despite|nevertheless)\b", lower)),
            "mixed_polarity_lexicon": bool(words & POSITIVE_WORDS) and bool(words & NEGATIVE_WORDS),
            "explicit_rating": bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:/\s*10|out of (?:10|ten))\b", lower)),
            "irony_marker_heuristic": bool(re.search(r"\byeah right\b|\bas if\b|\bsarcas\w*\b|\bso bad (?:it|that)\b", lower)),
            "question": "?" in text,
        })
    return pd.DataFrame(rows)


def subgroup_metrics(rows, group, value):
    metrics = evaluate(rows.label, rows.p_positive)
    fpr_lo, fpr_hi = wilson(metrics["fp"], metrics["fp"] + metrics["tn"])
    fnr_lo, fnr_hi = wilson(metrics["fn"], metrics["fn"] + metrics["tp"])
    return {"group": group, "value": str(value), **metrics, "fpr_ci_low": fpr_lo, "fpr_ci_high": fpr_hi,
            "fnr_ci_low": fnr_lo, "fnr_ci_high": fnr_hi, "small_group": len(rows) < 50}


def cluster_names(texts, assignments, k):
    vectorizer = CountVectorizer(stop_words="english", max_features=15000, min_df=3, ngram_range=(1, 2))
    counts = vectorizer.fit_transform(texts)
    aggregates = np.vstack([np.asarray(counts[assignments == i].sum(axis=0)).ravel() for i in range(k)])
    tf = aggregates / np.maximum(aggregates.sum(axis=1, keepdims=True), 1)
    # Training-only contrastive labels: ubiquitous film/movie words should not
    # hide words that distinguish a group. TF * log-lift is a contribution to
    # divergence from the training corpus term distribution, not a human label.
    background = aggregates.sum(axis=0) / aggregates.sum()
    score = tf * np.log((tf + 1e-12) / (background[None, :] + 1e-12))
    terms = vectorizer.get_feature_names_out()
    return {str(i): terms[np.argsort(score[i])[-10:][::-1]].tolist() for i in range(k)}


def run():
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    cfg, splits = config(), load_splits()
    destination = ROOT / "results" / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    comparison()
    for name in ("tfidf", "cnn", "bert"):
        if not (ROOT / "results" / name / "predictions.parquet").exists():
            raise FileNotFoundError(f"Train and evaluate {name} before the combined analysis")
    start = time.perf_counter()
    # Independent sentence encoder. Clustering is fitted without test labels.
    source = cfg["analysis"]["encoder"]
    revision = cfg["analysis"].get("revision") or HfApi().model_info(source).sha
    encoder = SentenceTransformer(source, revision=revision, device=str(device()))
    training_sample = splits["train"].sample(n=min(cfg["analysis"]["train_cluster_sample"], len(splits["train"])), random_state=cfg["seed"])
    cache = destination / "embeddings.npz"
    if cache.exists():
        arrays = np.load(cache)
        if list(arrays["test_ids"]) != splits["test"].id.tolist() or list(arrays["train_ids"]) != training_sample.id.tolist():
            raise ValueError("Embedding cache does not match current data")
        train_embedding, test_embedding = arrays["train"], arrays["test"]
    else:
        train_embedding = encoder.encode(training_sample.clean_text.tolist(), batch_size=cfg["analysis"]["batch_size"],
                                         normalize_embeddings=True, show_progress_bar=True)
        test_embedding = encoder.encode(splits["test"].clean_text.tolist(), batch_size=cfg["analysis"]["batch_size"],
                                        normalize_embeddings=True, show_progress_bar=True)
        np.savez_compressed(cache, train=train_embedding, test=test_embedding,
                            train_ids=training_sample.id.to_numpy(dtype=str), test_ids=splits["test"].id.to_numpy(dtype=str))
    encoder.to("cpu")
    del encoder
    release_memory()
    clusterer = KMeans(n_clusters=cfg["analysis"]["n_clusters"], n_init=10, random_state=cfg["seed"])
    train_clusters = clusterer.fit_predict(train_embedding)
    test_clusters = clusterer.predict(test_embedding)
    joblib.dump(clusterer, ROOT / "artifacts" / "semantic_clusterer.joblib")
    terms = cluster_names(training_sample.clean_text.tolist(), train_clusters, cfg["analysis"]["n_clusters"])
    save_json(destination / "cluster_terms.json", terms)
    test = splits["test"].copy()
    test["semantic_cluster"] = test_clusters
    test["length_group"] = pd.cut(test.word_count, bins=[-1, 100, 250, 500, np.inf],
                                  labels=["<=100", "101-250", "251-500", ">500"])
    markers = textual_features(test.clean_text)
    for col in markers:
        test[col] = markers[col].to_numpy()
    test[["id", "semantic_cluster", "length_group", *markers.columns]].to_parquet(destination / "test_segments.parquet", index=False)
    subgroup_rows, examples, outcomes, token_contrasts = [], [], {}, []
    for model in ("tfidf", "cnn", "bert"):
        pred = pd.read_parquet(ROOT / "results" / model / "predictions.parquet")
        pred = pred.loc[pred.split == "test"].drop(columns=["word_count", "label"])
        rows = test.merge(pred, on="id", validate="one_to_one")
        if len(rows) != len(test):
            raise ValueError("Incomplete model predictions")
        outcomes[model] = rows.set_index("id").outcome
        for grouping in ("length_group", "semantic_cluster", *markers.columns):
            for value, part in rows.groupby(grouping, observed=True):
                subgroup_rows.append({"model": model, **subgroup_metrics(part, grouping, value)})
        rows["error_confidence"] = np.where(rows.label == 0, rows.p_positive, 1 - rows.p_positive)
        for outcome in ("FP", "FN"):
            part = rows.loc[rows.outcome == outcome].sort_values("error_confidence", ascending=False)
            selected = part.head(3)
            if len(part) > 3:
                selected = pd.concat([selected, part.iloc[3:].sample(min(3, len(part) - 3), random_state=cfg["seed"])])
            for _, row in selected.iterrows():
                examples.append({"model": model, "id": row.id, "outcome": outcome, "label": int(row.label),
                                 "p_positive": float(row.p_positive), "word_count": int(row.word_count),
                                 "semantic_cluster": int(row.semantic_cluster), "text": row.clean_text,
                                 "markers": {marker: bool(row[marker]) for marker in markers}})
        # Compare errors against correct cases of the SAME true class.
        for error, control, true_label in (("FP", "TN", 0), ("FN", "TP", 1)):
            relevant = rows.loc[rows.label == true_label]
            error_rows, control_rows = relevant.loc[relevant.outcome == error], relevant.loc[relevant.outcome == control]
            for marker in markers:
                p_error, p_control = float(error_rows[marker].mean()), float(control_rows[marker].mean())
                token_contrasts.append({"model": model, "error": error, "control": control, "marker": marker,
                                        "n_error": len(error_rows), "n_control": len(control_rows),
                                        "frequency_error": p_error, "frequency_control": p_control,
                                        "difference": p_error - p_control,
                                        "interpretation": "Exploratory association; heuristic marker, not a causal explanation."})
    pd.DataFrame(subgroup_rows).to_csv(destination / "subgroup_metrics.csv", index=False)
    pd.DataFrame(token_contrasts).to_csv(destination / "marker_contrasts.csv", index=False)
    save_json(destination / "error_examples.json", examples)
    pair_rows = []
    for a, b in (("tfidf", "cnn"), ("tfidf", "bert"), ("cnn", "bert")):
        aa = outcomes[a].isin(["FP", "FN"])
        bb = outcomes[b].reindex(aa.index).isin(["FP", "FN"])
        both = int((aa & bb).sum())
        only_a, only_b = int((aa & ~bb).sum()), int((~aa & bb).sum())
        pair_rows.append({"model_a": a, "model_b": b, "both_wrong": both, "only_a_wrong": only_a,
                          "only_b_wrong": only_b, "both_correct": int((~aa & ~bb).sum()),
                          "mcnemar_exact_p": float(binomtest(only_a, only_a + only_b, .5).pvalue) if only_a + only_b else 1.,
                          "caveat": "Unadjusted exploratory paired test; retained test duplicates violate strict independence."})
    pd.DataFrame(pair_rows).to_csv(destination / "error_overlap.csv", index=False)
    save_json(destination / "metadata.json", {"seconds": time.perf_counter() - start,
              "encoder": source, "revision": revision, "encoder_max_length": 256,
              "cluster_fit": "Unlabeled training sample only", "cluster_fit_n": len(training_sample),
              "cluster_terms": "Training-only contrastive TF times log-lift terms; themes require human interpretation",
              "segment_evaluation": "All official test reviews; no oversampling of errors",
              "caveats": ["Exploratory associations, not established causes of errors.",
                           "Lexical markers are noisy and overlapping, especially irony.",
                           "Sentence embeddings see a token-limited prefix of each review.",
                           "Interpret comparisons conditional on true label (FP vs TN; FN vs TP).",
                           "Small subgroup estimates are uncertain; Wilson intervals supplied."]})
    print("Semantic analysis complete", flush=True)
