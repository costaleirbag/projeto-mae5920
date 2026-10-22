import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .common import ROOT, config, environment
from .data import load_splits
from .metrics import save_predictions


def run():
    cfg, splits = config(), load_splits()
    start = time.perf_counter()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.99,
                               max_features=cfg["baseline"]["max_features"], sublinear_tf=True)
    # Negations retained; the vectorizer is fitted on training text only.
    matrix = {"train": vectorizer.fit_transform(splits["train"].clean_text)}
    for name in ("validation", "calibration", "test"):
        matrix[name] = vectorizer.transform(splits[name].clean_text)
    featurize_seconds = time.perf_counter() - start
    best, best_f1, search = None, -1, []
    for c in cfg["baseline"]["c_grid"]:
        model = LogisticRegression(C=c, solver="liblinear", max_iter=1000, random_state=cfg["seed"])
        model.fit(matrix["train"], splits["train"].label)
        score = f1_score(splits["validation"].label, model.predict(matrix["validation"]), average="macro")
        search.append({"C": c, "validation_f1_macro": float(score)})
        print("TF-IDF", search[-1], flush=True)
        if score > best_f1:
            best, best_f1 = model, score
    training_seconds = time.perf_counter() - start
    destination = ROOT / "artifacts" / "tfidf"
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": best}, destination / "model.joblib")
    inference_start = time.perf_counter()
    logits = {name: best.decision_function(mat) for name, mat in matrix.items()}
    metadata = {"environment": environment(), "config": cfg["baseline"], "search": search,
                "selected_C": best.C, "feature_count": len(vectorizer.vocabulary_),
                "trainable_parameters": int(best.coef_.size + best.intercept_.size),
                "training_seconds": training_seconds, "featurization_seconds": featurize_seconds,
                "all_splits_inference_seconds": time.perf_counter() - inference_start,
                "selection_split": "validation", "input_scope": "Full cleaned review"}
    save_predictions("tfidf", splits, logits, metadata)
