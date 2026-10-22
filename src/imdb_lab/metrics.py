from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, roc_auc_score)

from .common import ROOT, save_json


def fit_temperature(logits, labels):
    logits, labels = np.asarray(logits), np.asarray(labels)
    def objective(log_t):
        z = logits / np.exp(log_t)
        return float(np.mean(np.logaddexp(0, z) - labels * z))
    result = minimize_scalar(objective, bounds=(np.log(.05), np.log(10)), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature calibration failed")
    return float(np.exp(result.x))


def calibration_bins(labels, probs, n_bins=10):
    labels, probs = np.asarray(labels), np.asarray(probs)
    assignment = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    rows = []
    for i in range(n_bins):
        select = assignment == i
        if select.any():
            rows.append({"bin": i, "n": int(select.sum()), "mean_p_positive": float(probs[select].mean()),
                         "fraction_positive": float(labels[select].mean())})
    return rows


def evaluate(labels, probs):
    labels = np.asarray(labels, dtype=int)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-7, 1 - 1e-7)
    predictions = (probs >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    bins = calibration_bins(labels, probs)
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
            "f1_macro": float(f1_score(labels, predictions, average="macro", labels=[0, 1], zero_division=0)),
            "log_loss": float(log_loss(labels, probs, labels=[0, 1])),
            "brier": float(brier_score_loss(labels, probs)),
            "roc_auc": float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else None,
            "ece_10": float(sum(b["n"] / len(labels) * abs(b["mean_p_positive"] - b["fraction_positive"]) for b in bins)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None}


def wilson(successes, total, z=1.96):
    if not total:
        return (None, None)
    p = successes / total
    denom = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / denom
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (float(max(0, middle - radius)), float(min(1, middle + radius)))


def save_predictions(model_name, splits, logits_by_split, metadata):
    destination = ROOT / "results" / model_name
    destination.mkdir(parents=True, exist_ok=True)
    temperature = fit_temperature(logits_by_split["calibration"], splits["calibration"].label)
    metadata["temperature"] = temperature
    metadata["calibration_fit_split"] = "calibration"
    metadata["threshold"] = .5
    all_rows, metrics = [], {}
    for name, frame in splits.items():
        logits = np.asarray(logits_by_split[name], dtype=float)
        if logits.shape != (len(frame),) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid predictions for {model_name}/{name}")
        rows = frame[["id", "label", "word_count"]].copy()
        rows["split"], rows["model"] = name, model_name
        rows["logit"] = logits
        rows["p_raw"] = expit(logits)
        rows["p_positive"] = expit(logits / temperature)
        rows["prediction"] = (logits >= 0).astype(int)
        rows["outcome"] = np.select([(rows.label == 0) & (rows.prediction == 0),
                                     (rows.label == 0) & (rows.prediction == 1),
                                     (rows.label == 1) & (rows.prediction == 0)], ["TN", "FP", "FN"], default="TP")
        all_rows.append(rows)
        metrics[name] = {"raw": evaluate(rows.label, rows.p_raw), "calibrated": evaluate(rows.label, rows.p_positive),
                         "reliability_raw": calibration_bins(rows.label, rows.p_raw),
                         "reliability_calibrated": calibration_bins(rows.label, rows.p_positive)}
    pd.concat(all_rows, ignore_index=True).to_parquet(destination / "predictions.parquet", index=False)
    save_json(destination / "metrics.json", metrics)
    save_json(destination / "metadata.json", metadata)
    print(model_name, "test", metrics["test"]["calibrated"], flush=True)


def comparison():
    from .common import read_json
    rows = []
    for p in sorted((ROOT / "results").glob("*/metrics.json")):
        if p.parent.name not in {"tfidf", "cnn", "bert"}:
            continue
        met = read_json(p)
        for split, views in met.items():
            for view in ("raw", "calibrated"):
                rows.append({"model": p.parent.name, "split": split, "probabilities": view, **views[view]})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results" / "comparison.csv", index=False)
    return result
