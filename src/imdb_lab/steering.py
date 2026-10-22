from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch

from .common import ROOT, config, device, file_sha256, read_json, release_memory, save_json
from .data import load_splits, stratified_sample
from .rfm import ActivationSteering, LaplacianRFM, bert_tail_logits
from .training import load_model

SYNTHETIC_PROBES = [
    ("neutral_1", "neutral_probe", "The film follows a family moving to a new town. It lasts about two hours."),
    ("neutral_2", "neutral_probe", "I watched the movie yesterday. The story takes place in a city and has several characters."),
    ("neutral_3", "neutral_probe", "The movie was ordinary and the acting was average."),
    ("negative_1", "negative_probe", "The acting was terrible and the story was boring. I regret watching this movie."),
    ("negative_2", "negative_probe", "I found the film dull, disappointing and forgettable."),
    ("positive_1", "positive_probe", "The acting was excellent and the story was wonderful. I loved this movie."),
    ("mixed_1", "mixed_probe", "The actors were excellent, but the story was so boring that I wanted to leave."),
    ("negation_1", "negation_probe", "I expected to hate this film, but it was not bad at all. I enjoyed it."),
]


@torch.inference_mode()
def collect_cls(model, tokenizer, frame, layers, max_length, batch_size=8):
    target = next(model.parameters()).device
    result = {layer: [] for layer in layers}
    for start in range(0, len(frame), batch_size):
        inputs = tokenizer(frame.clean_text.iloc[start:start + batch_size].tolist(), padding=True,
                           truncation=True, max_length=max_length, return_tensors="pt").to(target)
        outputs = model.bert(**inputs, output_hidden_states=True, return_dict=True)
        for layer in layers:
            result[layer].append(outputs.hidden_states[layer][:, 0, :].cpu().numpy())
    return {layer: np.concatenate(values) for layer, values in result.items()}


@torch.inference_mode()
def cache_blocks(model, tokenizer, frame, layer, max_length, batch_size=8):
    target = next(model.parameters()).device
    cached, reference = [], []
    for start in range(0, len(frame), batch_size):
        inputs = tokenizer(frame.clean_text.iloc[start:start + batch_size].tolist(), padding=True,
                           truncation=True, max_length=max_length, return_tensors="pt").to(target)
        outputs = model.bert(**inputs, output_hidden_states=True, return_dict=True)
        logits = model.classifier(model.dropout(outputs.pooler_output))
        reference.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
        cached.append((outputs.hidden_states[layer].cpu(), inputs.attention_mask.cpu()))
    return cached, np.concatenate(reference)


@torch.inference_mode()
def evaluate_cached(model, cached, layer, direction, alpha, scale):
    target = next(model.parameters()).device
    vector = torch.as_tensor(direction, device=target, dtype=torch.float32)
    outputs = []
    for original, mask in cached:
        hidden = original.to(target).clone()
        hidden[:, 0, :] += float(alpha * scale) * vector
        logits = bert_tail_logits(model, hidden, mask.to(target), layer)
        outputs.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(outputs)


def intervention_summary(frame, base, logits):
    p, base_p = expit(logits), expit(base)
    negative, positive = base < 0, base >= 0
    return {"n": len(frame), "mean_p_positive_raw": float(p.mean()),
            "mean_delta_p_raw": float((p - base_p).mean()),
            "baseline_negative_n": int(negative.sum()), "baseline_positive_n": int(positive.sum()),
            "negative_to_positive_fraction": float((logits[negative] >= 0).mean()) if negative.any() else None,
            "positive_to_negative_fraction": float((logits[positive] < 0).mean()) if positive.any() else None,
            "prediction_flip_fraction": float(((logits >= 0) != (base >= 0)).mean())}


def run():
    full_cfg, splits = config(), load_splits()
    cfg = full_cfg["steering"]
    destination = ROOT / "results" / "steering"
    destination.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    frames = {"train": stratified_sample(splits["train"], cfg["train_n"], full_cfg["seed"] + 10),
              "validation": stratified_sample(splits["validation"], cfg["validation_n"], full_cfg["seed"] + 11),
              "test": stratified_sample(splits["test"], cfg["test_n"], full_cfg["seed"] + 12)}
    manifest = {name: frame.id.tolist() for name, frame in frames.items()}
    manifest["bert_checkpoint_sha256"] = file_sha256(ROOT / "artifacts" / "bert" / "model.safetensors")
    save_json(destination / "subset_manifest.json", manifest)
    model, tokenizer, bert_cfg = load_model("bert")
    hidden = {name: collect_cls(model, tokenizer, frame, cfg["layers"], bert_cfg["max_length"])
              for name, frame in frames.items()}
    print("BERT activations extracted for train, validation and held-out test subsets", flush=True)
    # Keep only the small CLS vectors while fitting CPU kernel models.
    model.to("cpu")
    release_memory()
    candidates, fits = [], {}
    for layer in cfg["layers"]:
        for bw in cfg["bandwidths"]:
            rfm = LaplacianRFM(bandwidth=bw, regularization=cfg["regularization"], iterations=cfg["rfm_iterations"])
            rfm.fit(hidden["train"][layer], frames["train"].label.to_numpy(), hidden["validation"][layer], frames["validation"].label.to_numpy())
            row = {"layer": layer, "bandwidth": bw, "validation_auc": rfm.validation_auc,
                   "selected_iteration": rfm.selected_iteration, "history": rfm.history}
            candidates.append(row)
            fits[(layer, bw)] = rfm
            print("RFM", {k: v for k, v in row.items() if k != "history"}, flush=True)
    # Select layer/bandwidth/iteration without looking at the held-out test.
    selected = max(candidates, key=lambda row: row["validation_auc"])
    layer, bw = selected["layer"], selected["bandwidth"]
    rfm = fits[(layer, bw)]
    rfm.save(destination / "selected_rfm.npz")
    x, y = hidden["train"][layer], frames["train"].label.to_numpy()
    means = x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)
    means /= np.linalg.norm(means)
    logistic = LogisticRegression(C=1, max_iter=1000, solver="liblinear").fit(x, y)
    logistic_direction = logistic.coef_[0] / np.linalg.norm(logistic.coef_[0])
    rng = np.random.default_rng(full_cfg["seed"] + 99)
    directions = {"rfm_agop": rfm.direction, "difference_in_means": means, "logistic_probe": logistic_direction}
    for i in range(cfg["random_controls"]):
        vector = rng.normal(size=x.shape[1])
        directions[f"random_{i + 1}"] = vector / np.linalg.norm(vector)
    # Every direction has unit norm; one COMMON perturbation scale ensures
    # norm-matched controls instead of unfair per-direction standard deviations.
    scale = rfm.projection_std
    np.savez_compressed(destination / "directions.npz", **directions, scale=scale, layer=layer)
    save_json(destination / "probe_selection.json", {"selected": selected, "all_candidates": candidates,
              "heldout_test_rfm_auc": float(roc_auc_score(frames["test"].label, rfm.predict(hidden["test"][layer]))),
              "scale": scale, "scaling_policy": "One training-only RFM projection standard deviation, shared by all unit directions",
              "leading_eigenvalue_fraction": float(rfm.eigenvalues[-1] / np.maximum(rfm.eigenvalues.sum(), 1e-30)),
              "agop_min_eigenvalue": float(rfm.eigenvalues.min()),
              "positive_orientation_covariance": rfm.positive_orientation_covariance})
    del fits, hidden
    release_memory()
    model.to(device()).eval()
    synthetic = pd.DataFrame([{"id": id_, "probe_kind": kind, "clean_text": text} for id_, kind, text in SYNTHETIC_PROBES])
    evaluated_frames = {"validation": frames["validation"], "test": frames["test"], "synthetic": synthetic}
    summaries, individual, zero_errors = [], [], {}
    for split, frame in evaluated_frames.items():
        cached, base = cache_blocks(model, tokenizer, frame, layer, bert_cfg["max_length"])
        zero = evaluate_cached(model, cached, layer, rfm.direction, 0, scale)
        identity_error = float(np.max(np.abs(zero - base)))
        if identity_error > 2e-5:
            raise AssertionError(f"Cached tail differs from unsteered forward by {identity_error}")
        zero_errors[split] = identity_error
        for method, direction in directions.items():
            for alpha in cfg["alphas"]:
                logits = zero if alpha == 0 else evaluate_cached(model, cached, layer, direction, alpha, scale)
                summaries.append({"split": split, "method": method, "alpha": alpha,
                                  "perturbation_l2": float(abs(alpha) * scale), **intervention_summary(frame, base, logits)})
                for i in range(len(frame)):
                    individual.append({"split": split, "id": frame.id.iloc[i], "method": method, "alpha": alpha,
                                       "base_logit": float(base[i]), "steered_logit": float(logits[i]),
                                       "p_positive_raw": float(expit(logits[i])),
                                       "input_text_unchanged": True})
            print("Steering", split, method, "complete", flush=True)
        del cached
        release_memory()
    # Verify the actual forward hook against a fresh unchanged model call.
    inputs = tokenizer(synthetic.clean_text.iloc[:2].tolist(), padding=True, truncation=True,
                       max_length=bert_cfg["max_length"], return_tensors="pt").to(device())
    with torch.inference_mode():
        original = model(**inputs).logits
        with ActivationSteering(model, layer, rfm.direction, 1, scale):
            changed = model(**inputs).logits
        restored = model(**inputs).logits
    if not torch.equal(original, restored):
        raise AssertionError("Steering hook leaked into later inference")
    save_json(destination / "metadata.json", {"seconds": time.perf_counter() - start, "config": cfg,
              "selected_layer": layer, "subset_sizes": {name: len(frame) for name, frame in frames.items()},
              "zero_alpha_max_logit_difference": zero_errors,
              "hook_restoration_exact": bool(torch.equal(original, restored)),
              "nonzero_hook_changes_logits": bool(not torch.equal(original, changed)),
              "source": "https://arxiv.org/html/2502.03708v2", "scope": "CLS after an intermediate BERT encoder block",
              "agop_definition": "mean_i grad f_RFM(h_i) grad f_RFM(h_i)^T, computed analytically",
              "hessian_note": "AGOP is not the exact Hessian. For squared loss, its scaled contribution is a Gauss-Newton term; a residual Hessian term remains.",
              "method_adaptation": "Paper's Laplace-kernel RFM applied to an encoder classifier's CLS, rather than a decoder's final-token representation.",
              "limitations": ["Classifier predictions change while the source review remains identical.",
                              "Steered probabilities are out of distribution; ordinary temperature calibration is not validated for interventions.",
                              "Synthetic neutral probes are demonstrations, not a neutral class in IMDb.",
                              "Norm-matched random controls and simple concept directions are included.",
                              "The separate masked-LM exercise is required to actually modify text."]})
    pd.DataFrame(summaries).to_csv(destination / "intervention_summary.csv", index=False)
    pd.DataFrame(individual).to_parquet(destination / "interventions.parquet", index=False)
    save_json(destination / "synthetic_probes.json", synthetic.to_dict("records"))
    print("RFM steering complete, selected block", layer, flush=True)
