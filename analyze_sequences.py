#!/usr/bin/env python3
"""Post-freeze synthetic examples and exact linear readout decomposition of CNN."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from imdb_lab.common import read_json, save_json, file_sha256
from imdb_lab import sequences as lab
from imdb_lab.models import tokenize

EXAMPLES = [
    {"id": "synthetic_positive", "intended": "positivo", "text": "The opening is quiet, but the patient storytelling pays off. The actors make the characters believable, the final scene is moving, and the film stays with you after the credits. I would happily watch it again."},
    {"id": "synthetic_negative", "intended": "negativo", "text": "The premise sounds promising, but the film wastes it. The dialogue is stiff, the characters never develop, and the final twist makes no sense. Even the attractive photography cannot rescue two hours of dull storytelling."},
    {"id": "synthetic_reversal", "intended": "positivo", "text": "I expected to hate this film. I was wrong. The strange opening leads to a clever story, the cast is excellent, and the ending is surprisingly warm. What looked like a disaster turned out to be a wonderful evening."},
    {"id": "synthetic_mixed", "intended": "negativo", "text": "Beautiful images, a talented cast, and a memorable score are all here. Unfortunately the story is a mess. The impressive first ten minutes give way to repetitive scenes, empty speeches, and an ending that feels completely unearned. I cannot recommend it."}
]


@torch.inference_mode()
def cnn_readout(model, inputs, texts):
    model.eval(); embedded = model.embedding(inputs["input_ids"]).transpose(1, 2)
    parts, descriptors = [], []
    for conv, k in zip(model.convs, model.kernels):
        z = torch.relu(conv(embedded))
        valid = (inputs["lengths"] - k + 1).clamp_min(1)
        keep = torch.arange(z.shape[-1], device=z.device)[None, :] < valid[:, None]
        z = z.masked_fill(~keep[:, None, :], torch.finfo(z.dtype).min)
        values, locations = z.max(2); parts.append(values)
        for f in range(values.shape[1]): descriptors.append((k, f, locations[:, f].cpu().tolist()))
    h = torch.cat(parts, 1)
    delta = model.classifier.weight[1] - model.classifier.weight[0]
    contribution = (h * delta).cpu().numpy()
    bias = float(model.classifier.bias[1] - model.classifier.bias[0])
    logits = model(**inputs).cpu().numpy()
    outputs = []
    for i, text in enumerate(texts):
        windows = []
        words = tokenize(text)[:lab.settings()["common_input"]["token_cap"]]
        for j, (k, f, locations) in enumerate(descriptors):
            start = locations[i]
            windows.append({"kernel": k, "filter": f, "start": start, "words": " ".join(words[start:start + k]), "contribution": float(contribution[i, j])})
        reconstructed = float(contribution[i].sum()) + bias
        assert abs(reconstructed - float(logits[i, 1] - logits[i, 0])) < 1e-5
        unique = {}
        for w in windows:
            key = (w["kernel"], w["start"])
            if key not in unique: unique[key] = {"kernel": w["kernel"], "start": w["start"], "words": w["words"], "contribution": 0.0}
            unique[key]["contribution"] += w["contribution"]
        ranked = sorted(unique.values(), key=lambda x: x["contribution"])
        outputs.append({"bias_margin": bias, "reconstructed_margin": reconstructed, "logit_margin": float(logits[i, 1] - logits[i, 0]),
                        "positive_windows": [x for x in ranked[::-1] if x["contribution"] > 0][:3],
                        "negative_windows": [x for x in ranked if x["contribution"] < 0][:3], "filter_contributions": windows})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, default=Path(".cache/sequence_pretrained"))
    parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    from transformers import AutoTokenizer
    import joblib
    target = lab.configure(args.device); frozen = read_json(lab.SROOT / "selection_frozen.json")
    assert frozen["code"] == lab.fingerprint()
    frame = pd.DataFrame(EXAMPLES); frame["text"] = frame.text.map(lambda t: lab.excerpt(t, 128))
    labels = torch.zeros(len(frame), dtype=torch.long, device=target)
    records = [read_json(lab.SROOT / "runs" / f"{r}.json") for r in frozen["audit_run_ids"]]
    assert all(r["seed"] in [17, 29, 43] for r in records)
    examples = [dict(row, probabilities={}) for row in EXAMPLES]
    base_cnn = next(r for r in records if r["spec"]["family"] == "cnn" and r["spec"]["group"] == "optimizer" and r["spec"]["optimizer"] == "adam" and r["seed"] == 17)
    model = lab.initialize_cnn(base_cnn["spec"], 17, target)
    assert file_sha256(ROOT / base_cnn["checkpoint"]["path"]) == base_cnn["checkpoint"]["sha256"]
    model.load_state_dict(torch.load(ROOT / base_cnn["checkpoint"]["path"], map_location="cpu", weights_only=True))
    x, _ = lab.cnn_inputs(frame, target)
    z = lab.predict(model, x, labels, 64, target)["logits"]
    decomposition = cnn_readout(model, x, frame.text.tolist())
    for i, ex in enumerate(examples): ex["probabilities"]["cnn"] = float(softmax(z, axis=1)[i, 1]); ex["cnn_readout"] = decomposition[i]
    del model
    for entry in lab.settings()["encoders"]["models"]:
        name = entry["name"]; modeldir = args.pretrained / entry["model_id"].split("/")[-1]
        tok = AutoTokenizer.from_pretrained(modeldir, local_files_only=True); x, _ = lab.encoder_inputs(tok, frame, target)
        stock, _ = lab.load_encoder(name, args.pretrained); stock.to(target)
        features = lab.predict(stock, x, labels, 64, target, True, features=True)
        del stock
        for policy in ["probe", "full"]:
            r = next(r for r in records if r["spec"]["family"] == name and r["spec"]["group"] == policy and r["seed"] == 17)
            cp = ROOT / r["checkpoint"]["path"]; assert file_sha256(cp) == r["checkpoint"]["sha256"]
            if policy == "probe": model = lab.Probe(features.shape[1]).to(target); xx = {"x": torch.tensor(features, device=target)}
            else: model, _ = lab.load_encoder(name, args.pretrained); model.to(target); xx = x
            model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
            z = lab.predict(model, xx, labels, 64, target, policy == "full")["logits"]
            for i, ex in enumerate(examples): ex["probabilities"][f"{name}_{policy}"] = float(softmax(z, axis=1)[i, 1])
            del model
    vectorizer, model = joblib.load(ROOT / frozen["baseline"]["checkpoint"]["path"])
    p = model.predict_proba(vectorizer.transform(frame.text))[:, 1]
    for i, ex in enumerate(examples): ex["probabilities"]["tfidf"] = float(p[i])
    save_json(lab.SROOT / "synthetic_examples.json", {"purpose": "Illustrative texts predefined before observing predictions, evaluated after freezing the models; not a labeled benchmark", "seed": 17,
              "selection_frozen_sha256": file_sha256(lab.SROOT / "selection_frozen.json"), "examples": examples,
              "cnn_readout_interpretation": "Exact decomposition of the final linear positive-minus-negative logit margin in eval mode. Contributions combine max-pooled filters at their winning windows. They are not causal word attributions: editing text can change all windows and maxima."})
    print("Synthetic examples and exact CNN readout verified", len(examples))

if __name__ == "__main__": main()
