"""Actual text editing with a BERT masked-LM head and RFM intervention.

This is a separate exercise from changing a classifier's prediction. It edits
explicitly masked opinion words, not unrestricted autoregressive generation.
All intervention settings are retained, including unsuccessful/nonsensical ones.
"""
from __future__ import annotations

import re
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from .common import ROOT, config, device, read_json, release_memory, save_json
from .rfm import ActivationSteering
from .training import predict_texts

EDIT_PROBES = [
    {"id": "neutral_a", "kind": "synthetic_neutral", "text": "The movie was ordinary and the acting was average.", "mask_words": ["ordinary", "average"]},
    {"id": "neutral_b", "kind": "synthetic_neutral", "text": "The story was simple and the ending was predictable.", "mask_words": ["simple", "predictable"]},
    {"id": "negative_a", "kind": "synthetic_negative", "text": "The acting was terrible and the story was boring.", "mask_words": ["terrible", "boring"]},
    {"id": "negative_b", "kind": "synthetic_negative", "text": "The movie was dull and the performances were poor.", "mask_words": ["dull", "poor"]},
    {"id": "positive_a", "kind": "synthetic_positive", "text": "The movie was wonderful and the actors were brilliant.", "mask_words": ["wonderful", "brilliant"]},
]


def mask_opinion_words(tokenizer, text, words):
    encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping")[0].tolist()
    spans = [(m.start(), m.end()) for m in re.finditer(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", text, flags=re.I)]
    positions = [i for i, (a, b) in enumerate(offsets) if b > a and any(a < end and b > start for start, end in spans)]
    if len(positions) != len(words):
        raise ValueError("These demonstration words must each tokenize as exactly one WordPiece")
    for pos in positions:
        encoded.input_ids[0, pos] = tokenizer.mask_token_id
    return encoded, positions


@torch.inference_mode()
def fill_masks(model, tokenizer, encoded, positions, layer, direction, alpha, scale, allowed_ids):
    inputs = {key: value.clone().to(device()) for key, value in encoded.items()}
    choices = []
    allowed = torch.tensor(allowed_ids, device=device())
    # Fixed left-to-right ordering is shared by all conditions.
    for pos in positions:
        with ActivationSteering(model, layer, direction, alpha, scale, scope="all", attention_mask=inputs["attention_mask"]):
            scores = model(**inputs).logits[0, pos]
        values, where = torch.topk(scores[allowed], k=3)
        candidates = allowed[where].cpu().tolist()
        inputs["input_ids"][0, pos] = candidates[0]
        choices.append({"position": pos, "top_tokens": tokenizer.convert_ids_to_tokens(candidates)})
    text = tokenizer.decode(inputs["input_ids"][0].cpu(), skip_special_tokens=True)
    return text, choices


def run():
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, BertForMaskedLM

    started = time.perf_counter()
    destination = ROOT / "results" / "rewrite"
    destination.mkdir(parents=True, exist_ok=True)
    bert_metadata = read_json(ROOT / "results" / "bert" / "metadata.json")
    directions_file = np.load(ROOT / "results" / "steering" / "directions.npz")
    layer, scale = int(directions_file["layer"]), float(directions_file["scale"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "artifacts" / "bert", local_files_only=True)
    model = BertForMaskedLM.from_pretrained(bert_metadata["pretrained_model_id"], revision=bert_metadata["revision"], attn_implementation="eager")
    saved = load_file(str(ROOT / "artifacts" / "bert" / "model.safetensors"))
    encoder_weights = {key.removeprefix("bert."): value for key, value in saved.items() if key.startswith("bert.") and not key.startswith("bert.pooler.")}
    incompatible = model.bert.load_state_dict(encoder_weights, strict=True)
    matches = all(torch.equal(value, model.bert.state_dict()[key]) for key, value in encoder_weights.items())
    if not matches:
        raise AssertionError("Masked LM does not share the fine-tuned classifier's encoder weights")
    tied_decoder = model.cls.predictions.decoder.weight.data_ptr() == model.bert.embeddings.word_embeddings.weight.data_ptr()
    del saved, encoder_weights
    release_memory()
    model.to(device()).eval()
    # Keep single alphabetic WordPieces, allowing genuine model choices while
    # preventing a punctuation/special-token artifact from replacing an adjective.
    allowed_ids = [i for token, i in tokenizer.get_vocab().items() if re.fullmatch("[a-z]+", token)]
    alphas = [-8, -4, -2, 0, 2, 4, 8]
    rows = []
    for probe in EDIT_PROBES:
        encoded, positions = mask_opinion_words(tokenizer, probe["text"], probe["mask_words"])
        masked = tokenizer.decode(encoded.input_ids[0], skip_special_tokens=False)
        for method in ("rfm_agop", "random_1"):
            for alpha in alphas:
                edited, choices = fill_masks(model, tokenizer, encoded, positions, layer,
                                             directions_file[method], alpha, scale, allowed_ids)
                rows.append({"id": probe["id"], "kind": probe["kind"], "method": method, "alpha": alpha,
                             "original": probe["text"], "masked_input": masked, "edited": edited, "choices": choices})
        print("Masked-LM edit probe", probe["id"], "complete", flush=True)
    model.to("cpu")
    del model
    release_memory()
    # Evaluate unchanged originals and edited outputs WITHOUT any steering hook.
    unique_texts = list(dict.fromkeys([p["text"] for p in EDIT_PROBES] + [row["edited"] for row in rows]))
    scores = {}
    baseline = joblib.load(ROOT / "artifacts" / "tfidf" / "model.joblib")
    raw = baseline["model"].decision_function(baseline["vectorizer"].transform(unique_texts))
    temp = read_json(ROOT / "results" / "tfidf" / "metadata.json")["temperature"]
    scores["tfidf"] = dict(zip(unique_texts, expit(raw / temp).tolist()))
    del baseline
    for kind in ("cnn", "bert"):
        raw = predict_texts(kind, unique_texts)
        temp = read_json(ROOT / "results" / kind / "metadata.json")["temperature"]
        scores[kind] = dict(zip(unique_texts, expit(raw / temp).tolist()))
    from sentence_transformers import SentenceTransformer
    analysis_metadata = read_json(ROOT / "results" / "analysis" / "metadata.json")
    encoder = SentenceTransformer(analysis_metadata["encoder"], revision=analysis_metadata["revision"], device=str(device()))
    embeddings = encoder.encode(unique_texts, normalize_embeddings=True)
    text_to_embedding = dict(zip(unique_texts, embeddings))
    for row in rows:
        for kind in scores:
            row[kind + "_original_p_positive"] = scores[kind][row["original"]]
            row[kind + "_edited_p_positive"] = scores[kind][row["edited"]]
        row["semantic_cosine_to_original"] = float(text_to_embedding[row["original"]] @ text_to_embedding[row["edited"]])
        row["human_fluency_and_sentiment_review"] = "Required for a linguistic success claim; not automatically inferred from model probability."
    save_json(destination / "examples.json", rows)
    pd.DataFrame([{k: v for k, v in row.items() if k != "choices"} for row in rows]).to_csv(destination / "examples.csv", index=False)
    save_json(destination / "metadata.json", {"seconds": time.perf_counter() - started, "selected_layer": layer,
              "encoder_matches_classifier": matches, "encoder_transfer_missing_keys": incompatible.missing_keys,
              "mlm_head": "MLM transform and bias from pretraining; output projection remains tied to the fine-tuned input embeddings. Encoder exactly matches the classifier.",
              "decoder_tied_to_fine_tuned_embeddings": tied_decoder,
              "method": "Sequentially fill two explicit opinion masks; additive RFM direction at every non-padding position",
              "alphas": alphas, "norm_matched_random_control": "random_1", "probe_n": len(EDIT_PROBES),
              "results_n": len(rows), "selection": "All generated outputs retained. No sentiment-guided candidate reranking.",
              "caveats": ["A masked-LM editing extension, not the paper's autoregressive generation experiment.",
                           "Synthetic probes are not a held-out benchmark of general rewriting.",
                           "The masked LM sees masks, so compare each steering condition to alpha=0 as well as the original.",
                           "Transferring a CLS concept direction to all tokens and an MLM head may fail.",
                           "Large interventions can damage fluency or change unrelated content.",
                           "Classifier probabilities and semantic similarity do not establish human-rated success.",
                           "No claim that a neutral IMDb class exists."]})
    print("Actual text-editing exercise complete", flush=True)
