"""Mede cobertura do BERT e compara dois tetos com os pesos originais."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, __version__ as transformers_version

from imdb_lab.common import ROOT, file_sha256, read_json, save_json
from imdb_lab.metrics import evaluate
from imdb_lab.models import EncodedReviews, PadBatch
from imdb_lab.training import predict
from medir_comprimentos import LENGTH_BANDS, load_data


PREDICTIONS = ROOT / "results/bert/predictions.parquet"
RELEASE_HASHES = {
    "config.json": "2cf9d952422f43f31e386a482c61fdb50324a3a6fe7b0b376815d349bce1bb23",
    "model.safetensors": "62be04b22f7e00fc72e62082e76bfbc1ed251356ea9be88aad9b9022e743b0aa",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "1f92c93d2e31974acffab86d48e80a9357acb18c68346c5472ef1d12c72d5d77",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def run(folder, checkpoint, output, target, batch_size):
    hashes = {name: file_sha256(checkpoint / name) for name in RELEASE_HASHES}
    if hashes != RELEASE_HASHES:
        raise ValueError("Use os pesos e o tokenizer do BERT original.")
    manifest = read_json(ROOT / "data/manifest.json")
    test = load_data(folder, manifest, "test")
    saved = pd.read_parquet(PREDICTIONS).query('split == "test"').set_index("id", verify_integrity=True)
    assert set(saved.index) == set(test.index)
    saved = saved.loc[test.index]
    np.testing.assert_array_equal(saved.label, test.label)
    meta = read_json(ROOT / "results/bert/metadata.json")
    cap = meta["config"]["max_length"]
    architecture_cap = read_json(checkpoint / "config.json")["max_position_embeddings"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    assert special_count == 2 and tokenizer.truncation_side == "right"
    caps = (cap, architecture_cap)

    words = test.word_count.to_numpy()
    texts = test.clean_text.tolist()
    lengths = []
    for start in range(0, len(texts), 256):
        lengths.extend(tokenizer(texts[start:start + 256], add_special_tokens=False, truncation=False,
                                 verbose=False, return_length=True)["length"])
    lengths = np.asarray(lengths)
    bands = []
    for low, high in [*LENGTH_BANDS, (0, None)]:
        group = (words >= low) & ((words <= high) if high is not None else True)
        values = lengths[group]
        bands.append({"min_words": low, "max_words": high, "n": int(group.sum()),
                      "median_wordpieces_without_specials": float(np.median(values)),
                      "truncated": {limit: int((values > limit - special_count).sum()) for limit in caps},
                      "retained": {limit: float(np.minimum(values, limit - special_count).sum() / values.sum())
                                   for limit in caps}})
    print("Cobertura", bands, flush=True)

    low, high = LENGTH_BANDS[-1]
    assert high is None
    selected = np.flatnonzero(words >= low)
    labels = test.label.to_numpy()[selected]
    long_texts = [texts[i] for i in selected]
    model = BertForSequenceClassification.from_pretrained(checkpoint, local_files_only=True,
                                                           attn_implementation="eager").to(target).eval()
    def forward(text_batch, label_batch, limit):
        ids = tokenizer(text_batch, truncation=True, max_length=limit, padding=False)["input_ids"]
        loader = DataLoader(EncodedReviews(ids, label_batch), batch_size=batch_size,
                            collate_fn=PadBatch("bert", tokenizer.pad_token_id))
        return predict(model, loader, target)

    check = np.linspace(0, len(selected) - 1, 64, dtype=int)
    original = saved.logit.to_numpy()[selected]
    first = perf_counter()
    margins = {}
    for limit in caps:
        batches = []
        for start in range(0, len(long_texts), 128):
            end = min(len(long_texts), start + 128)
            batches.append(forward(long_texts[start:end], labels[start:end], limit))
            print(f"{limit} tokens · {end}/{len(long_texts)} · {perf_counter() - first:.1f}s", flush=True)
        margins[limit] = np.concatenate(batches).astype(np.float64)
        assert np.isfinite(margins[limit]).all()
        if limit == cap:
            np.testing.assert_allclose(margins[cap][check], original[check], atol=.01, rtol=0)
    replay, new = (margins[limit] for limit in caps)
    original_metrics = evaluate(labels, expit(original / meta["temperature"]))
    replay_metrics = evaluate(labels, expit(replay / meta["temperature"]))
    new_metrics = evaluate(labels, expit(new / meta["temperature"]))
    before, after = (replay >= 0) == labels, (new >= 0) == labels
    changed_on_replay = int(((replay >= 0) != (original >= 0)).sum())
    assert changed_on_replay <= 2
    output.mkdir(parents=True, exist_ok=False)
    paired = output / "predicoes.npz"
    np.savez_compressed(paired, ids=test.index.to_numpy(dtype=str)[selected], labels=labels,
                        word_count=words[selected], wordpieces=lengths[selected],
                        original=original, replay=replay, extended=new)
    metrics = ("n", "accuracy", "f1_macro", "log_loss")
    result = {
        "caps_including_specials": list(caps), "special_tokens": tokenizer.build_inputs_with_special_tokens([]),
        "coverage": bands, "evaluated_min_words": low,
        "original": {key: original_metrics[key] for key in metrics},
        "replay": {key: replay_metrics[key] for key in metrics},
        "extended": {key: new_metrics[key] for key in metrics},
        "corrected": int((~before & after).sum()), "lost": int((before & ~after).sum()),
        "prefix_replay": {"n": len(replay), "max_logit_difference": float(np.abs(replay - original).max()),
                          "changed_decisions": changed_on_replay},
        "temperature": meta["temperature"], "test_sha256": manifest["splits"]["test"]["sha256"],
        "original_predictions_sha256": file_sha256(PREDICTIONS), "checkpoint_hashes": hashes,
        "paired_predictions_sha256": file_sha256(paired), "source_sha256": file_sha256(Path(__file__)),
        "device": str(target), "torch": str(torch.__version__), "transformers": transformers_version,
    }
    save_json(output / "resumo.json", result)
    print({key: result[key] for key in ("original", "extended", "corrected", "lost")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.data, args.checkpoint, args.output, torch.device(args.device), args.batch_size)
