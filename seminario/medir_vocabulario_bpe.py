"""Compara tamanhos do vocabulário BPE na validação do ensaio pareado."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import torch
from tokenizers import Tokenizer, __version__ as tokenizers_version
from torch.nn.utils.rnn import pad_sequence

from imdb_lab import bpe_study as bpe
from imdb_lab import sequences as shared
from imdb_lab.common import ROOT, file_sha256, save_json


def encode(frame, tokenizer, cap, device):
    tokens = [entry.ids for entry in tokenizer.encode_batch([bpe.canonicalize(text) for text in frame.text])]
    stats = bpe.token_stats(tokens, cap)
    sequences = [torch.tensor(ids[:cap] or [1], dtype=torch.long) for ids in tokens]
    inputs = {
        "input_ids": pad_sequence(sequences, batch_first=True).to(device),
        "lengths": torch.tensor([len(ids) for ids in sequences], device=device),
    }
    return (inputs, torch.tensor(frame.label.to_numpy(), dtype=torch.long, device=device)), stats


def measure(device, output_dir):
    cfg = bpe.settings()
    target = shared.configure(device)
    frames = {role: bpe.load_role(role) for role in ("train", "validation")}
    reference_path = bpe.BASE / "tokenizer.json"
    reference = Tokenizer.from_file(str(reference_path))
    sizes = (5000, 10000, reference.get_vocab_size())
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = []
    for size in sizes:
        path = reference_path if size == sizes[-1] else output_dir / f"tokenizer_{size}.json"
        if not path.exists():
            tokenizer = bpe.fit_bpe(frames["train"].text.tolist(), size, cfg["tokenizer"]["min_frequency"])
            tokenizer.save(str(path))
        tokenizer = Tokenizer.from_file(str(path))
        if tokenizer.get_vocab_size() != size:
            raise ValueError(f"Tamanho de vocabulário inesperado para {size}.")
        train, _ = encode(frames["train"], tokenizer, cfg["token_cap"], target)
        validation, stats = encode(frames["validation"], tokenizer, cfg["token_cap"], target)
        labels = validation[1].cpu().numpy()
        runs = []
        for seed in cfg["seeds"]:
            model = shared.initialize_cnn(cfg["cnn"], seed, target)
            spec = {**cfg["cnn"], "run_id": f"bpe_vocab_{size}"}
            run = shared.train_one(model, train, validation, spec, seed, target)
            metrics = bpe.metrics(run["best_predictions"]["validation"], labels)
            runs.append({
                "seed": seed,
                "best_epoch": run["best_epoch"],
                "initial_state_sha256": run["initial_state_sha256"],
                "batch_order_sha256": run["batch_order_sha256"],
                **metrics,
            })
        groups.append({
            "vocabulary_size": size,
            "tokenizer_sha256": file_sha256(path),
            "validation_tokenization": stats,
            "validation_mean": {key: mean(run[key] for run in runs) for key in ("accuracy", "f1_macro", "ce")},
            "runs": runs,
        })
    for seed_runs in zip(*(group["runs"] for group in groups), strict=True):
        for key in ("seed", "initial_state_sha256", "batch_order_sha256"):
            if len({run[key] for run in seed_runs}) != 1:
                raise ValueError(f"Pareamento divergente em {key}.")
    return {
        "scope": "Exploratório, somente validação. A época de cada treino foi escolhida pela mesma perda de validação.",
        "runtime": {"torch": torch.__version__, "tokenizers": tokenizers_version,
                    "device": torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU"},
        "train_documents": len(frames["train"]),
        "validation_documents": len(frames["validation"]),
        "training": cfg["cnn"],
        "embedding_rows": reference.get_vocab_size(),
        "whitespace_words": cfg["whitespace_words"],
        "token_cap": cfg["token_cap"],
        "data_manifest_sha256": file_sha256(bpe.BASE / "data_manifest.json"),
        "reference_tokenizer_sha256": file_sha256(reference_path),
        "groups": groups,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "results/bpe_vocab/summary.json")
    args = parser.parse_args()
    save_json(args.output, measure(args.device, args.output.parent))
