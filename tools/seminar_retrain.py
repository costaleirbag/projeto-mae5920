"""Reexecute os estudos em um diretório extraído pelo notebook do seminário."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
from statistics import mean
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "referencia"
OUTPUT = ROOT / "results/seminar_retrain"
sys.path.insert(0, str(ROOT / "src"))


def command(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                   cwd=ROOT, check=True)


def prepare_corpus():
    import pandas as pd
    from imdb_lab.common import config, file_sha256, read_json

    expected = read_json(REFERENCE / "data/manifest.json")
    roles = read_json(REFERENCE / "derived/papeis_originais.json")
    cfg = config()
    if cfg["dataset"] != expected["source"] or cfg["dataset_revision"] != expected["revision"]:
        raise ValueError("A origem e a revisão do IMDb devem coincidir com a referência.")
    data = ROOT / "data"
    if not all((data / name).is_file() for name in ["manifest.json", *(f"{r}.parquet" for r in roles)]):
        command("run.py", "prepare")
    prepared = read_json(data / "manifest.json")
    for key in ("source", "revision", "audit"):
        if prepared[key] != expected[key]:
            raise ValueError(f"O IMDb preparado diverge do original em {key}.")
    hashes = {}
    for role, reference in roles.items():
        path = data / f"{role}.parquet"
        if prepared["splits"][role]["sha256"] != file_sha256(path):
            raise ValueError(f"Arquivo local alterado desde a preparação: {role}.")
        frame = pd.read_parquet(path, columns=["id", "label", "text_sha256"])
        if frame.id.tolist() != reference["ids"] or frame.label.tolist() != reference["labels"]:
            raise ValueError(f"IDs, ordem ou rótulos divergentes em {role}.")
        hashes[role] = set(frame.text_sha256)
    for role, fingerprints in hashes.items():
        if any(fingerprints & other for name, other in hashes.items() if name != role):
            raise ValueError(f"Textos repetidos entre {role} e outro papel.")
    print("IMDb conferido pela revisão, deduplicação, IDs, rótulos e ordem dos quatro papéis.", flush=True)


def run_comparisons(observed, references):
    import pandas as pd

    return pd.DataFrame([
        {"grupo": group, "métrica": metric, "nova execução": value,
         "publicado": references[group][metric], "diferença": value - references[group][metric]}
        for group, values in observed.items()
        for metric, value in values.items()
    ])


def report_metrics(stage, observed, reference, metadata=None):
    from imdb_lab.common import save_json

    comparison = run_comparisons(observed, reference)
    target = OUTPUT / stage / "comparacao.json"
    save_json(target, {"comparacao": comparison.to_dict(orient="records"), **(metadata or {})})
    print(comparison.to_string(index=False, float_format="{:.6f}".format), flush=True)
    print("Comparação em", target.relative_to(ROOT), flush=True)


def official_results(stage, directories):
    from imdb_lab.common import read_json

    observed, reference = {}, {}
    for model, folder in directories.items():
        versions = (observed, reference)
        for path, results in zip((ROOT, REFERENCE), versions, strict=True):
            scores = read_json(path / folder / "metrics.json")
            for role in scores:
                raw = scores[role]["raw"]
                calibrated = scores[role]["calibrated"]
                results[f"{model}/{role}"] = {
                    "accuracy": raw["accuracy"], "f1_macro": raw["f1_macro"],
                    "ce_bruta": raw["log_loss"], "ce_calibrada": calibrated["log_loss"],
                }
    report_metrics(stage, observed, reference)


def original(models):
    for model in models:
        command("run.py", "baseline" if model == "tfidf" else model)
    official_results("original", {model: f"results/{model}" for model in models})


def role_frame(source, role, words):
    import numpy as np
    from imdb_lab.sequences import array_hash, excerpt

    frame = source.loc[role["ids"]].reset_index()
    prefix = [excerpt(text, words) for text in frame.clean_text]
    if (len(frame) != role["n"] or frame.text_sha256.tolist() != role["text_sha256"]
            or [hashlib.sha256(text.encode()).hexdigest() for text in prefix] != role["excerpt_sha256"]
            or array_hash(frame.label.to_numpy(dtype=np.int64)) != role["labels_sha256"]):
        raise ValueError("IDs, rótulos e textos devem coincidir com o manifesto dos ensaios internos.")
    return frame.assign(text=prefix)[["id", "label", "text"]]


def paired_cnn(device):
    import numpy as np
    import pandas as pd
    import torch
    from imdb_lab import bpe_study as bpe, sequences as shared
    from imdb_lab.common import file_sha256, read_json, save_json
    from imdb_lab.models import Vocabulary

    destination = OUTPUT / "textcnn"
    weights = ROOT / "artifacts/seminar_retrain/textcnn"
    if (destination / "selection.json").exists():
        raise FileExistsError("A seleção destas seis CNNs já existe. Para repeti-la, extraia um novo diretório.")
    cfg = bpe.settings()
    target = shared.configure(device)
    trace = REFERENCE / "results/bpe"
    roles = read_json(trace / "data_manifest.json")["roles"]
    source = pd.read_parquet(ROOT / "data/train.parquet").set_index("id", verify_integrity=True)
    train, validation = (role_frame(source, roles[r], cfg["whitespace_words"]) for r in ("train", "validation"))
    historical_vocab = REFERENCE / "results/sequences/vocabulary.json"
    fitted_vocab = Vocabulary.fit(train.text, cfg["tokenizer"]["vocab_size"]).token_to_id
    if fitted_vocab != read_json(historical_vocab):
        raise ValueError("O vocabulário por palavras aprendido no treino diverge da referência.")
    shared.SROOT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(historical_vocab, shared.SROOT / "vocabulary.json")
    tokenizer = trace / "tokenizer.json"
    tokenizer_record = read_json(trace / "tokenizer_training.json")
    canonical = "\n".join(bpe.canonicalize(t) for t in train.text)
    if (file_sha256(tokenizer) != tokenizer_record["tokenizer_sha256"]
            or hashlib.sha256(canonical.encode()).hexdigest() != tokenizer_record["canonical_train_sha256"]):
        raise ValueError("O tokenizer BPE congelado ou seu texto de treino divergem da referência.")
    bpe.BASE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizer, bpe.BASE / "tokenizer.json")
    weights.mkdir(parents=True, exist_ok=True)
    runs = []
    for kind in ("word", "bpe"):
        prepared = []
        for frame in (train, validation):
            inputs = shared.cnn_inputs(frame, target)[0] if kind == "word" else bpe.inputs(frame, kind, target)[0]
            prepared.append((inputs, torch.tensor(frame.label.to_numpy(), dtype=torch.long, device=target)))
        for seed in cfg["seeds"]:
            key = f"{kind}__s{seed}"
            spec = {**cfg["cnn"], "run_id": key}
            model = shared.initialize_cnn(spec, seed, target)
            result = shared.train_one(model, *prepared, spec, seed, target)
            checkpoint = weights / f"{key}.pt"
            torch.save(result.pop("best_state"), checkpoint)
            result.pop("best_predictions")
            runs.append({**result, "kind": kind, "run_id": key, "sha256": file_sha256(checkpoint),
                         "path": checkpoint.relative_to(ROOT).as_posix()})
            del model
    for seed in cfg["seeds"]:
        a, b = (run for run in runs if run["seed"] == seed)
        for field in ("initial_state_sha256", "batch_order_sha256", "trainable_parameters"):
            if a[field] != b[field]:
                raise ValueError(f"Semente {seed}: {field} diverge entre as representações pareadas.")
    save_json(destination / "selection.json", {"runs": runs, "audit_prepared_after_selection": True,
                                               "tokenizer_sha256": file_sha256(tokenizer)})
    audit = role_frame(source, roles["audit"], cfg["whitespace_words"])
    labels = torch.tensor(audit.label.to_numpy(), dtype=torch.long, device=target)
    arrays = {"ids": audit.id.to_numpy(dtype=str), "labels": audit.label.to_numpy(dtype=np.int64)}
    tokenization, observed = {}, {}
    for kind in ("word", "bpe"):
        inputs, tokenization[kind] = bpe.inputs(audit, kind, target)
        for run in (r for r in runs if r["kind"] == kind):
            key = run["run_id"]
            model = shared.initialize_cnn(run["spec"], run["seed"], target)
            model.load_state_dict(torch.load(ROOT / run["path"], map_location=target, weights_only=True))
            arrays[key] = shared.predict(model, inputs, labels, 64, target)["logits"]
            observed[key] = bpe.metrics(arrays[key], arrays["labels"])
            del model
    np.savez_compressed(destination / "audit_predictions.npz", **arrays)
    reference = {x["run_id"]: x for x in read_json(trace / "audit_metrics.json")["metrics"]}
    report_metrics("textcnn", observed, reference, {"device": device, "tokenization": tokenization})


def encoders(device):
    import numpy as np
    from imdb_lab.bpe_study import metrics
    from imdb_lab.common import read_json

    base = ROOT / "results/sequences"
    trace = REFERENCE / "results/sequences"
    manifest = base / "data_manifest.json"
    frozen = base / "selection_frozen.json"
    archive = base / "audit_predictions.npz"
    if not manifest.exists():
        prev = ROOT / "results/didactic/data_manifest.json"
        prev.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REFERENCE / prev.relative_to(ROOT), prev)
        base.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(trace / "models_pinned.json", base / "models_pinned.json")
        command("run_sequences.py", "prepare", "--device", "cpu")
    fresh, expected = read_json(manifest)["roles"], read_json(trace / "data_manifest.json")["roles"]
    for role in ("train", "validation", "audit"):
        for field in ("ids", "labels_sha256", "text_sha256", "excerpt_sha256"):
            if fresh[role][field] != expected[role][field]:
                raise ValueError(f"Ensaio sequencial: {field} não coincide na divisão {role}.")
    staged = ROOT / ".cache/sequence_inputs/audit.parquet"
    withheld = ROOT.parent / f".{ROOT.name}-auditoria-sequencias.parquet"
    if not frozen.exists():
        if staged.exists():
            command("tools/sequence_audit_input.py", "withhold", "--file", withheld)
        if not withheld.is_file():
            raise FileNotFoundError("A entrada de auditoria separada do ambiente de treino foi perdida.")
        command("tools/fetch_sequence_models.py")
        command("run_sequences.py", "train", "--device", device)
    if not archive.exists():
        if not staged.exists():
            command("tools/sequence_audit_input.py", "restore", "--file", withheld)
        command("run_sequences.py", "audit", "--device", device)
    selections = read_json(frozen)["selections"]
    summary = read_json(trace / "summary.json")
    observed, reference = {}, {}
    with np.load(archive, allow_pickle=False) as saved:
        for group in summary["groups"]:
            key = group["key"]
            rate = selections.get(key, selections["cnn_adam"])
            ids = [re.sub(r"__lr[^_]+(?=__s)", f"__lr{rate:g}", run_id) for run_id in group["run_ids"]]
            results = [metrics(saved[run_id], saved["y"]) for run_id in ids]
            fields = ("accuracy", "f1_macro", "ce")
            observed[key] = {k: mean(row[k] for row in results) for k in fields}
            reference[key] = {k: group[f"{k}_mean"] for k in fields}
    report_metrics("encoders", observed, reference, {"device": device, "taxas_escolhidas": selections})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("original", "textcnn", "encoders", "qwen"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--original-models", nargs="+", choices=("tfidf", "cnn", "bert"),
                        default=("tfidf", "cnn", "bert"))
    args = parser.parse_args()
    if (ROOT / ".git").exists() or not (REFERENCE / "data/manifest.json").is_file():
        parser.error("Execute a cópia extraída pelo notebook, separada do repositório publicado.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from imdb_lab.common import device

    detected = device().type
    selected = ("cuda" if detected == "cuda" else "cpu") if args.device == "auto" else args.device
    if args.stage == "original" and args.device not in ("auto", detected):
        parser.error(f"run.py escolhe o dispositivo disponível ({detected}); use --device auto.")
    if args.stage in ("qwen", "encoders") and (detected != "cuda" or selected != "cuda"):
        parser.error("Esta receita usa CUDA para repetir o protocolo BF16 publicado.")
    if args.stage == "textcnn" and selected == "mps":
        parser.error("O ensaio pareado oferece CUDA ou CPU.")
    prepare_corpus()
    if args.stage == "original":
        original(args.original_models)
    elif args.stage == "textcnn":
        paired_cnn(selected)
    elif args.stage == "encoders":
        encoders(selected)
    else:
        command("run_qwen.py", "classify")
        official_results("qwen", {"qwen": "results/qwen/classification"})


if __name__ == "__main__":
    main()
