"""Compara duas leituras dos mesmos picos, com os filtros da TextCNN congelados."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import torch

from demo import HERE, ROOT, load_models
from imdb_lab.bpe_study import metrics
from imdb_lab.common import file_sha256, read_json
from imdb_lab.data import clean_text
from imdb_lab.sequences import array_hash, cnn_inputs, excerpt, indexed


SEEDS = (17, 29, 43)
C_GRID = (.001, .01, .1, 1., 10.)
POSITIVE = "wonderful excellent brilliant fantastic amazing enjoyable delightful superb".split()
NEGATIVE = "terrible awful dreadful horrible boring disappointing abysmal mediocre".split()
TEMPLATES = {
    "expectativa_antes": "I expected the movie to be {initial}, but overall it was {verdict}.",
    "expectativa_depois": "The movie was {verdict} overall, although I expected it to be {initial}.",
    "impressao_antes": "At first the film seemed {initial}, but in the end it was {verdict}.",
    "impressao_depois": "In the end it was {verdict}, although at first the film seemed {initial}.",
}
CONTRAST_REV = "e2b87316ba7b30093bc9ffbc340e6550cab79f67"
CONTRAST_FILES = {
    "test_original.tsv": "9dc78ddeb5cb6ff6060d24df33a83b6d29f9828994f4b1a5e0086f7e774713bd",
    "test_contrast.tsv": "3bcb3bd41218b6a5d7bef7fe020a243919f3df0f12f55291b25ef8bbdbe04173",
}
HEADS = ("original", "global_reajustado", "dois_picos_sem_posicao", "dois_picos_com_posicao")


def contrast_data(folder: Path, seen: set, encode) -> tuple[pd.DataFrame, list[int], int]:
    files = []
    for name, digest in CONTRAST_FILES.items():
        path = folder / name
        if file_sha256(path) != digest:
            raise ValueError(f"Versão inesperada do Contrast Set: {name}")
        with path.open(encoding="utf-8", newline="") as stream:
            files.append(list(csv.DictReader(stream, delimiter="\t")))
    assert all(len(rows) == 488 for rows in files)
    pairs, row_ids, identical = [], [], 0
    for line, (original, contrast) in enumerate(zip(*files), 2):
        pair = [(clean_text(row["Text"]), {"Negative": 0, "Positive": 1}[row["Sentiment"]])
                for row in (original, contrast)]
        assert pair[0][1] != pair[1][1]
        visible = [tuple(encode(text)) for text, _ in pair]
        if any(ids in seen for ids in visible):
            continue
        identical += visible[0] == visible[1]
        row_ids.append(line)
        pairs.extend(pair)
    return pd.DataFrame(pairs, columns=["text", "label"]), row_ids, identical


def synthetic_data(encode) -> pd.DataFrame:
    pairs = []
    for family, template in TEMPLATES.items():
        for positive in POSITIVE:
            for negative in NEGATIVE:
                a = template.format(initial=negative, verdict=positive)
                b = template.format(initial=positive, verdict=negative)
                first, second = encode(a), encode(b)
                assert first != second and 1 not in first and 1 not in second
                assert Counter(first) == Counter(second)
                pairs.extend([(a, 1, family), (b, 0, family)])
    return pd.DataFrame(pairs, columns=["text", "label", "family"])


@torch.inference_mode()
def extract(model, inputs: dict) -> tuple[dict, np.ndarray]:
    """Divide as posições válidas de cada largura, deixando o excedente na primeira metade."""
    early, late, originals = [], [], []
    for start in range(0, len(inputs["lengths"]), 64):
        batch = indexed(inputs, slice(start, start + 64))
        x = model.embedding(batch["input_ids"]).transpose(1, 2)
        first, second = [], []
        for conv, width in zip(model.convs, model.kernels):
            features = conv(x).relu()
            n_valid = (batch["lengths"] - width + 1).clamp_min(1)[:, None, None]
            positions = torch.arange(features.shape[-1])[None, None, :]
            midpoint = (n_valid + 1) // 2
            first.append(features.masked_fill(positions >= midpoint, 0).amax(dim=2))
            keep_second = (positions >= midpoint) & (positions < n_valid)
            second.append(features.masked_fill(~keep_second, 0).amax(dim=2))
        first, second = torch.cat(first, dim=1), torch.cat(second, dim=1)
        early.append(first.numpy())
        late.append(second.numpy())
        originals.append(model.classifier(torch.maximum(first, second)).numpy())
    first, second = np.concatenate(early), np.concatenate(late)
    return {
        "global_reajustado": np.maximum(first, second),
        "dois_picos_sem_posicao": np.concatenate([np.minimum(first, second), np.maximum(first, second)], axis=1),
        "dois_picos_com_posicao": np.concatenate([first, second], axis=1),
    }, np.concatenate(originals)


def pair_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    predictions = logits.argmax(axis=1)
    pairs = (predictions == labels).reshape(-1, 2).all(axis=1)
    return {**metrics(logits, labels), "pares_corretos": int(pairs.sum()), "fracao_pares": float(pairs.mean())}


def fit(train, validation, y_train, y_val):
    candidates = []
    for c in C_GRID:
        head = LogisticRegression(C=c, max_iter=2000, tol=1e-6, solver="lbfgs")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            head.fit(train, y_train)
        margin = head.decision_function(validation)
        ce = np.logaddexp(0, (1 - 2 * y_val) * margin).mean()
        candidates.append((float(ce), c, head))
    return min(candidates, key=lambda item: (item[0], item[1]))


def exact_control(case: dict, encode, widths: tuple) -> None:
    a, b = (encode(case[key]) for key in ("a", "b"))
    for k in widths:
        windows_a = [tuple(a[i:i+k]) for i in range(len(a)-k+1)]
        windows_b = [tuple(b[i:i+k]) for i in range(len(b)-k+1)]
        midpoint = (len(windows_a) + 1) // 2
        assert Counter(windows_a) == Counter(windows_b)
        assert set(windows_a[:midpoint]) == set(windows_b[midpoint:])
        assert set(windows_b[:midpoint]) == set(windows_a[midpoint:])


def evaluate(source: Path, contrast_folder: Path, artifacts: Path, output: Path) -> dict:
    torch.set_num_threads(4)
    manifest_path = ROOT / "results/bpe/data_manifest.json"
    manifest = read_json(manifest_path)
    if file_sha256(source) != manifest["source_sha256"]:
        raise ValueError("O arquivo precisa ser o train.parquet original verificado.")
    config = read_json(ROOT / "bpe_config.json")
    words, cap = config["whitespace_words"], config["token_cap"]
    dataset = pd.read_parquet(source).set_index("id")
    frames = {}
    used = set()
    for role in ("train", "validation", "audit"):
        declared = manifest["roles"][role]
        assert used.isdisjoint(declared["ids"])
        used.update(declared["ids"])
        frame = dataset.loc[declared["ids"]].copy()
        assert frame["text_sha256"].tolist() == declared["text_sha256"]
        assert array_hash(frame["label"].to_numpy(dtype=np.int64)) == declared["labels_sha256"]
        frame["text"] = frame["clean_text"].map(lambda value: excerpt(value, words))
        assert frame["text"].map(lambda value: hashlib.sha256(value.encode()).hexdigest()).tolist() == declared["excerpt_sha256"]
        frames[role] = frame

    models = [load_models(artifacts, seed)[0] for seed in SEEDS]
    vocabulary = models[0].tokenizer

    def encode(text):
        return vocabulary.encode(excerpt(text, words), cap)

    seen = {tuple(encode(value)) for role in ("train", "validation") for value in frames[role]["text"]}
    frames["contrast"], row_ids, identical = contrast_data(contrast_folder, seen, encode)
    frames["synthetic"] = synthetic_data(encode)
    cases = read_json(HERE / "casos.json")["pares"]
    case = next(case for case in cases if case["id"] == "janelas")
    exact_control(case, encode, tuple(config["cnn"]["kernels"]))
    frames["control"] = pd.DataFrame({"text": [case["a"], case["b"]]})
    for role in ("contrast", "synthetic", "control"):
        frames[role]["text"] = frames[role]["text"].map(lambda value: excerpt(value, words))

    inputs, labels = {}, {}
    for role, frame in frames.items():
        inputs[role], _ = cnn_inputs(frame, torch.device("cpu"))
        if "label" in frame:
            labels[role] = frame["label"].to_numpy(dtype=np.int64)

    # Só guardamos rótulos e predições das avaliações, sem republicar o texto de terceiros.
    roles = ("audit", "synthetic", "contrast")
    arrays = {f"{role}_labels": labels[role] for role in roles}
    arrays["contrast_tsv_rows"] = np.asarray(row_ids)
    original_path = ROOT / "results/bpe/audit_predictions.npz"
    results = []
    with np.load(original_path, allow_pickle=False) as reference:
        np.testing.assert_array_equal(labels["audit"], reference["labels"])
        for word in models:
            seed = word.seed
            features, original = {}, {}
            for role, batch in inputs.items():
                features[role], original[role] = extract(word.model, batch)
            np.testing.assert_allclose(original["audit"], reference[f"word__s{seed}"], atol=5e-5, rtol=0)
            np.testing.assert_array_equal(original["audit"].argmax(1), reference[f"word__s{seed}"].argmax(1))
            a, b = features["control"]["dois_picos_com_posicao"]
            np.testing.assert_allclose(a, np.concatenate(np.split(b, 2)[::-1]), atol=1e-6, rtol=0)
            per_seed = {"semente": seed, "checkpoint_sha256": word.checkpoint_sha256, "cabecas": {}}
            for name in HEADS:
                if name == "original":
                    logits = original
                    record = {}
                else:
                    ce, c, head = fit(features["train"][name], features["validation"][name],
                                      labels["train"], labels["validation"])
                    logits = {role: np.column_stack([np.zeros(len(frames[role])), head.decision_function(features[role][name])])
                              for role in (*roles, "control")}
                    record = {"C": c, "ce_validacao": ce}
                margins = logits["control"][:, 1] - logits["control"][:, 0]
                if name != "dois_picos_com_posicao":
                    np.testing.assert_allclose(margins[0], margins[1], atol=1e-6, rtol=0)
                record["margens_controle_A_B"] = margins.tolist()
                for role in roles:
                    arrays[f"{role}_{name}_s{seed}"] = logits[role]
                    record[role] = metrics(logits[role], labels[role]) if role == "audit" else pair_metrics(logits[role], labels[role])
                record["familias_sinteticas"] = {}
                for family in TEMPLATES:
                    keep = frames["synthetic"]["family"] == family
                    record["familias_sinteticas"][family] = pair_metrics(logits["synthetic"][keep], labels["synthetic"][keep])
                per_seed["cabecas"][name] = record
            results.append(per_seed)
            print("Semente concluída:", seed, flush=True)

    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "posicao_predicoes.npz"
    np.savez_compressed(predictions, **arrays)
    return {
        "manifesto_sha256": file_sha256(manifest_path), "auditoria_original_sha256": file_sha256(original_path),
        "predicoes_sha256": file_sha256(predictions),
        "n": {role: len(frame) for role, frame in frames.items()},
        "sintetico": {"positivos": POSITIVE, "negativos": NEGATIVE, "moldes": TEMPLATES},
        "contraste": {"revisao_github": CONTRAST_REV, "arquivos_sha256": CONTRAST_FILES,
                      "pares_excluidos_por_coincidencia_com_treino_ou_validacao": 488 - len(row_ids),
                      "pares_com_entrada_identica_apos_recorte": identical},
        "controle": case["id"], "grade_C": C_GRID, "execucoes": results,
        "medias": {
            name: {role: {key: mean(run["cabecas"][name][role][key] for run in results)
                          for key in results[0]["cabecas"][name][role]} for role in roles}
            for name in HEADS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--contrast-dir", type=Path, required=True, help="Pasta com os dois TSV de teste de Gardner et al.")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "dados")
    args = parser.parse_args()
    data = evaluate(args.train, args.contrast_dir, args.artifacts, args.output_dir)
    (args.output_dir / "posicao_resultados.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    for name, values in data["medias"].items():
        print(name, values["audit"]["accuracy"], values["synthetic"]["fracao_pares"], values["contrast"]["fracao_pares"])


if __name__ == "__main__":
    main()
