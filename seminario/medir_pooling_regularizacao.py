"""Compara resumos com a mesma cabeça na CNN e amplia a busca do baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import mean

import numpy as np

from imdb_lab.common import ROOT, file_sha256, read_json, save_json


def fingerprint(*paths):
    sources = ("data/manifest.json", "config.json", "seminario/medir_pooling_regularizacao.py", *paths)
    return {path: file_sha256(ROOT / path) for path in sources}


def baseline(folder, output):
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer

    from auditar_baseline import load, make_logistic, scores
    from imdb_lab.metrics import fit_temperature

    cfg = read_json(ROOT / "config.json")
    c_grid = [0.125, 0.5, 2, 4, 8, 16, 32, 64]
    train, validation = (load(folder, role) for role in ("train", "validation"))
    protocol = {
        "scikit_learn": sklearn.__version__,
        "C_grid": c_grid,
        "penalty": "L2",
        "selection": "F1 macro na validação. Em empate, menor C e manutenção do IDF.",
        "shared_vectorizer": {"ngram_range": [1, 2], "min_df": 2, "max_df": 0.99,
                              "norm": "l2", "sublinear_tf": True,
                              "max_features": cfg["baseline"]["max_features"]},
        "sources": fingerprint("seminario/auditar_baseline.py", "src/imdb_lab/baseline.py",
                               "src/imdb_lab/metrics.py"),
    }
    save_json(output / "protocol.json", protocol)
    candidates, evaluation, choices = [], {}, {}
    for use_idf in (True, False):
        name = "com_idf" if use_idf else "sem_idf"
        settings = protocol["shared_vectorizer"] | {"ngram_range": (1, 2), "use_idf": use_idf}
        vectorizer = TfidfVectorizer(**settings)
        xt = vectorizer.fit_transform(train.clean_text)
        xv = vectorizer.transform(validation.clean_text)
        rows, models = [], {}
        for c in c_grid:
            model = make_logistic(c, cfg["seed"]).fit(xt, train.label)
            assert model.n_iter_.max() < model.max_iter
            row = {"variant": name, "C": c,
                   "train": scores(train.label, model.decision_function(xt)),
                   "validation": scores(validation.label, model.decision_function(xv))}
            rows.append(row)
            models[c] = model
            print(name, c, row["validation"], flush=True)
        best = max(rows, key=lambda row: row["validation"]["f1_macro"])
        choices[name] = best
        for c in (8, best["C"]):
            evaluation[f"{name}_c{c}"] = (vectorizer, models[c])
        candidates.extend(rows)
    winner = max(choices.values(), key=lambda row: (row["validation"]["f1_macro"], -row["C"]))
    save_json(output / "selection.json", {"winner": winner, "choices": choices, "candidates": candidates})

    calibration, test = (load(folder, role) for role in ("calibration", "test"))
    reference = evaluation["com_idf_c8"][0]
    frequencies = {}
    for token in ("movie", "awful", "not good"):
        position = reference.vocabulary_[token]
        idf = reference.idf_[position]
        df = round((len(train) + 1) / np.exp(idf - 1) - 1)
        frequencies[token] = {"df": df, "idf": float(idf)}
    margins = {"test_ids": test.id.to_numpy(dtype=str), "test_labels": test.label.to_numpy(),
               "calibration_labels": calibration.label.to_numpy()}
    test_results = {}
    for name, (vectorizer, model) in evaluation.items():
        zc = model.decision_function(vectorizer.transform(calibration.clean_text))
        zt = model.decision_function(vectorizer.transform(test.clean_text))
        temperature = fit_temperature(zc, calibration.label)
        margins[f"test_{name}"], margins[f"calibration_{name}"] = zt, zc
        test_results[name] = {"temperature": temperature, **scores(test.label, zt, temperature)}
    predictions = output / "predictions.npz"
    np.savez_compressed(predictions, **margins)
    save_json(output / "summary.json", {
        "selected_by_validation": f"{winner['variant']}_c{winner['C']}", "test": test_results,
        "train_documents": len(train), "document_frequencies": frequencies,
        "prediction_sha256": file_sha256(predictions), "selection_sha256": file_sha256(output / "selection.json"),
    })


def cnn(folder, output, device):
    import torch

    from imdb_lab import sequences
    from imdb_lab.metrics import fit_temperature
    from imdb_lab.models import TextCNN, Vocabulary
    from medir_posicao import synthetic_data
    from treinar_cnn_comprimento import (
        BATCH_SIZE, EPOCHS, SEEDS, ReviewBatches, encode_frame, load_splits, margins,
        pair_summary, subset_scores, train_one,
    )
    from treinar_pooling import masked_mean

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    target = sequences.configure(device)
    temperature_pool = 0.5

    class MatchedCNN(TextCNN):
        """Uma saída por filtro, sem pesos extras no pooling."""

        def __init__(self, size, spec):
            super().__init__(size, dropout=spec["dropout"])
            self.pooling = spec["pooling"]

        def forward(self, input_ids, lengths):
            if self.pooling == "max":
                return super().forward(input_ids, lengths)
            x = self.embedding(input_ids).transpose(1, 2)
            summaries = []
            for conv, width in zip(self.convs, self.kernels, strict=True):
                features = conv(x).relu()
                count = (lengths - width + 1).clamp_min(1)[:, None, None]
                positions = torch.arange(features.shape[-1], device=features.device)[None, None, :]
                keep = positions < count
                if self.pooling == "mean":
                    summaries.append(masked_mean(features, keep))
                    continue
                # log of a weighted mean of exponentials, normalized by total weight.
                weights = torch.ones_like(keep, dtype=features.dtype)
                if self.pooling == "smooth_end":
                    weights = 1 + 2 * positions / (count - 1).clamp_min(1)
                log_weights = weights.log().masked_fill(~keep, -torch.inf)
                numerator = (features / temperature_pool + log_weights).logsumexp(dim=2)
                denominator = log_weights.logsumexp(dim=2)
                summaries.append(temperature_pool * (numerator - denominator))
            return self.classifier(self.dropout(torch.cat(summaries, dim=1)))

    default = {"cap": 768, "pooling": "max", "dropout": 0.5, "weight_decay": 0.01}
    changes = {
        "max_400": {"cap": 400},
        "smooth_400": {"cap": 400, "pooling": "smooth"},
        "max_768": {},
        "smooth_768": {"pooling": "smooth"},
        "mean_768": {"pooling": "mean"},
        "smooth_end_768": {"pooling": "smooth_end"},
        "max_dropout02_768": {"dropout": 0.2},
        "max_decay01_768": {"weight_decay": 0.1},
    }
    recipes = {name: default | change for name, change in changes.items()}
    save_json(output / "protocol.json", {
        "seeds": list(SEEDS), "batch_size": BATCH_SIZE, "epochs": EPOCHS, "recipes": recipes,
        "kernels": [3, 4, 5], "features_to_classifier": 300, "smooth_pool_tau": temperature_pool,
        "smooth_end": "Pesos de 1 a 3 em função da posição relativa, antes da normalização.",
        "selection": "Menor CE de validação por época e por semente; receita pela média das três.",
        "sources": fingerprint("src/imdb_lab/models.py", "src/imdb_lab/sequences.py",
                               "src/imdb_lab/metrics.py", "seminario/treinar_cnn_comprimento.py",
                               "seminario/treinar_pooling.py", "seminario/medir_posicao.py"),
        "torch": str(torch.__version__),
    })
    fit_roles = ("train", "validation")
    frames = load_splits(folder, fit_roles)
    vocab = Vocabulary.fit(frames["train"].clean_text, 20000)
    encoded = {role: encode_frame(frame, vocab) for role, frame in frames.items()}
    labels = {role: frame.label.to_numpy(dtype=np.int64) for role, frame in frames.items()}
    batches = {cap: {role: ReviewBatches(encoded[role], labels[role], cap, 5, target) for role in fit_roles}
               for cap in {spec["cap"] for spec in recipes.values()}}
    weight_dir = output / "weights"
    weight_dir.mkdir()
    records, initials, orders = [], {}, {}
    for name, spec in recipes.items():
        for seed in SEEDS:
            print("INÍCIO", name, seed, flush=True)
            torch.manual_seed(seed)
            model = MatchedCNN(len(vocab.token_to_id), spec).to(target)
            initial = sequences.state_hash(model)
            assert initials.setdefault(seed, initial) == initial
            cap = spec["cap"]
            state, record = train_one(model, batches[cap], labels, spec, seed, target)
            val = margins(model, batches[cap]["validation"])
            record["validation"] = subset_scores(labels["validation"], val, 1)
            assert orders.setdefault(seed, record["batch_order_sha256"]) == record["batch_order_sha256"]
            run = f"{name}_s{seed}"
            checkpoint = weight_dir / f"{run}.pt"
            torch.save(state, checkpoint)
            records.append(record | {"recipe": name, "seed": seed, "run": run,
                                     "initial_sha256": initial, "checkpoint_sha256": file_sha256(checkpoint)})
            save_json(output / "progress.json", records)
    by_recipe = {name: mean(row["validation"]["log_loss"] for row in records if row["recipe"] == name)
                 for name in recipes}
    winner = min(by_recipe, key=by_recipe.get)
    save_json(output / "selection.json", {"winner": winner, "validation_ce": by_recipe, "runs": records})
    del batches, model

    evaluation = load_splits(folder, ("calibration", "test"))
    synthetics = synthetic_data(lambda text: vocab.encode(text, None))
    evaluation["synthetic"] = synthetics.rename(columns={"text": "clean_text"})
    inputs = {role: encode_frame(frame, vocab) for role, frame in evaluation.items()}
    seen = {tuple(text) for rows in encoded.values() for text in rows}
    assert all(tuple(text) not in seen for text in inputs["synthetic"])
    y = {role: frame.label.to_numpy(dtype=np.int64) for role, frame in evaluation.items()}
    last = synthetics.family.str.endswith("_antes").to_numpy()
    long = np.array([len(text) > 400 for text in inputs["test"]])
    arrays = {f"{role}_labels": y[role] for role in evaluation}
    arrays |= {"test_ids": evaluation["test"].id.to_numpy(dtype=str), "test_over_400": long,
               "synthetic_family": synthetics.family.to_numpy(dtype=str)}
    test_rows = []
    for name, spec in recipes.items():
        data = {role: ReviewBatches(inputs[role], y[role], spec["cap"], 5, target) for role in evaluation}
        model = MatchedCNN(len(vocab.token_to_id), spec).to(target)
        for row in (row for row in records if row["recipe"] == name):
            run = row["run"]
            checkpoint = weight_dir / f"{run}.pt"
            assert file_sha256(checkpoint) == row["checkpoint_sha256"]
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            logits = {role: margins(model, data[role]) for role in evaluation}
            arrays.update({f"{role}_{run}": values for role, values in logits.items()})
            tau = fit_temperature(logits["calibration"], y["calibration"])
            test_rows.append({
                "recipe": name, "seed": row["seed"], "temperature": tau,
                "train": row["train"], "validation": row["validation"],
                "test": subset_scores(y["test"], logits["test"], tau),
                "test_long": subset_scores(y["test"][long], logits["test"][long], tau),
                "verdict_last": pair_summary(logits["synthetic"][last], y["synthetic"][last]),
                "verdict_first": pair_summary(logits["synthetic"][~last], y["synthetic"][~last]),
            })
    averages = {}
    for name in recipes:
        rows = [row for row in test_rows if row["recipe"] == name]
        averages[name] = {group: {key: mean(row[group][key] for row in rows) for key in rows[0][group]
                                  if key != "n"} for group in ("train", "validation", "test", "test_long")}
        averages[name].update({key: mean(row[key] for row in rows) for key in ("verdict_last", "verdict_first")})
    predictions = output / "predictions.npz"
    np.savez_compressed(predictions, **arrays)
    save_json(output / "summary.json", {
        "selected_by_validation": winner, "mean": averages, "runs": test_rows,
        "test_long_count": int(long.sum()), "synthetic_pairs_per_position": int(last.sum() // 2),
        "prediction_sha256": file_sha256(predictions), "selection_sha256": file_sha256(output / "selection.json"),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("baseline", "cnn"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.model == "baseline":
        baseline(args.data, args.output)
    else:
        cnn(args.data, args.output, args.device)
