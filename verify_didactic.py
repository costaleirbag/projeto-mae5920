#!/usr/bin/env python3
"""Recompute the new scientific evidence, including all stored readout predictions."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from imdb_lab.common import read_json, save_json, file_sha256
from imdb_lab.didactic import (DROOT, TeachingMLP, assess, settings, load_role,
                               early_stop_trace, code_fingerprints)
from imdb_lab.didactic_toys import synthetic_case


def main():
    cfg = settings(); evidence = {}; torch.set_num_threads(cfg["threads"])
    manifest = read_json(DROOT / "data_manifest.json")
    assert manifest["config_sha256"] == file_sha256(ROOT / "didactic_config.json")
    assert manifest["official_test_opened"] is False
    for role in ["train", "validation", "audit"]:
        data = load_role(role); info = manifest["roles"][role]
        assert data["x"].shape == (info["n"], cfg["representation"]["svd_components"])
        assert data["ids"].tolist() == info["ids"] and np.isfinite(data["x"]).all()
        assert np.bincount(data["y"]).tolist() == [info["n"] // 2] * 2
        assert all(v.startswith("imdb_train_") for v in data["ids"])
        for other, other_info in manifest["roles"].items():
            if role != other:
                assert set(info["ids"]).isdisjoint(other_info["ids"])
                assert set(info["text_sha256"]).isdisjoint(other_info["text_sha256"])
    assert file_sha256(DROOT / "representation.npz") == manifest["representation_sha256"]
    evidence["data"] = "6144/1536/2048 balanced, text-disjoint internal roles use only the original cleaned training pool; representation fitted on the 6144 training rows."
    frozen = read_json(DROOT / "selection_frozen.json")
    assert frozen["code"] == code_fingerprints() and not frozen["audit_opened"]
    assert frozen["data_manifest_sha256"] == file_sha256(DROOT / "data_manifest.json")
    for path, digest in frozen["run_sha256"].items(): assert file_sha256(ROOT / path) == digest
    runs = [read_json(p) for p in sorted((DROOT / "runs").glob("*.json"))]
    assert len(runs) == frozen["n_runs"] == 51
    for run in runs:
        epochs = cfg[run["group"]]["epochs"]
        assert [r["epoch"] for r in run["history"]] == list(range(epochs + 1))
        assert all(np.isfinite(float(v)) for row in run["history"] for v in row.values())
        for row in run["history"]:
            assert np.isclose(row["objective"], row["train_ce"] + .5 * run["l2"] * row["weight_norm"] ** 2, atol=1e-7)
        model = TeachingMLP(**run["spec"])
        assert sum(p.numel() for p in model.parameters()) == run["parameters"]
    for seed in cfg["seeds"]:
        paired = [r for r in runs if r["group"] == "optimizers" and r["seed"] == seed]
        assert len(paired) == 9
        assert len({r["initial_state_sha256"] for r in paired}) == 1
        assert len({r["minibatch_order_sha256"] for r in paired}) == 1
        deep = [r for r in runs if r["group"] == "architectures" and r["spec"]["depth"] == 12 and r["seed"] == seed]
        assert len(deep) == 3 and len({r["parameters"] for r in deep}) == 1
        assert len({r["initial_state_sha256"] for r in deep}) == 1
        assert len({r["minibatch_order_sha256"] for r in deep}) == 1
    for opt, rates in cfg["optimizers"]["learning_rates"].items():
        scores = {str(lr): float(np.mean([r["history"][-1]["validation_ce"] for r in runs
                  if r["group"] == "optimizers" and r["optimizer"] == opt and r["lr"] == lr])) for lr in rates}
        assert scores == frozen["optimizer_choices"][opt]["mean_final_validation_ce_by_lr"]
        chosen = float(min(scores, key=lambda lr: (scores[lr], float(lr))))
        assert chosen == frozen["optimizer_choices"][opt]["lr"]
    for run in runs:
        if run["group"] != "regularization": continue
        actual = early_stop_trace([r["validation_ce"] for r in run["history"][1:]], cfg["regularization"]["patience"], cfg["regularization"]["min_delta"])
        assert actual == (run["early_epoch"], run["virtual_stop_epoch"])
    evidence["training"] = "All 51 trajectories, paired starts and minibatches, equal-size residual ablations, validation-only rate selection and causal early-stop decisions verified."
    audit = load_role("audit"); table = pd.read_csv(DROOT / "audit_metrics.csv")
    meta = read_json(DROOT / "audit_metadata.json")
    assert meta["selection_frozen_sha256"] == file_sha256(DROOT / "selection_frozen.json")
    assert meta["predictions_sha256"] == file_sha256(DROOT / "audit_predictions.npz")
    assert not meta["official_test_opened"] and len(table) == meta["rows"] == 42
    assert not table.key.duplicated().any()
    max_probability_difference, max_ce_difference = 0., 0.
    with np.load(DROOT / "audit_predictions.npz") as archive:
        assert set(archive.files) == set(table.key)
        for row in table.itertuples():
            run = next(r for r in runs if r["group"] == row.group and r["name"] == row.name and r["seed"] == row.seed)
            if row.group == "optimizers": assert row.lr == frozen["optimizer_choices"][run["optimizer"]]["lr"]
            checkpoint = ROOT / row.checkpoint_path
            assert file_sha256(checkpoint) == row.checkpoint_sha256
            model = TeachingMLP(**run["spec"])
            with np.load(checkpoint) as arrays: model.load_state_dict({k: torch.from_numpy(arrays[k]) for k in arrays.files})
            result = assess(model, torch.from_numpy(audit["x"]), torch.from_numpy(audit["y"]))
            max_probability_difference = max(max_probability_difference, float(np.max(np.abs(result["p"] - archive[row.key]))))
            max_ce_difference = max(max_ce_difference, abs(result["ce"] - row.ce))
            np.testing.assert_allclose(result["p"], archive[row.key], atol=3e-5, rtol=1e-4)
            assert abs(result["ce"] - row.ce) < 1e-4 and abs(result["accuracy"] - row.accuracy) < 1e-6
    evidence["audit"] = "42 frozen checkpoints replayed on CPU; per-review probabilities, cross-entropy and accuracy match the single post-freeze internal audit."
    ini = pd.read_csv(DROOT / "initialization.csv")
    assert len(ini) == 3 * 3 * (cfg["initialization"]["depth"] + 1)
    assert np.isfinite(ini.select_dtypes("number").to_numpy()).all()
    for seed in cfg["seeds"]:
        base = ini.loc[(ini.seed == seed) & (ini.gain == 1)].set_index("layer")
        for gain in [.1, 3.]:
            other = ini.loc[(ini.seed == seed) & (ini.gain == gain)].set_index("layer")
            for layer in base.index:
                for column, exponent in [("activation_rms", layer), ("gradient_rms", cfg["initialization"]["depth"] - layer)]:
                    expected = base.loc[layer, column] * gain ** exponent
                    assert np.isclose(other.loc[layer, column], expected, rtol=1e-8, atol=1e-40)
    evidence["initialization"] = "All gain/seed/layer measurements obey the independently checked positive-homogeneity scaling of a bias-free ReLU chain."
    quad = pd.read_csv(DROOT / "quadratic.csv")
    for name, lr in [("gd_stable", .08), ("gd_unstable", .102)]:
        rows = quad.loc[quad.condition == name]
        np.testing.assert_allclose(rows.x, 4 * (1 - .5 * lr) ** rows.step, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(rows.y, 3 * (1 - 20 * lr) ** rows.step, rtol=1e-10, atol=1e-10)
    evidence["quadratic"] = "The GD trajectories match the exact eigenvalue recurrence, including the prescribed unstable rate above 2/lambda_max."
    spec = cfg["double_descent"]; rows = pd.read_csv(DROOT / "double_descent.csv")
    assert len(rows) == len(spec["seeds"]) * len(spec["widths"]) * len(spec["lambdas"]) == 880
    assert set(rows.seed) == set(spec["seeds"]) and not rows.duplicated(["seed", "p", "lambda"]).any()
    with np.load(DROOT / "double_descent_weights.npz") as archive:
        assert set(archive.files) == set(rows.weights_key)
        for seed in spec["seeds"]:
            x, xt, y, yt, f, ft, hidden = synthetic_case(seed, spec)
            a, b = np.maximum(x @ hidden, 0), np.maximum(xt @ hidden, 0)
            for row in rows.loc[rows.seed == seed].itertuples():
                # lambda is a Python keyword; use the corresponding dataframe cell.
                lam = float(rows.loc[row.Index, "lambda"])
                phi, test = a[:, :row.p] / np.sqrt(row.p), b[:, :row.p] / np.sqrt(row.p)
                beta = archive[row.weights_key]; pred, pred_t = phi @ beta, test @ beta
                assert np.isclose(np.mean((pred - y) ** 2), row.train_mse, rtol=1e-8, atol=1e-12)
                assert np.isclose(np.mean((pred_t - yt) ** 2), row.test_mse, rtol=1e-8, atol=1e-12)
                assert np.isclose(np.mean((pred_t - ft) ** 2), row.test_signal_mse, rtol=1e-8, atol=1e-12)
                assert np.isclose(np.linalg.norm(beta), row.coefficient_norm, rtol=1e-9)
                if lam == 0 and row.p >= len(y): assert row.train_mse < 1e-15
                if row.p in [64, 512] and lam == 0:
                    np.testing.assert_allclose(beta, np.linalg.lstsq(phi, y, rcond=spec["rcond"])[0], atol=1e-7, rtol=1e-7)
                if lam > 0:
                    normal_residual = phi.T @ (phi @ beta - y) + len(y) * lam * beta
                    assert np.linalg.norm(normal_residual) < 1e-7
    evidence["double_descent"] = "All 880 conditions across 20 fixed seeds replayed from stored coefficients; exact interpolation, minimum norm and ridge normal equations checked."
    correction_path = DROOT / "numerical_audit_correction.json"
    if correction_path.exists():
        correction = read_json(correction_path)
        assert correction["trained_parameter_arrays_bit_identical"] and correction["optimizer_choices_identical"]
        if correction.get("repeated_parameter_fingerprints"):
            fingerprints = read_json(ROOT / correction["repeated_parameter_fingerprints"])
            assert len(fingerprints) == 60
            for name, expected in fingerprints.items():
                with np.load(DROOT / "runs" / name) as arrays:
                    assert set(arrays.files) == set(expected)
                    for key, digest in expected.items():
                        assert hashlib.sha256(np.ascontiguousarray(arrays[key]).tobytes()).hexdigest() == digest
        correction.update(status="passed", maximum_cpu_gpu_probability_difference_after_correction=max_probability_difference,
                          maximum_cpu_gpu_ce_difference_after_correction=max_ce_difference)
        save_json(correction_path, correction)
    save_json(DROOT / "verification.json", {"status": "passed", "evidence": evidence,
              "limitations": ["Three initialization seeds do not quantify all data-sampling uncertainty.",
              "Dense networks share a training-only TF-IDF/SVD representation; architecture results are not a CNN/transformer leaderboard.",
              "Synthetic random-feature regression is separate from IMDb; no end-to-end IMDb double-descent claim.",
              "Internal audit comes from the old training pool; it is not a new external benchmark."]})
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
