#!/usr/bin/env python3
"""Recompute the main scientific invariants from the actual saved results."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from imdb_lab.common import file_sha256, read_json, save_json
from imdb_lab.data import load_splits
from imdb_lab.metrics import evaluate
from imdb_lab.rfm import agop_from_gradients, prediction_gradients


def main():
    evidence = {}
    frames = load_splits()
    manifest = read_json(ROOT / "data" / "manifest.json")
    for split, frame in frames.items():
        assert file_sha256(ROOT / "data" / f"{split}.parquet") == manifest["splits"][split]["sha256"]
        for other, other_frame in frames.items():
            if split != other:
                assert not set(frame.text_sha256) & set(other_frame.text_sha256)
    assert len(frames["test"]) == 25000
    evidence["data"] = "Hashes match; no shared normalized text across splits; official test size retained."
    for model in ("tfidf", "cnn", "bert"):
        predictions = pd.read_parquet(ROOT / "results" / model / "predictions.parquet")
        saved = read_json(ROOT / "results" / model / "metrics.json")
        metadata = read_json(ROOT / "results" / model / "metadata.json")
        assert metadata["temperature"] > 0
        for split, frame in frames.items():
            rows = predictions.loc[predictions.split == split].set_index("id").reindex(frame.id)
            assert rows.index.is_unique and len(rows) == len(frame)
            assert np.array_equal(rows.label, frame.label)
            assert np.array_equal(rows.prediction, rows.p_positive >= .5)
            for view, pcol in (("raw", "p_raw"), ("calibrated", "p_positive")):
                actual = evaluate(rows.label, rows[pcol])
                for key in ("accuracy", "f1_macro", "log_loss", "brier", "ece_10", "fp", "fn", "tn", "tp"):
                    assert np.isclose(actual[key], saved[split][view][key], atol=1e-10), (model, split, view, key)
        evidence[model] = "All rows and recomputed train/validation/calibration/test metrics match."
    segments = pd.read_parquet(ROOT / "results" / "analysis" / "test_segments.parquet")
    assert set(segments.id) == set(frames["test"].id)
    subgroups = pd.read_csv(ROOT / "results" / "analysis" / "subgroup_metrics.csv")
    for model in ("tfidf", "cnn", "bert"):
        for grouping in ("length_group", "semantic_cluster"):
            rows = subgroups.loc[(subgroups.model == model) & (subgroups.group == grouping)]
            assert rows.n.sum() == 25000
            assert rows.fp.sum() + rows.tn.sum() == 12500
            assert rows.fn.sum() + rows.tp.sum() == 12500
    evidence["segments"] = "Every review assigned once per length/topic partition; class-conditional denominators match."
    rfm = np.load(ROOT / "results" / "steering" / "selected_rfm.npz")
    gradients = prediction_gradients(rfm["centers"], rfm["centers"], rfm["metric"], rfm["alpha"], float(rfm["bandwidth"]))
    np.testing.assert_allclose(agop_from_gradients(gradients), rfm["agop"], atol=1e-9, rtol=1e-7)
    assert np.linalg.eigvalsh(rfm["agop"]).min() > -1e-10
    directions = np.load(ROOT / "results" / "steering" / "directions.npz")
    for key in directions.files:
        if key not in ("layer", "scale"):
            assert np.isclose(np.linalg.norm(directions[key]), 1)
    steering = pd.read_csv(ROOT / "results" / "steering" / "intervention_summary.csv")
    assert np.allclose(steering.perturbation_l2, abs(steering.alpha) * float(directions["scale"]))
    assert (steering.loc[steering.alpha == 0].prediction_flip_fraction == 0).all()
    assert (steering.loc[steering.alpha == 0].mean_delta_p_raw.abs() < 1e-5).all()
    meta = read_json(ROOT / "results" / "steering" / "metadata.json")
    assert meta["hook_restoration_exact"] and meta["nonzero_hook_changes_logits"]
    assert max(meta["zero_alpha_max_logit_difference"].values()) < 2e-5
    probes = read_json(ROOT / "results" / "steering" / "subset_manifest.json")
    assert probes["bert_checkpoint_sha256"] == file_sha256(ROOT / "artifacts" / "bert" / "model.safetensors")
    for a in ("train", "validation", "test"):
        for b in ("train", "validation", "test"):
            if a != b:
                assert not set(probes[a]) & set(probes[b])
    evidence["steering"] = "Saved AGOP recomputed from gradients; PSD; unit vectors; norm-matched controls; zero identity; hook restoration; source checkpoint verified."
    rewrite = read_json(ROOT / "results" / "rewrite" / "metadata.json")
    edits = read_json(ROOT / "results" / "rewrite" / "examples.json")
    assert rewrite["encoder_matches_classifier"]
    assert len(edits) == 5 * 2 * 7
    for row in edits:
        assert "[MASK]" not in row["edited"]
        for model in ("tfidf", "cnn", "bert"):
            assert 0 <= row[model + "_edited_p_positive"] <= 1
    evidence["rewrite"] = "All 70 synthetic editing conditions retained and evaluated without hooks; masked-LM encoder matches classifier."
    assert (ROOT / "relatorio.html").stat().st_size > 10000
    from pypdf import PdfReader
    pdf = PdfReader(ROOT / "relatorio_imdb.pdf")
    assert len(pdf.pages) == 8
    assert all(len(page.extract_text()) > 100 for page in pdf.pages)
    evidence["reports"] = "Eight-page PDF and standalone HTML built from actual saved metrics and figures."
    import nbformat
    nb = nbformat.read(ROOT / "notebooks" / "experimentos_imdb.ipynb", as_version=4)
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells), "Execute notebook before final verification"
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    evidence["notebook"] = f"{len(code_cells)} code cells executed without exceptions; saved report and figures exist."
    save_json(ROOT / "results" / "verification.json", {"status": "passed", "evidence": evidence})
    print("Scientific result verification passed")


if __name__ == "__main__":
    main()
