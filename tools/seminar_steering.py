"""Recapture as ativações e gere o trio do slide de apoio."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_qwen_improve import (
    CFG, Q, capture, load_qwen, packed_conditions, plot_source, read_json,
    rewriting_prompt, save_json, seed_all,
)


def main():
    if (ROOT / ".git").exists():
        raise RuntimeError("Execute na pasta de reprodução criada pelo notebook.")
    evidence = ROOT / "referencia/results/qwen_improved/presentation_examples.json"
    example = next(case for case in read_json(evidence)["cases"] if case["id"] == "test_plot_000")
    name = "intent_256_L28_means"
    layer = 28
    seed_all(CFG["seed"], 8)
    model, tokenizer = load_qwen()
    train = [plot_source(i, "train") for i in range(CFG["paired_train_n"])]
    pair = ("negative", "positive")
    prompts = [rewriting_prompt(tokenizer, row["source"], label) for row in train for label in pair]
    states = capture(model, tokenizer, prompts)[layer].astype(np.float64)
    difference = states[1::2].mean(0) - states[::2].mean(0)
    scale = float(np.linalg.norm(difference))
    np.savez_compressed(Q / "directions.npz", **{name: difference / scale})

    output = ROOT / "results/seminar_retrain/steering"
    folder = output / "geracoes"
    folder.mkdir(parents=True, exist_ok=True)
    conditions = []
    for row in example["outputs"]:
        condition = row["controller"].copy()
        if condition["layers"]:
            condition["layers"] = [[layer, name, scale]]
        conditions.append(condition)
    source = plot_source(0, "test")
    if source["source"] != example["source"]:
        raise ValueError("O enredo do slide diverge do gerador original.")
    rows = packed_conditions(model, tokenizer, conditions, [source], folder)
    checks = [
        {"alpha": row["gain"],
         "mesmo_prompt": row["prompt_sha256"] == saved["prompt_sha256"],
         "mesmos_tokens": row["token_ids"] == saved["token_ids"],
         "nova_saida": row["generated"], "publicado": saved["generated"]}
        for row, saved in zip(rows, example["outputs"], strict=True)
    ]
    save_json(output / "comparacao.json", {"escala": scale, "comparacao": checks})
    for check in checks:
        print(f"α={check['alpha']:+g} · mesmos tokens: {check['mesmos_tokens']}", flush=True)
        print(check["nova_saida"], flush=True)


if __name__ == "__main__":
    main()
