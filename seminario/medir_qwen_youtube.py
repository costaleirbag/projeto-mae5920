"""Aplica a classificação binária congelada do Qwen aos dois prints do seminário."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from imdb_lab.common import ROOT, file_sha256, read_json
from imdb_lab.qwen_core import (
    classification_prompts, forward_labels, generation_inputs, label_ids, load_qwen, qconfig,
)


GENERATION = {
    "num_return_sequences": 3, "do_sample": True, "temperature": .6,
    "top_p": .95, "top_k": 20, "max_new_tokens": 1536,
}


def prompts_with_thinking(tokenizer, prompts: list[str]) -> list[str]:
    # The pinned template disables thinking by prefilling an empty think block.
    example = [{"role": "user", "content": "Example."}]
    enabled = tokenizer.apply_chat_template(example, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    disabled = tokenizer.apply_chat_template(example, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    suffix = disabled.removeprefix(enabled)
    assert disabled == enabled + suffix and suffix == "<think>\n\n</think>\n\n"
    assert all(prompt.endswith(suffix) for prompt in prompts)
    return [prompt.removesuffix(suffix) for prompt in prompts]


@torch.inference_mode()
def generated_labels(model, tokenizer, prompts: list[str], *, thinking: bool, seed: int) -> list[list[dict]]:
    if thinking:
        prompts = prompts_with_thinking(tokenizer, prompts)
    inputs = generation_inputs(tokenizer, prompts, next(model.parameters()).device)
    torch.manual_seed(seed)
    output = model.generate(**inputs, **GENERATION, pad_token_id=tokenizer.pad_token_id)
    completions = output[:, inputs.input_ids.shape[1]:].tolist()
    end_ids = model.generation_config.eos_token_id
    stop = set(end_ids if isinstance(end_ids, list) else [end_ids])
    think_end, = tokenizer.encode("</think>", add_special_tokens=False)
    records = []
    for completion in completions:
        end = next((i for i, token in enumerate(completion) if token in stop), len(completion))
        tokens = completion[:end]
        after_thinking = 0
        if thinking and think_end in tokens:
            after_thinking = len(tokens) - tokens[::-1].index(think_end)
        category = None
        if not thinking or after_thinking:
            answer = tokenizer.decode(tokens[after_thinking:], skip_special_tokens=True).strip()
            category = {"positive": "positiva", "negative": "negativa"}.get(answer)
        records.append({
            "classe": category, "tokens_gerados": len(tokens), "tokens_ate_fim_do_raciocinio": after_thinking,
            "fim_do_raciocinio": bool(after_thinking) if thinking else None,
            "encerrou_com_eos": end < len(completion),
        })
    n = GENERATION["num_return_sequences"]
    assert len(records) == n * len(prompts)
    return [records[start:start + n] for start in range(0, len(records), n)]


def evaluate(source: Path) -> dict:
    comments = {item["id"]: item for item in read_json(source)["comentarios_youtube"]}
    cases = [
        {key: comments[case_id][key] for key in ("id", "origem", "texto", "rotulo_referencia")}
        for case_id in ("E", "F")
    ]
    controls = [
        {"texto": "This film is wonderful. I loved every minute of it.", "rotulo_referencia": "positiva"},
        {"texto": "This film is terrible. I hated every minute of it.", "rotulo_referencia": "negativa"},
    ]
    samples = cases + controls
    assert all(sample["rotulo_referencia"] is None for sample in cases)
    torch.set_num_threads(2)
    model, tokenizer = load_qwen()
    texts = [sample["texto"] for sample in samples]
    for sample in samples:
        sample["testes"] = {}
    for protocol, reverse in (("padrao", False), ("ordem_invertida", True)):
        prompts, metadata = classification_prompts(tokenizer, texts, reverse=reverse)
        logits, masses, _ = forward_labels(model, tokenizer, prompts)
        probabilities = torch.as_tensor(logits, dtype=torch.float64).softmax(dim=1)
        assert torch.isfinite(probabilities).all()
        for sample, prompt, info, z, mass, p in zip(
            samples, prompts, metadata, logits, masses, probabilities, strict=True,
        ):
            assert not info["review_truncated"]
            sample["testes"][protocol] = {
                "classe": "positiva" if p[1] >= .5 else "negativa",
                "escore_positivo_restrito": float(p[1]),
                "massa_dos_rotulos_no_vocabulario": float(mass),
                "logits": {"negative": float(z[0]), "positive": float(z[1])},
                "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                "tokens_da_entrada": info["original_review_tokens"],
            }
        assert all(control["testes"][protocol]["classe"] == control["rotulo_referencia"] for control in controls)

    cfg = qconfig()
    prompts, _ = classification_prompts(tokenizer, [case["texto"] for case in cases])
    for case in cases:
        case["geracao"] = {}
    for protocol, thinking in (("sem_raciocinio", False), ("com_raciocinio", True)):
        generated = generated_labels(model, tokenizer, prompts, thinking=thinking, seed=cfg["seed"])
        for case, outputs in zip(cases, generated, strict=True):
            case["geracao"][protocol] = outputs
    return {
        "modelo": cfg["model_id"], "revisao": cfg["revision"], "precisao": cfg["dtype"],
        "atencao": cfg["attention"], "transformers": version("transformers"), "torch": version("torch"),
        "protocolo": "Somente o texto. Prompt original de resenha e pesos congelados. "
                     "No teste direto comparamos os logits com e sem inverter a ordem dos rótulos. "
                     "Na geração comparamos a resposta final com raciocínio desativado e ativado.",
        "amostragem": {**GENERATION, "semente": cfg["seed"]},
        "tokens_dos_rotulos": dict(zip(("negative", "positive"), label_ids(tokenizer), strict=True)),
        "entrada_sha256": file_sha256(source),
        "configuracao_sha256": file_sha256(ROOT / "qwen_config.json"),
        "codigo_qwen_sha256": file_sha256(ROOT / "src/imdb_lab/qwen_core.py"),
        "comentarios": cases, "controles": controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "dados/qwen_youtube_resultados.json")
    args = parser.parse_args()
    result = evaluate(HERE / "dados/fora_dominio_resultados.json")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
