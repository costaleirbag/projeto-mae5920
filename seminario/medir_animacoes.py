"""Salva as fusões BPE e as respostas de filtros da TextCNN para os slides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from demo import HERE, ROOT, load_models
from imdb_lab.common import file_sha256, read_json
from imdb_lab.models import PadBatch


def bpe_steps(tokenizer, word: str) -> dict:
    model = json.loads(tokenizer.to_str())["model"]
    suffix = model["end_of_word_suffix"]
    ranks = {tuple(pair): i + 1 for i, pair in enumerate(model["merges"])}
    pieces = [*word[:-1], word[-1] + suffix]
    steps = [{"pecas": pieces.copy()}]
    while True:
        pairs = list(zip(pieces, pieces[1:]))
        eligible = [(ranks[pair], index) for index, pair in enumerate(pairs) if pair in ranks]
        if not eligible:
            break
        rank, index = min(eligible)
        left, right = pairs[index]
        pieces[index:index + 2] = [left + right]
        steps.append({"pecas": pieces.copy(), "par": [left, right], "indice": index, "regra": rank})
    assert pieces == tokenizer.encode(word).tokens
    return {"palavra": word, "sufixo": suffix, "etapas": steps}


@torch.inference_mode()
def cnn_trace(frozen, texts: dict[str, str]) -> dict:
    cases = {}
    responses = {}
    for name, text in texts.items():
        prediction = frozen.predict(text)
        ids = prediction["ids"]
        inputs, _ = PadBatch("cnn")([(ids, 0)])
        embedded = frozen.model.embedding(inputs["input_ids"]).transpose(1, 2)
        responses[name] = {width: conv(embedded)[0, :, :len(ids) - width + 1]
                           for width, conv in zip(frozen.model.kernels, frozen.model.convs)}
        cases[name] = {"tokens": prediction["tokens"], "logits": prediction["logits"],
                       "classe": prediction["classe"], "p_positiva": prediction["p_positiva"],
                       "janelas": {}}
    for width in frozen.model.kernels:
        example = responses["A"][width].relu()
        # A maior amplitude torna a passagem da janela mais legível. O filtro fica fixo em B.
        index = int((example.amax(dim=1) - example.amin(dim=1)).argmax())
        for name, case in cases.items():
            raw = responses[name][width][index]
            activated = raw.relu()
            case["janelas"][str(width)] = {"filtro": index, "pre_relu": raw.tolist(),
                                          "ativacoes": activated.tolist(), "max_indice": int(activated.argmax())}
    return {"semente": frozen.seed, "checkpoint_sha256": frozen.checkpoint_sha256,
            "larguras": list(frozen.model.kernels), "casos": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=HERE / "dados/animacoes.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    word, bpe = load_models(args.artifacts, 17)
    cases = read_json(HERE / "casos.json")
    contrast = next(pair for pair in cases["pares"] if pair["id"] == "contraste")
    data = {
        "bpe": {**bpe_steps(bpe.tokenizer, "pallbearer"), "frase": cases["cobertura"],
                "tokenizer_sha256": file_sha256(ROOT / "results/bpe/tokenizer.json")},
        "cnn": cnn_trace(word, {"A": contrast["a"], "B": contrast["b"]}),
    }
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
