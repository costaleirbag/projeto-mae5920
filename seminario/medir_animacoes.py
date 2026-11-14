"""Salva as fusões BPE e as respostas de filtros da TextCNN para os slides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from demo import HERE, ROOT, load_models
from imdb_lab.common import file_sha256, read_json
from imdb_lab.models import PadBatch, tokenize
from medir_pares import windows


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
def cnn_responses(frozen, text: str) -> tuple[dict, dict]:
    prediction = frozen.predict(text)
    ids = prediction["ids"]
    inputs, _ = PadBatch("cnn")([(ids, 0)])
    embedded = frozen.model.embedding(inputs["input_ids"]).transpose(1, 2)
    responses = {width: conv(embedded)[0, :, :len(ids) - width + 1]
                 for width, conv in zip(frozen.model.kernels, frozen.model.convs)}
    return prediction, responses


def cnn_trace(frozen, texts: dict[str, str]) -> dict:
    cases, responses = {}, {}
    for name, text in texts.items():
        prediction, responses[name] = cnn_responses(frozen, text)
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
    # Registra a passagem de um mesmo filtro de k=3 cujo pico em A
    # ocorre em "it was wonderful" e favorece a classe positiva.
    margin_weights = frozen.model.classifier.weight.detach().diff(dim=0)[0]
    peaks_at_end, alternate = {}, {}
    for branch, (width, raw) in enumerate(responses["A"].items()):
        n_filters = raw.shape[0]
        peaks, positions = raw.relu().max(dim=1)
        last = (positions == raw.shape[1] - 1) & (peaks > 0)
        contributions = peaks * margin_weights[branch * n_filters:(branch + 1) * n_filters]
        peaks_at_end[str(width)] = int(last.sum())
        if width == 3:
            index = int(contributions.masked_fill(~last, -torch.inf).argmax())
            assert contributions[index] > 0
            for name in texts:
                values = responses[name][width][index]
                alternate[name] = {"filtro": index, "pre_relu": values.tolist(),
                                   "ativacoes": values.relu().tolist(), "max_indice": int(values.relu().argmax())}
            alternate["contribuicao_a"] = float(contributions[index])
    return {"semente": frozen.seed, "checkpoint_sha256": frozen.checkpoint_sha256,
            "larguras": list(frozen.model.kernels), "casos": cases, "picos_fim_a": peaks_at_end,
            "filtro_fim_a": alternate}


def pooling_trace(frozen, pair: dict) -> dict:
    separator = "I have a few thoughts about this movie."
    a, b = ([block.strip() for block in pair[key].split(separator) if block.strip()] for key in ("a", "b"))
    assert len(a) == 2 and a == b[::-1]
    opinion, criticism = a
    width = 3
    focus = [tokenize(separator)[-1], *tokenize(opinion)[:2]]
    predictions, responses = zip(*(cnn_responses(frozen, pair[key]) for key in ("a", "b")), strict=True)
    for k in frozen.model.kernels:
        assert windows(predictions[0]["ids"], k) == windows(predictions[1]["ids"], k)
        assert torch.equal(responses[0][k].relu().amax(dim=1), responses[1][k].relu().amax(dim=1))
    assert predictions[0]["logits"] == predictions[1]["logits"]

    positions = []
    for prediction in predictions:
        tokens = prediction["tokens"]
        found = [start for start in range(len(tokens) - width + 1) if tokens[start:start + width] == focus]
        (position,) = found
        positions.append(position)
    assert positions[0] != positions[1]

    first, second = (response[width].relu() for response in responses)
    top = first.topk(2, dim=1).values
    eligible = ((first.argmax(1) == positions[0]) & (second.argmax(1) == positions[1])
                & (top[:, 0] > top[:, 1]))
    assert eligible.any()
    # Escolhe a resposta mais forte entre os filtros com pico único nessa janela.
    index = int(top[:, 0].masked_fill(~eligible, -1).argmax())
    cases = {}
    for name, position, values in zip(("A", "B"), positions, (first, second), strict=True):
        activations = values[index].tolist()
        midpoint = (len(activations) + 1) // 2
        cases[name] = {
            "max_indice": position, "ativacoes": activations,
            "picos_por_metade": [max(activations[:midpoint]), max(activations[midpoint:])],
        }
    assert cases["A"]["picos_por_metade"] == cases["B"]["picos_por_metade"][::-1]
    return {
        "semente": frozen.seed, "checkpoint_sha256": frozen.checkpoint_sha256,
        "largura": width, "filtro": index, "janela": focus,
        "separador": separator, "elogio": opinion, "critica": criticism,
        "casos": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=HERE / "dados/animacoes.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    word, bpe = load_models(args.artifacts, 17)
    cases = read_json(HERE / "casos.json")
    contrast = next(pair for pair in cases["pares"] if pair["id"] == "contraste")
    blocks = next(pair for pair in cases["pares"] if pair["id"] == "janelas")
    data = {
        "bpe": {**bpe_steps(bpe.tokenizer, "pallbearer"), "frase": cases["cobertura"],
                "tokenizer_sha256": file_sha256(ROOT / "results/bpe/tokenizer.json")},
        "cnn": cnn_trace(word, {"A": contrast["a"], "B": contrast["b"]}),
        "pooling": pooling_trace(word, blocks),
    }
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
