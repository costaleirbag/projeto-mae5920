"""Inferência local com os checkpoints de palavras e BPE do seminário."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from string import Template
import sys
from urllib.parse import parse_qs

import torch
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from imdb_lab.bpe_study import canonicalize
from imdb_lab.common import file_sha256, read_json
from imdb_lab.data import clean_text
from imdb_lab.models import PadBatch, TextCNN, Vocabulary
from imdb_lab.sequences import excerpt


@dataclass
class FrozenCNN:
    label: str
    seed: int
    epoch: int
    checkpoint_sha256: str
    model: TextCNN
    tokenizer: Vocabulary | Tokenizer
    prefix_words: int
    token_cap: int

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)
        prefix = excerpt(cleaned, self.prefix_words)
        if isinstance(self.tokenizer, Vocabulary):
            ids = self.tokenizer.encode(prefix, self.token_cap)
            words = canonicalize(prefix).split()
            tokens = [word if index != 1 else "<unk>" for word, index in zip(words, ids)] or ["<unk>"]
        else:
            encoded = self.tokenizer.encode(canonicalize(prefix))
            ids = encoded.ids[:self.token_cap] or [1]
            tokens = encoded.tokens[:self.token_cap] or ["<unk>"]
        batch, _ = PadBatch("cnn")([(ids, 0)])
        logits = self.model(**batch)[0]
        return {
            "modelo": self.label,
            "semente": self.seed,
            "epoca": self.epoch,
            "classe": ("negativa", "positiva")[int(logits.argmax())],
            "p_positiva": float(logits.softmax(dim=0)[1]),
            "logits": logits.tolist(),
            "tokens": tokens,
            "ids": ids,
            "palavras_no_prefixo": len(prefix.split()),
        }


def load_models(artifacts: Path, seed: int) -> tuple[FrozenCNN, FrozenCNN]:
    config = read_json(ROOT / "sequence_config.json")
    study = read_json(ROOT / "bpe_config.json")
    vocab_file = ROOT / "results/sequences/vocabulary.json"
    bpe_file = ROOT / "results/bpe/tokenizer.json"
    expected_vocab = read_json(ROOT / "results/sequences/data_manifest.json")["vocabulary_sha256"]
    expected_bpe = read_json(ROOT / "results/bpe/tokenizer_training.json")["tokenizer_sha256"]
    for file, expected in [(vocab_file, expected_vocab), (bpe_file, expected_bpe)]:
        if file_sha256(file) != expected:
            raise ValueError(f"Tokenizer diferente da versão do experimento: {file.name}")
    vocabulary = Vocabulary(read_json(vocab_file))
    bpe = Tokenizer.from_file(str(bpe_file))
    records = [
        ("Palavras", "sequences", f"{study['word_reference']}__s{seed}", vocabulary),
        ("BPE", "bpe", f"cnn_bpe20k__s{seed}", bpe),
    ]
    models = []
    for label, family, run_id, tokenizer in records:
        record = read_json(ROOT / "results" / family / "runs" / f"{run_id}.json")
        checkpoint = record["checkpoint"]
        relative_path = Path(checkpoint["path"]).relative_to("artifacts")
        path = artifacts / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint ausente em {path}. Consulte seminario/README.md.")
        if file_sha256(path) != checkpoint["sha256"]:
            raise ValueError(f"Checkpoint diferente do congelado: {path.name}")
        spec = record["spec"]
        size = len(vocabulary.token_to_id) if label == "Palavras" else bpe.get_vocab_size()
        model = TextCNN(size, spec["embedding_dim"], spec["filters"], spec["kernels"], spec["dropout"])
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        models.append(FrozenCNN(
            label, seed, record["best_epoch"], checkpoint["sha256"], model, tokenizer,
            config["common_input"]["whitespace_words"], config["common_input"]["token_cap"],
        ))
    return models[0], models[1]


def classify(models: tuple[FrozenCNN, FrozenCNN], text: str) -> list[dict]:
    if not isinstance(text, str) or not canonicalize(clean_text(text)) or len(text) > 4000:
        raise ValueError("Escreva uma resenha em inglês com palavras e até 4.000 caracteres.")
    return [model.predict(text) for model in models]


def serve(models: tuple[FrozenCNN, FrozenCNN], port: int) -> None:
    template = Template((HERE / "demo.html").read_text(encoding="utf-8"))
    cases = read_json(HERE / "casos.json")["pares"][0]

    class Handler(BaseHTTPRequestHandler):
        def send(self, content: str, kind: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{kind}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def page(self, text: str, result: str = "", status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send(template.substitute(
                review=escape(text), result=result, example_a=escape(cases["a"]),
                example_b=escape(cases["b"]), seed=models[0].seed,
            ), "text/html", status)

        def do_GET(self) -> None:
            if self.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.page(cases["a"])

        def do_POST(self) -> None:
            if self.path not in {"/", "/predict"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            text = ""
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise ValueError("O corpo da requisição deve ter de 1 a 32.768 bytes.")
                body = self.rfile.read(length).decode("utf-8")
                if self.path == "/predict":
                    payload = json.loads(body)
                    text = payload.get("text", "") if isinstance(payload, dict) else ""
                else:
                    text = parse_qs(body).get("text", [""])[0]
                predictions = classify(models, text)
            except ValueError as error:
                if self.path == "/predict":
                    self.send(json.dumps({"erro": str(error)}, ensure_ascii=False),
                              "application/json", HTTPStatus.BAD_REQUEST)
                else:
                    self.page(text, f'<p role="alert">{escape(str(error))}</p>', HTTPStatus.BAD_REQUEST)
                return
            if self.path == "/predict":
                self.send(json.dumps(predictions, ensure_ascii=False), "application/json")
                return
            rows = []
            for prediction in predictions:
                tokens = " · ".join(escape(token) for token in prediction["tokens"])
                score = f"{prediction['p_positiva']:.1%}".replace(".", ",")
                rows.append(f"<tr><th>{prediction['modelo']}</th><td>{prediction['classe']}</td>"
                            f"<td>{score}</td><td><code>{tokens}</code></td></tr>")
            table = ('<h2>Predição para esta resenha</h2><table><thead><tr><th>Tokenizer</th>'
                     '<th>Classe</th><th>Escore positivo</th><th>Tokens usados</th></tr></thead>'
                     '<tbody>' + "".join(rows) + "</tbody></table>")
            self.page(text, table)

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Demonstração local em http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seed", type=int, default=17, choices=[17, 29, 43])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--text", help="Emite a predição em JSON e encerra.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        models = load_models(args.artifacts, args.seed)
        if args.text is None:
            serve(models, args.port)
        else:
            print(json.dumps(classify(models, args.text), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
