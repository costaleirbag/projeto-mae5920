"""Empacota a apresentação 16:9 em um HTML que abre offline."""

from __future__ import annotations

import argparse
from base64 import b64encode
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FRAME = re.compile(r"(?:^% lab:(?P<lab>[a-z]+)\n)?\\begin{frame}(?:\[[^\]]+\])?(?:\{(?P<title>[^{}\n]+)\})?", re.M)
MODEL = "src/imdb_lab/models.py"


def code_excerpt(path: str, *anchors: str, context: str = "") -> dict:
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    selected = []
    for anchor in anchors:
        matches = [index for index, line in enumerate(lines, 1) if line.lstrip().startswith(anchor)]
        if len(matches) != 1:
            raise ValueError(f"Trecho ambíguo ou ausente em {path}: {anchor}")
        index = matches[0]
        selected.append((index, lines[index - 1]))
    indent = min(len(line) - len(line.lstrip()) for _, line in selected)
    return {"path": path, "context": context,
            "lines": [{"number": index, "text": line[indent:]} for index, line in selected]}


def architecture_steps() -> list[dict]:
    return [
        {
            "label": "01 · Entrada",
            "title": "Um ID por token",
            "reason": "Os índices mantêm a ordem da frase. O padding forma lotes retangulares e os comprimentos reais seguem junto com os IDs.",
            "detail": "ID 0 para padding · ID 1 para palavras fora do vocabulário",
            "code": [code_excerpt(MODEL, "return [self.token_to_id.get", "tokens = torch.full", 'inputs["lengths"] = lengths')],
        },
        {
            "label": "02 · Embedding",
            "title": "Dos índices aos vetores",
            "reason": "Cada índice busca um vetor de 128 números aprendido durante o treino. O transpose põe as 128 dimensões no eixo de canais que a Conv1d recebe.",
            "detail": "B × T × 128 → B × 128 × T",
            "code": [code_excerpt(MODEL, "self.embedding = nn.Embedding", "x = self.embedding(input_ids)")],
        },
        {
            "label": "03 · Conv1d e ReLU",
            "title": "Três larguras leem o mesmo texto",
            "reason": "Janelas de 3, 4 e 5 cobrem trechos de tamanhos diferentes. Cada ramo tem 100 filtros. Cada filtro reaproveita seus pesos pela frase e a ReLU zera respostas negativas.",
            "detail": "Cada filtro tem 128 × k pesos e um viés. O passo 1 dá 8, 7 e 6 posições na frase A.",
            "code": [code_excerpt(MODEL, "self.convs = nn.ModuleList", "for conv, k in zip", "features = torch.relu")],
        },
        {
            "label": "04 · Máscara e máximo global",
            "title": "Uma resposta por filtro",
            "reason": "A máscara exclui as janelas que avançam sobre o padding. O máximo conserva a maior resposta válida, gera um resumo de tamanho fixo e descarta a posição do pico.",
            "detail": "Para textos mais curtos que o filtro, a primeira janela com padding é mantida.",
            "code": [code_excerpt(MODEL, "n_valid = (lengths", "keep = torch.arange",
                                  "features = features.masked_fill", "pooled.append")],
        },
        {
            "label": "05 · Concatenação e dropout",
            "title": "300 sinais por resenha",
            "reason": "Juntamos os 100 máximos de cada largura. No treino, o dropout de 0,5 zera sinais ao acaso para reduzir a dependência de poucos filtros.",
            "detail": "100 + 100 + 100 → 300. Na avaliação, o dropout fica inativo.",
            "code": [code_excerpt(MODEL, "self.dropout = nn.Dropout(dropout)", "return self.classifier(self.dropout")],
        },
        {
            "label": "06 · Camada linear",
            "title": "Dois logits, um por classe",
            "reason": "A ReLU já introduziu não linearidade. A camada densa combina os 300 sinais com pesos e um viés próprios de cada classe.",
            "detail": "Linear(300, 2) · 602 parâmetros na saída",
            "code": [code_excerpt(MODEL, "self.classifier = nn.Linear", "return self.classifier(self.dropout")],
        },
        {
            "label": "07 · Treino e inferência",
            "title": "Dos logits às probabilidades",
            "reason": "A entropia cruzada recebe os logits e o rótulo no treino. Na inferência, o softmax calcula as probabilidades e o maior logit determina a classe.",
            "detail": "No caso binário, P(+) = σ(z₊ − z₋). É a sigmoide da diferença.",
            "code": [
                code_excerpt("src/imdb_lab/sequences.py", "logits = model(**indexed(tx, indices)); loss = F.cross_entropy",
                             context="Treino"),
                code_excerpt("seminario/demo.py", '"classe": ("negativa", "positiva")[int(logits.argmax())]',
                             '"p_positiva": float(logits.softmax', context="Inferência"),
            ],
        },
    ]


def links(page) -> list[dict]:
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    result = []
    for reference in page.get("/Annots", []):
        annotation = reference.get_object()
        url = str(annotation.get("/A", {}).get("/URI", ""))
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        x0, y0, x1, y1 = map(float, annotation["/Rect"])
        rect = [x0 / width, 1 - y1 / height, (x1 - x0) / width, (y1 - y0) / height]
        label = unquote(parsed.path.rsplit("/", 1)[-1]) or parsed.netloc
        if parsed.netloc == "costaleirbag.github.io" and parsed.path == "/projeto-mae5920/" and parsed.fragment == "tf-idf":
            url, label = "#tf-idf", "slide sobre TF-IDF"
        result.append({"url": url, "label": "Abrir referência " + label,
                       "rect": [round(100 * v, 3) for v in rect]})
    return result



def contrast_example() -> dict:
    cases = json.loads((HERE / "dados/pares_resultados.json").read_text(encoding="utf-8"))
    case = next(case for case in cases["pares"] if case["id"] == "contraste")
    pattern = r"(.+ )([a-z]+), (but it was )([a-z]+)\."
    first_a, impression, last_a, verdict = re.fullmatch(pattern, case["a"]).groups()
    first_b, second_b, last_b, fourth_b = re.fullmatch(pattern, case["b"]).groups()
    assert (first_a, verdict, last_a, impression) == (first_b, second_b, last_b, fourth_b)
    models = []
    for name, result in case["resultados"].items():
        runs = result["predicoes"]
        seeds = [run["semente"] for run in runs]
        predictions = {}
        for key in ("a", "b"):
            classes = [run[key]["classe"] for run in runs]
            positive = classes.count("positiva")
            predictions[key] = {"score": result[f"media_p_{key}"],
                                "label": classes[0] if positive in (0, len(seeds)) else "dividida",
                                "agreement": max(positive, len(seeds) - positive)}
        models.append({"name": name, "seeds": seeds, "predictions": predictions})
    assert models[0]["seeds"] == models[1]["seeds"]
    reversed_runs = sum(
        run["a"]["classe"] == case["leitura"][1] and run["b"]["classe"] == case["leitura"][0]
        for result in case["resultados"].values() for run in result["predicoes"]
    )
    return {"start": first_a, "connector": last_a, "adjectives": [impression, verdict],
            "expected": dict(zip(("a", "b"), case["leitura"], strict=True)), "models": models,
            "reversed": reversed_runs}


def cut_example() -> dict:
    results = json.loads((HERE / "dados/discordancias.json").read_text(encoding="utf-8"))
    case = next(case for case in results["casos"] if case["id"] == "imdb_test_03531")
    return {
        "cap": results["limite_cnn_tokens"], "total": case["tokens_cnn"],
        "label": ("Negativa", "Positiva")[case["rotulo"]], "excerpts": case["trechos"],
        "scores": {
            scope: {model: case[f"{model}_{scope}"]["p_positiva"] for model in ("tfidf", "cnn")}
            for scope in ("prefixo", "inteiro")
        },
    }


def build(pdf: Path, output: Path) -> None:
    reader = PdfReader(pdf)
    frames = list(FRAME.finditer((HERE / "slides.tex").read_text(encoding="utf-8")))
    assert len(frames) == len(reader.pages), "O PDF deve ter uma página por frame."
    assert [frame["lab"] for frame in frames if frame["lab"]] == [
        "bpe", "cnn", "tfidf", "lengths", "contrast", "pooling", "position", "youtube", "closing", "cut",
    ]

    slides = []
    with TemporaryDirectory(prefix="mae5920-slides-") as directory:
        prefix = Path(directory) / "s"
        subprocess.run(["pdftoppm", "-scale-to", "1440", "-png", str(pdf), str(prefix)], check=True)
        images = sorted(Path(directory).glob("s-*.png"))
        for frame, page, png in zip(frames, reader.pages, images, strict=True):
            slide = {"title": frame["title"] or reader.metadata.title}
            if frame["lab"]:
                slide["interactive"] = frame["lab"]
            else:
                slide.update(text=page.extract_text(), links=links(page),
                             png=b64encode(png.read_bytes()).decode("ascii"))
            slides.append(slide)

    evidence = json.loads((HERE / "dados/animacoes.json").read_text(encoding="utf-8"))
    position_results = json.loads((HERE / "dados/posicao_resultados.json").read_text(encoding="utf-8"))
    position = {
        "pairs": position_results["n"]["contrast"] // 2,
        "rates": {
            order: position_results["medias"][key]["contrast"]["fracao_pares"]
            for order, key in (("value", "dois_picos_sem_posicao"), ("half", "dois_picos_com_posicao"))
        },
    }
    youtube_results = json.loads((HERE / "dados/qwen_youtube_resultados.json").read_text(encoding="utf-8"))
    youtube_qwen = {
        case["id"]: {mode: [answer["classe"] for answer in answers] for mode, answers in case["geracao"].items()}
        for case in youtube_results["comentarios"]
    }
    lengths = json.loads((HERE / "dados/comprimentos.json").read_text(encoding="utf-8"))
    tfidf = json.loads((HERE / "dados/tfidf_slide.json").read_text(encoding="utf-8"))
    data = json.dumps({
        "slides": slides, "evidence": evidence, "architecture": architecture_steps(),
        "contrast": contrast_example(), "position": position, "youtubeQwen": youtube_qwen,
        "cut": cut_example(), "tfidf": tfidf, "lengthBands": lengths["bands"],
    }, ensure_ascii=False, separators=(",", ":"))
    template = (HERE / "apresentacao.html").read_text(encoding="utf-8")
    for marker, file in (("ROOM", "the-room"), ("FRIDAY", "friday")):
        asset = (HERE / "figuras" / f"youtube-{file}.png").read_bytes()
        template = template.replace(f"/* YOUTUBE_{marker} */", b64encode(asset).decode("ascii"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        template.replace("/* DECK_CSS */", (HERE / "apresentacao.css").read_text(encoding="utf-8"))
        .replace("/* DECK_JS */", (HERE / "apresentacao.js").read_text(encoding="utf-8"))
        .replace("<!-- ARCH_DIAGRAM -->", (HERE / "arquitetura_slide.svg").read_text(encoding="utf-8"))
        .replace("/* DECK_DATA */", data.replace("<", r"\u003c")),
        encoding="utf-8",
    )
    print(f"{len(slides)} slides no HTML, com animações e navegação offline.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "output/pdf/slides_15_min.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "output/slides/slides_15_min_interativos.html")
    args = parser.parse_args()
    build(args.pdf, args.output)


if __name__ == "__main__":
    main()
