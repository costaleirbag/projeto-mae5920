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
FRAME = re.compile(r"(?:^% lab:(?P<lab>bpe|cnn)\n)?\\begin{frame}(?:\[[^\]]+\])?(?:\{(?P<title>[^{}\n]+)\})?", re.M)


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
        result.append({"url": url, "label": "Abrir referência " + label,
                       "rect": [round(100 * v, 3) for v in rect]})
    return result


def build(pdf: Path, output: Path) -> None:
    reader = PdfReader(pdf)
    frames = list(FRAME.finditer((HERE / "slides.tex").read_text(encoding="utf-8")))
    assert len(frames) == len(reader.pages), "O PDF deve ter uma página por frame."
    assert [frame["lab"] for frame in frames if frame["lab"]] == ["bpe", "cnn"]

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
    data = json.dumps({"slides": slides, "evidence": evidence}, ensure_ascii=False, separators=(",", ":"))
    template = (HERE / "apresentacao.html").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        template.replace("/* DECK_CSS */", (HERE / "apresentacao.css").read_text(encoding="utf-8"))
        .replace("/* DECK_JS */", (HERE / "apresentacao.js").read_text(encoding="utf-8"))
        .replace("/* DECK_DATA */", data.replace("<", r"\u003c")),
        encoding="utf-8",
    )
    print(f"{len(slides)} slides no HTML, com duas animações e navegação offline.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "output/pdf/slides_15_min.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "output/slides/slides_15_min_interativos.html")
    args = parser.parse_args()
    build(args.pdf, args.output)


if __name__ == "__main__":
    main()
