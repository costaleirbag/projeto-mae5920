"""Atualiza os dados das figuras estáticas do relatório."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def macro(name: str, text: str) -> str:
    return rf"\newcommand{{\{name}}}{{{text}}}"


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}".replace(".", ",") + r"\%"


def position_rows(results: dict) -> str:
    recipes = {
        "original": "Máximo, cabeça original",
        "global_reajustado": "Máximo, cabeça reajustada",
        "dois_picos_sem_posicao": "Dois picos por valor",
        "dois_picos_com_posicao": "Dois picos por posição",
    }
    columns = (("audit", "accuracy"), ("synthetic", "fracao_pares"),
               ("contrast", "fracao_pares"))
    rows = []
    for key, label in recipes.items():
        metrics = results["medias"][key]
        values = [pct(metrics[role][metric]) for role, metric in columns]
        rows.append(" & ".join((label, *values)) + r" \\")
    return "\n".join(rows)


def main() -> None:
    data = HERE / "dados"
    pairs = json.loads((data / "pares_resultados.json").read_text(encoding="utf-8"))
    contrast = next(pair for pair in pairs["pares"] if pair["id"] == "contraste")
    scores = [
        rf"\linhacontraste{{{seed['semente']}}}"
        rf"{{{100 * seed['a']['p_positiva']:.5f}}}{{{100 * seed['b']['p_positiva']:.5f}}}"
        for name in ("Palavras", "BPE")
        for seed in contrast["resultados"][name]["predicoes"]
    ]
    trace = json.loads((data / "animacoes.json").read_text(encoding="utf-8"))["cnn"]
    position = json.loads((data / "posicao_resultados.json").read_text(encoding="utf-8"))

    errors = json.loads((data / "discordancias.json").read_text(encoding="utf-8"))
    short_review = next(case for case in errors["casos"] if case["id"] == "imdb_test_22831")
    lines = [
        "% Gerado por preparar_relatorio.py a partir das predições e ativações salvas.",
        macro("VotosPalavras", "%\n  ".join(scores[:3])),
        macro("VotosBPE", "%\n  ".join(scores[3:])),
        macro("LinhasPosicao", position_rows(position)),
        macro("CurtaCNN", pct(short_review["cnn_prefixo"]["p_positiva"], 1)),
        macro("CurtaTFIDF", pct(short_review["tfidf_inteiro"]["p_positiva"], 1)),
    ]
    for case in ("A", "B"):
        scan = trace["casos"][case]["janelas"]["3"]
        values = ",".join(f"{value:.6f}" for value in scan["ativacoes"])
        lines.extend((
            macro(f"Ativacoes{case}", values),
            macro(f"JanelaPico{case}", str(scan["max_indice"] + 1)),
        ))
    (data / "relatorio.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
