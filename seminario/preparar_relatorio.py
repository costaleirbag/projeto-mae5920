"""Atualiza os dados das tabelas e figuras do relatório."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def macro(name: str, text: str) -> str:
    return rf"\newcommand{{\{name}}}{{{text}}}"


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}".replace(".", ",") + r"\%"


def decimal(value, digits=0):
    return f"{value:,.{digits}f}".translate(str.maketrans(",.", ".,"))


def latex_rows(rows):
    return "\n".join(" & ".join(row) + r" \\" for row in rows)


def length_macros(data):
    lengths = json.loads((data / "comprimentos.json").read_text(encoding="utf-8"))
    bert = json.loads((data / "recorte_bert/resumo.json").read_text(encoding="utf-8"))
    populations = [
        ("Resenha no treino", lengths["lengths"]["train"]["review_words"]),
        ("Resenha no teste", lengths["lengths"]["test"]["review_words"]),
        ("Frase no teste*", lengths["lengths"]["test"]["sentence_words"]),
    ]

    def corpus_rows(columns):
        return latex_rows([
            [name, *(decimal(stats[key], 1 if key == "mean" else 0) for key in columns)]
            for name, stats in populations
        ])

    def metric_rows(models, digits):
        rows = []
        for group in lengths["bands"]:
            lo, hi = group["min_words"], group["max_words"]
            name = (r"Até 100" if lo == 0 else f"{lo} a {hi}") if hi else "Acima de 500"
            values = (group["metrics"][model] for model in models)
            cells = [" / ".join(decimal(100 * m[key], digits) for key in ("accuracy", "f1_macro"))
                     for m in values]
            rows.append([name, decimal(group["n"]), *cells])
        return latex_rows(rows)

    long = lengths["bands"][-1]
    return [
        "% Gerado por preparar_relatorio.py a partir de dados/comprimentos.json.",
        macro("LinhasDistribuicao", corpus_rows(("n", "min", "p10", "mean", "median", "p90", "max"))),
        macro("LinhasDistribuicaoSlide", corpus_rows(("mean", "median", "p90"))),
        macro("LinhasPorComprimento", metric_rows(("tfidf", "cnn", "bert", "qwen"), 2)),
        macro("LinhasPorComprimentoSlide", metric_rows(("tfidf", "cnn", "bert"), 1)),
        macro("QuantidadeLongas", decimal(long["n"])),
        macro("AcuraciaLongasCNN", pct(long["metrics"]["cnn"]["accuracy"], 2)),
        macro("AcuraciaLongasCNNIntegral", pct(long["metrics"]["cnn_integral"]["accuracy"], 2)),
        macro("BertTetoOriginal", str(bert["caps_including_specials"][0])),
        macro("BertTetoMaior", str(bert["caps_including_specials"][1])),
        macro("FUmLongasBert", pct(bert["replay"]["f1_macro"])),
        macro("FUmLongasBertMaior", pct(bert["extended"]["f1_macro"])),
    ]


def main() -> None:
    data = HERE / "dados"
    original = json.loads((HERE.parent / "data/manifest.json").read_text(encoding="utf-8"))["splits"]
    sequence_manifest = HERE.parent / "results/sequences/data_manifest.json"
    internal = json.loads(sequence_manifest.read_text(encoding="utf-8"))["roles"]
    split_rows = []
    for role, internal_role, label, purpose in (
        ("train", "train", "Treino", "Ajustar representação e pesos."),
        ("validation", "validation", "Validação", "Escolher configuração e época."),
        ("calibration", None, "Calibração", "Ajustar a temperatura."),
        ("test", "audit", "Teste/auditoria", "Medir o desempenho."),
    ):
        split = original[role]
        counts = split["label_counts"]
        internal_total = decimal(internal[internal_role]["n"]) if internal_role else "--"
        split_rows.append([label, decimal(split["n"]), decimal(counts["1"]),
                           decimal(counts["0"]), internal_total, purpose])
    pairs = json.loads((data / "pares_resultados.json").read_text(encoding="utf-8"))
    contrast = next(pair for pair in pairs["pares"] if pair["id"] == "contraste")
    scores = [
        rf"\linhacontraste{{{seed['semente']}}}"
        rf"{{{100 * seed['a']['p_positiva']:.5f}}}{{{100 * seed['b']['p_positiva']:.5f}}}"
        for name in ("Palavras", "BPE")
        for seed in contrast["resultados"][name]["predicoes"]
    ]
    trace = json.loads((data / "animacoes.json").read_text(encoding="utf-8"))["cnn"]
    errors = json.loads((data / "discordancias.json").read_text(encoding="utf-8"))
    short_review = next(case for case in errors["casos"] if case["id"] == "imdb_test_22831")
    error_rows = []
    for key, model in (("tfidf", "TF-IDF + logística"), ("cnn", "TextCNN"), ("bert", "BERT")):
        source = HERE.parent / f"results/{key}/metrics.json"
        test = json.loads(source.read_text(encoding="utf-8"))["test"]
        calibrated = test["calibrated"]
        error_rows.append([model, decimal(calibrated["fp"]), decimal(calibrated["fn"]),
                           decimal(calibrated["log_loss"], 4)])
    lines = [
        "% Gerado por preparar_relatorio.py a partir dos manifestos, das predições e das ativações salvas.",
        macro("LinhasDivisao", latex_rows(split_rows)),
        macro("PositivasTreino", pct(original["train"]["label_counts"]["1"] / original["train"]["n"], 1)),
        macro("VotosPalavras", "%\n  ".join(scores[:3])),
        macro("VotosBPE", "%\n  ".join(scores[3:])),
        macro("LinhasErros", latex_rows(error_rows)),
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
    (data / "comprimentos.tex").write_text("\n".join(length_macros(data)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
