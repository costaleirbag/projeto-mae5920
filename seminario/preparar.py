"""Gera figuras vetoriais e tabelas LaTeX a partir das evidências salvas."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGURES = HERE / "figuras"
DATA = HERE / "dados"
INK, TEAL, RUST, BLUE, GRID = "#142637", "#176C69", "#9C4028", "#355E94", "#D6E0E9"
SOURCES = (
    "results/comparison.csv",
    "results/tfidf/metadata.json",
    "results/cnn/metadata.json",
    "results/bert/metadata.json",
    "results/cnn/metrics.json",
    "results/analysis/subgroup_metrics.csv",
    "results/sequences/summary.json",
    "results/sequences/presentation_data.json",
    "results/bpe/summary.json",
    "results/bpe/audit_metrics.json",
    "seminario/casos.json",
    "seminario/dados/pares_resultados.json",
)


def read_json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict]:
    with (ROOT / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def pt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 2) -> str:
    return pt(100 * value, digits) + r"\%"


def row(*cells) -> str:
    return " & ".join(map(str, cells)) + r" \\"


def macro(name: str, body: str) -> str:
    return "\\newcommand{\\" + name + "}{" + body + "}\n"


def style_plots() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 14,
        "axes.labelsize": 12, "legend.fontsize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "grid.color": GRID,
        "axes.axisbelow": True, "pdf.fonttype": 42,
    })


def save(figure, name: str) -> None:
    figure.savefig(FIGURES / name, bbox_inches="tight", pad_inches=0.08,
                   metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def learning_curves() -> None:
    curves = read_json("results/sequences/presentation_data.json")["curves"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 3.6), layout="constrained")
    for key, name, color, linestyle in [
        ("cnn_sgd", "SGD · 0,3", BLUE, "--"),
        ("cnn_momentum", "Momentum · 0,1", RUST, "-."),
        ("cnn_adam", "Adam · 0,001", TEAL, "-"),
    ]:
        data = curves[key]["validation_ce"]
        left.plot(data["epochs"], data["mean"], color=color, lw=2.1, ls=linestyle, label=name)
        left.fill_between(data["epochs"], data["min"], data["max"], color=color, alpha=.10)
    left.set_title("Validação por otimizador", loc="left")
    left.set_ylabel("Entropia cruzada")
    left.legend(frameon=False)
    for key, name, color, linestyle in [
        ("train_ce", "Treino", BLUE, "--"),
        ("validation_ce", "Validação", TEAL, "-"),
    ]:
        data = curves["cnn_adam"][key]
        right.plot(data["epochs"], data["mean"], color=color, lw=2.2, ls=linestyle, label=name)
        right.fill_between(data["epochs"], data["min"], data["max"], color=color, alpha=.10)
    right.axvspan(5, 6, color=RUST, alpha=.12, label="Épocas escolhidas")
    right.set_title("Treino com Adam", loc="left")
    right.legend(frameon=False)
    for axis in (left, right):
        axis.set_xlabel("Época")
        axis.set_xticks([0, 2, 4, 6, 8, 10, 12])
        axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: pt(x, 1)))
        axis.grid(axis="y", alpha=.85)
    save(fig, "otimizacao.pdf")


def original_diagnostics() -> None:
    subgroup = [r for r in read_csv("results/analysis/subgroup_metrics.csv")
                if r["group"] == "length_group"]
    reliability = read_json("results/cnn/metrics.json")["test"]
    fig, (lengths, calibration) = plt.subplots(1, 2, figsize=(10.4, 3.65), layout="constrained")
    for key, label, color, marker in [
        ("tfidf", "TF-IDF + logística", BLUE, "o"),
        ("cnn", "CNN", TEAL, "s"),
        ("bert", "BERT", RUST, "^"),
    ]:
        rows = [r for r in subgroup if r["model"] == key]
        assert [int(r["n"]) for r in rows] == [3246, 14624, 5285, 1845]
        lengths.plot(range(4), [100 * float(r["accuracy"]) for r in rows],
                     label=label, color=color, marker=marker, lw=2.1)
    lengths.set_xticks(range(4), ["≤100", "101–250", "251–500", ">500"])
    lengths.set_ylim(78, 96)
    lengths.set_title("Acurácia por comprimento", loc="left")
    lengths.set_xlabel("Palavras no texto limpo")
    lengths.set_ylabel("Acurácia (%)")
    lengths.yaxis.set_major_formatter(FuncFormatter(lambda x, _: pt(x, 1)))
    lengths.legend(frameon=False, loc="lower left")
    lengths.grid(axis="y")
    calibration.plot([0, 1], [0, 1], color="#8190A0", ls=":", lw=2, label="Ideal")
    for key, name, color, marker in [
        ("reliability_raw", "CNN bruta", RUST, "o"),
        ("reliability_calibrated", "Com temperatura", TEAL, "x"),
    ]:
        bins = [r for r in reliability[key] if r["n"]]
        calibration.plot([r["mean_p_positive"] for r in bins],
                         [r["fraction_positive"] for r in bins], label=name,
                         color=color, marker=marker, ms=5, lw=1.5)
    calibration.set(xlim=(0, 1), ylim=(0, 1), xlabel="Probabilidade média prevista",
                    ylabel="Fração de rótulos positivos")
    calibration.set_title("Calibração da CNN", loc="left")
    calibration.legend(frameon=False, loc="upper left")
    for axis in (calibration.xaxis, calibration.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda x, _: pt(x, 1)))
    save(fig, "diagnostico_oficial.pdf")


def paired_bpe() -> None:
    rows = read_json("results/bpe/audit_metrics.json")["metrics"]
    pairs = {(r["kind"], r["seed"]): 100 * r["accuracy"] for r in rows}
    fig, axis = plt.subplots(figsize=(5.3, 2.85), layout="constrained")
    for seed, color, marker in [(17, TEAL, "o"), (29, RUST, "s"), (43, BLUE, "^")]:
        axis.plot([0, 1], [pairs["word", seed], pairs["bpe", seed]],
                  color=color, lw=2.1, marker=marker, ms=6, label=str(seed))
    axis.set_xticks([0, 1], ["Palavras", "BPE"])
    axis.set_xlim(-.18, 1.18)
    axis.set_ylim(81.0, 82.55)
    axis.set_ylabel("Acurácia (%)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: pt(x, 1)))
    axis.legend(title="Semente", frameon=False, loc="upper right", ncols=3)
    axis.grid(axis="y")
    save(fig, "bpe_pareado.pdf")


def tables() -> str:
    comparisons = read_csv("results/comparison.csv")
    baseline_names = {"tfidf": "TF-IDF + logística", "cnn": "CNN", "bert": "BERT"}
    splits = {"train": "Treino", "validation": "Validação", "test": "Teste"}
    split_rows = []
    for split, split_label in splits.items():
        if split_rows:
            split_rows.append(r"\addlinespace[2pt]")
        for key, name in baseline_names.items():
            r = next(x for x in comparisons if x["model"] == key and x["split"] == split
                     and x["probabilities"] == "raw")
            split_rows.append(row(split_label, name, pct(float(r["accuracy"])),
                                  pt(float(r["f1_macro"]), 4), pt(float(r["log_loss"]), 4)))
    calibrated_rows = []
    cost_rows = []
    for key, name in baseline_names.items():
        r = next(x for x in comparisons if x["model"] == key and x["split"] == "test"
                 and x["probabilities"] == "calibrated")
        calibrated_rows.append(row(name, pct(float(r["accuracy"])), pt(float(r["f1_macro"]), 4),
                                   pt(float(r["log_loss"]), 4)))
        meta = read_json(f"results/{key}/metadata.json")
        device = {"tfidf": "CPU do Apple M1", "cnn": "Apple M1 (MPS)",
                  "bert": "NVIDIA GB10 (CUDA)"}[key]
        cost_rows.append(row(name, device, pt(meta["training_seconds"], 1),
                             pt(meta["all_splits_inference_seconds"], 2)))
    sequences = read_json("results/sequences/summary.json")
    by_key = {r["key"]: r for r in sequences["groups"]}
    sequence_rows = [
        row(r["label"], pct(r["accuracy_mean"]), pt(r["f1_macro_mean"], 4), pt(r["ce_mean"], 4))
        for r in (by_key["cnn_adam"], by_key["bert_full"], by_key["distilbert_full"], by_key["modernbert_full"])
    ]
    baseline = sequences["baseline"]
    sequence_rows[0] = sequence_rows[0].replace("Adam", "CNN + Adam", 1)
    sequence_rows.insert(0, row("TF-IDF + logística", pct(baseline["accuracy"]),
                                pt(baseline["f1_macro"], 4), pt(baseline["ce"], 4)))
    regime_rows = [
        row(label, pct(by_key[family + "_probe"]["accuracy_mean"]),
            pct(by_key[family + "_full"]["accuracy_mean"]))
        for family, label in [("bert", "BERT-base"), ("distilbert", "DistilBERT"),
                              ("modernbert", "ModernBERT-base")]
    ]
    bpe = read_json("results/bpe/summary.json")
    bpe_rows = []
    for r in bpe["groups"]:
        stats = bpe["tokenization"][r["kind"]]["audit"]
        bpe_rows.append(row(r["label"], pct(r["accuracy_mean"]), pt(r["f1_macro_mean"], 4),
                            pt(r["ce_mean"], 4), pct(stats["unknown_rate"]),
                            pt(stats["mean_tokens"], 1)))
    cases = read_json("seminario/dados/pares_resultados.json")
    assert hashlib.sha256((HERE / "casos.json").read_bytes()).hexdigest() == cases["casos_sha256"]
    pair_rows = []
    for item in cases["pares"]:
        word, subwords = [item["resultados"][name] for name in ("Palavras", "BPE")]
        pair_rows.append(row(item["titulo"], pct(word["media_p_a"], 1), pct(word["media_p_b"], 1),
                             pct(subwords["media_p_a"], 1), pct(subwords["media_p_b"], 1),
                             f"{word['mudancas_de_classe']}/3 e {subwords['mudancas_de_classe']}/3"))
    main_pair = cases["pares"][0]
    negation = next(case for case in cases["pares"] if case["id"] == "negacao")
    fallback = []
    negation_scores = {}
    for name, result in negation["resultados"].items():
        seed_17 = next(r for r in result["predicoes"] if r["semente"] == 17)
        negation_scores[name] = seed_17["a"]["p_positiva"]
        fallback.append(row(name, pct(seed_17["a"]["p_positiva"], 1),
                            pct(seed_17["b"]["p_positiva"], 1)))
    props = {
        "LinhasOriginal": "\n".join(split_rows),
        "LinhasTeste": "\n".join(calibrated_rows),
        "LinhasCusto": "\n".join(cost_rows),
        "LinhasSeq": "\n".join(sequence_rows),
        "LinhasRegimes": "\n".join(regime_rows),
        "LinhasBPE": "\n".join(bpe_rows),
        "LinhasPares": "\n".join(pair_rows),
        "LinhasDemo": "\n".join(fallback),
        "TextoA": main_pair["a"], "TextoB": main_pair["b"],
        "ExemploCobertura": cases["cobertura"]["texto"],
        "PalavrasA": pct(main_pair["resultados"]["Palavras"]["media_p_a"], 1),
        "PalavrasB": pct(main_pair["resultados"]["Palavras"]["media_p_b"], 1),
        "BPEA": pct(main_pair["resultados"]["BPE"]["media_p_a"], 1),
        "BPEB": pct(main_pair["resultados"]["BPE"]["media_p_b"], 1),
        "NegacaoPalavrasA": pct(negation_scores["Palavras"], 1),
        "NegacaoBPEA": pct(negation_scores["BPE"], 1),
    }
    return "% Gerado por preparar.py a partir das evidências publicadas.\n" + "".join(
        macro(name, value) for name, value in props.items()
    )


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    style_plots()
    learning_curves()
    original_diagnostics()
    paired_bpe()
    (DATA / "tabelas.tex").write_text(tables(), encoding="utf-8")
    provenance = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    (DATA / "fontes.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Três figuras vetoriais e tabelas LaTeX atualizadas.")


if __name__ == "__main__":
    main()
