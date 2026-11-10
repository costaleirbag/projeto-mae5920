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
    "seminario/dados/fora_dominio_resultados.json",
    "seminario/dados/animacoes.json",
    "seminario/dados/discordancias.json",
    "seminario/dados/passo_resultados.json",
    "seminario/dados/posicao_resultados.json",
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


def tex_call(name: str, *args) -> str:
    return "\\" + name + "".join("{" + str(value) + "}" for value in args)


def style_plots() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 15,
        "axes.labelsize": 12, "legend.fontsize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.major.size": 0, "ytick.major.size": 0, "axes.titlepad": 13,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "grid.color": GRID,
        "axes.axisbelow": True, "pdf.fonttype": 42,
    })


def save(figure, name: str) -> None:
    figure.savefig(FIGURES / name, bbox_inches="tight", pad_inches=0.045, transparent=True,
                   metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def learning_curves() -> None:
    curves = read_json("results/sequences/presentation_data.json")["curves"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 3.6), layout="constrained")
    for key, name, color, linestyle in [
        ("cnn_sgd", "SGD", BLUE, "--"),
        ("cnn_momentum", "Momentum", RUST, "-."),
        ("cnn_adam", "Adam", TEAL, "-"),
    ]:
        data = curves[key]["validation_ce"]
        left.plot(data["epochs"], data["mean"], color=color, lw=2.35, ls=linestyle, label=name)
        left.fill_between(data["epochs"], data["min"], data["max"], color=color, alpha=.08)
    left.set_title("Entropia cruzada na validação", loc="left")
    left.legend(loc="upper right", frameon=False, handlelength=1.45)
    for key, name, color, linestyle in [
        ("train_ce", "Treino", BLUE, "--"),
        ("validation_ce", "Validação", TEAL, "-"),
    ]:
        data = curves["cnn_adam"][key]
        right.plot(data["epochs"], data["mean"], color=color, lw=2.4, ls=linestyle, label=name)
        right.fill_between(data["epochs"], data["min"], data["max"], color=color, alpha=.08)
        right.text(12.17, data["mean"][-1], name, color=color, va="center", fontsize=11, weight="bold")
    chosen = curves["cnn_adam"]["validation_ce"]["mean"]
    right.axvspan(4.85, 6.15, color=TEAL, alpha=.11, linewidth=0)
    right.scatter([5, 6], chosen[5:7], color=TEAL, s=31, zorder=4)
    right.text(5.5, .89, "épocas 5–6", ha="center", color=TEAL, fontsize=11, weight="bold")
    right.set_title("Treino e validação com Adam", loc="left")
    for axis, end in ((left, 12.25), (right, 14.55)):
        axis.set(xlabel="Época", xlim=(-.2, end), ylim=(-.035, .98), yticks=[0, .2, .4, .6, .8])
        axis.set_xticks([0, 2, 4, 6, 8, 10, 12])
        axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: pt(x, 1)))
        axis.grid(axis="y", alpha=.8)
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
    fig, axis = plt.subplots(figsize=(6.25, 3.05), layout="constrained")
    for seed, color, marker in [(17, TEAL, "o"), (29, RUST, "s"), (43, BLUE, "^")]:
        first, last = pairs["word", seed], pairs["bpe", seed]
        axis.plot([0, 1], [first, last], color=color, lw=2.2, marker=marker, ms=7)
        offset = {17: 0, 29: -.085, 43: .085}[seed]
        axis.plot([1, 1.06], [last, last + offset], color=color, lw=.9)
        axis.text(1.08, last + offset, str(seed), color=color, weight="bold", va="center")
    axis.set_xticks([0, 1], ["Palavras", "BPE"])
    axis.set(xlim=(-.22, 1.24), ylim=(81.0, 82.65), yticks=[81, 81.5, 82, 82.5])
    axis.set_title("Acurácia por semente (%)", loc="left")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: pt(x, 1)))
    axis.grid(axis="y")
    save(fig, "bpe_pareado.pdf")


def cnn_scan() -> None:
    example = read_json("seminario/dados/animacoes.json")["cnn"]["casos"]["A"]
    trace = example["janelas"]["3"]
    values, winner = trace["ativacoes"], trace["max_indice"]
    fig, axis = plt.subplots(figsize=(8.2, 2.0), layout="constrained")
    axis.axvspan(winner - .46, winner + 2.46, color="#D9EBE3", linewidth=0)
    centers = list(range(1, len(values) + 1))
    colors = [TEAL if i == winner else "#9BB6AB" for i in range(len(values))]
    axis.bar(centers, values, width=.63, color=colors, zorder=3)
    axis.scatter(centers, [0] * len(values), color=colors, s=19, zorder=4)
    axis.text(centers[winner], values[winner] + .05, pt(values[winner]),
              ha="center", color=TEAL, fontweight="bold", fontsize=13)
    axis.set(xlim=(-.5, len(example["tokens"]) - .5), ylim=(0, 1.18),
             xticks=range(len(example["tokens"])), xticklabels=example["tokens"], yticks=[0, .5, 1])
    axis.set_ylabel("Após ReLU", fontsize=11)
    axis.tick_params(labelsize=10)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: pt(x, 1)))
    axis.grid(axis="y", alpha=.65)
    save(fig, "cnn_varredura.pdf")


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
    calibrated_rows, chart_rows = [], []
    cost_rows = []
    for y, (key, name) in zip((31, 20, 9), baseline_names.items(), strict=True):
        r = next(x for x in comparisons if x["model"] == key and x["split"] == "test"
                 and x["probabilities"] == "calibrated")
        calibrated_rows.append(row(name, pct(float(r["accuracy"])), pt(float(r["f1_macro"]), 4),
                                   pt(float(r["log_loss"]), 4)))
        chart_rows.append(tex_call("officialpoint", y, name, float(r["accuracy"]) * 100,
                                   pct(float(r["accuracy"])), pt(float(r["f1_macro"]), 4),
                                   pt(float(r["log_loss"]), 4), "Teal" if key == "cnn" else "Quiet"))
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
    regime_chart = [
        tex_call("encoderrow", label, pct(by_key[family + "_probe"]["accuracy_mean"]),
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
    reading_names = {"positiva": "Pos.", "negativa": "Neg.", "mista": "Mista"}
    for item in cases["pares"]:
        word, subwords = [item["resultados"][name] for name in ("Palavras", "BPE")]
        reading = " / ".join(reading_names[label] for label in item["leitura"])
        pair_rows.append(row(item["titulo"], reading, pct(word["media_p_a"], 1), pct(word["media_p_b"], 1),
                             pct(subwords["media_p_a"], 1), pct(subwords["media_p_b"], 1),
                             f"{word['mudancas_de_classe']}/3 e {subwords['mudancas_de_classe']}/3"))
    main_pair = cases["pares"][0]
    negation = next(case for case in cases["pares"] if case["id"] == "negacao")
    negation_scores = {
        name: next(r["a"]["p_positiva"] for r in result["predicoes"] if r["semente"] == 17)
        for name, result in negation["resultados"].items()
    }
    external = read_json("seminario/dados/fora_dominio_resultados.json")
    external_rows = []
    for domain, record in external["dominios_rotulados"].items():
        cells = [domain]
        for name in ("Palavras", "BPE"):
            scores = record["medias"][name]
            cells.extend((pct(scores["accuracy"]), pt(scores["f1_macro"], 4), pt(scores["ce"], 4)))
        external_rows.append(row(*cells))
    demo = read_json("seminario/dados/animacoes.json")
    steps = demo["bpe"]["etapas"]
    trace = demo["cnn"]["casos"]["A"]["janelas"]["3"]
    bpe_steps = {}
    for name, step in zip(("BPEInicio", "BPEFusao", "BPEFim"), (steps[0], steps[1], steps[-1])):
        tokens = []
        for index, token in enumerate(step["pecas"]):
            command = r"\fused" if index == step.get("indice") else r"\chip"
            tokens.append(command + "{" + token + "}")
        bpe_steps[name] = "".join(tokens)
    kernels = (("3", 100, "cnn_kernel_3"), ("5", 100, "cnn_kernel_5"), ("3/4/5", 300, "cnn_base"))
    kernel_cards = [tex_call("labvalue", width, f"{filters} filtros", pct(by_key[key]["accuracy_mean"]))
                    for width, filters, key in kernels]
    stride = read_json("seminario/dados/passo_resultados.json")["medias"]
    stride_cards = [tex_call("labvalue", step, pct(scores["fracao_de_janelas"], 1) + " das janelas",
                            pct(scores["accuracy"])) for step, scores in stride.items()]
    position = read_json("seminario/dados/posicao_resultados.json")
    position_values = {
        "SemPosHumanos": pct(position["medias"]["dois_picos_sem_posicao"]["contrast"]["fracao_pares"], 1),
        "ComPosHumanos": pct(position["medias"]["dois_picos_com_posicao"]["contrast"]["fracao_pares"], 1),
        "NParesHumanos": str(position["n"]["contrast"] // 2),
    }
    disagreements = read_json("seminario/dados/discordancias.json")
    long = disagreements["casos"][0]
    excerpts = {}
    for source, name in (("prefixo", "Prefixo"), ("final", "Final")):
        excerpt = long["trechos"][source]
        excerpts["Trecho" + name] = excerpt["texto"]
        excerpts["PosTrecho" + name] = f"{excerpt['primeiro_token']}--{excerpt['ultimo_token']}"
    cut_excerpts = (
        ("Primeiros 400 tokens", "tfidf_prefixo", "cnn_prefixo"),
        ("Texto inteiro", "tfidf_inteiro", "cnn_inteiro"),
    )
    cut_cards = [
        tex_call("cutrow", label, pct(long[tfidf]["p_positiva"], 1), pct(long[cnn]["p_positiva"], 1),
                 "Teal" if tfidf.endswith("prefixo") else "Rust")
        for label, tfidf, cnn in cut_excerpts
    ]
    total = disagreements["total"]
    only_tfidf, only_cnn = total["so_tfidf_acerta"], total["so_cnn_acerta"]
    props = {
        "LinhasOriginal": "\n".join(split_rows),
        "LinhasTeste": "\n".join(calibrated_rows),
        "PontosTeste": "\n".join(chart_rows),
        "RegimesVisuais": "\n".join(regime_chart),
        "KernelsVisuais": "\n".join(kernel_cards),
        "PassosVisuais": "\n".join(stride_cards),
        "RecorteVisual": "\n".join(cut_cards),
        **excerpts,
        **position_values,
        "LinhasCusto": "\n".join(cost_rows),
        "LinhasSeq": "\n".join(sequence_rows),
        "LinhasBPE": "\n".join(bpe_rows),
        "LinhasPares": "\n".join(pair_rows),
        "LinhasForaDominio": "\n".join(external_rows),
        "SoTFIDF": f"{only_tfidf:,}".replace(",", "."),
        "SoCNN": f"{only_cnn:,}".replace(",", "."),
        "Discordancias": f"{only_tfidf + only_cnn:,}".replace(",", "."),
        "PassoUm": pct(stride["1"]["accuracy"]),
        "PassoDois": pct(stride["2"]["accuracy"]),
        "PassoTres": pct(stride["3"]["accuracy"]),
        "TextoA": main_pair["a"], "TextoB": main_pair["b"],
        "ExemploCobertura": cases["cobertura"]["texto"],
        "PalavrasA": pct(main_pair["resultados"]["Palavras"]["media_p_a"], 1),
        "PalavrasB": pct(main_pair["resultados"]["Palavras"]["media_p_b"], 1),
        "BPEA": pct(main_pair["resultados"]["BPE"]["media_p_a"], 1),
        "BPEB": pct(main_pair["resultados"]["BPE"]["media_p_b"], 1),
        "NegacaoPalavrasA": pct(negation_scores["Palavras"], 1),
        "NegacaoBPEA": pct(negation_scores["BPE"], 1),
        **bpe_steps,
        "FiltroDemo": str(trace["filtro"]),
        "MaximoDemo": pt(max(trace["ativacoes"])),
        "MaximoPassoDois": pt(max(trace["ativacoes"][::2])),
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
    cnn_scan()
    (DATA / "tabelas.tex").write_text(tables(), encoding="utf-8")
    provenance = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    (DATA / "fontes.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Figuras vetoriais e tabelas LaTeX atualizadas.")


if __name__ == "__main__":
    main()
