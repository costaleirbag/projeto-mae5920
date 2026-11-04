#!/usr/bin/env python3
"""Build the reader's metric tables from frozen results, without inference or fitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "results/qwen/evaluation/model_comparison.csv",
    "results/comparison.csv",
    "results/qwen/classification/metrics.json",
    "results/sequences/summary.json",
    "results/bpe/summary.json",
]
FIELDS = ["estudo", "condicao", "grupo", "n_avaliacao", "resumo",
          "probabilidades", "acuracia", "f1_macro", "loss_ce", "fonte"]


def read_json(name):
    return json.loads((ROOT / name).read_text())


def read_csv(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def row(study, label, group, n, summary, probabilities, accuracy, f1, ce, source):
    assert 0 <= accuracy <= 1 and 0 <= f1 <= 1 and math.isfinite(ce) and ce >= 0
    return dict(zip(FIELDS, [study, label, group, n, summary, probabilities,
                            accuracy, f1, ce, source]))


def collect():
    rows = []
    original = read_csv(SOURCES[0])
    reference = {r["model"]: r for r in read_csv(SOURCES[1])
                 if r["split"] == "test" and r["probabilities"] == "calibrated"}
    reference["qwen3_8b"] = read_json(SOURCES[2])["test"]["calibrated"]
    names = {"tfidf": "TF-IDF + regressão logística", "cnn": "CNN original",
             "bert": "BERT-base ajustado", "qwen3_8b": "Qwen3-8B congelado"}
    assert {r["model"] for r in original} == set(names)
    for r in original:
        assert int(r["n"]) == 25000
        for metric in ["accuracy", "f1_macro", "log_loss"]:
            assert math.isclose(float(r[metric]), float(reference[r["model"]][metric]), abs_tol=1e-12)
        rows.append(row("original", names[r["model"]], "classificação", 25000,
                        "execução original", "calibradas", float(r["accuracy"]),
                        float(r["f1_macro"]), float(r["log_loss"]), SOURCES[0]))
    sequences = read_json(SOURCES[3])
    baseline = sequences["baseline"]
    rows.append(row("sequencias", "TF-IDF + regressão logística", "baseline", 2048,
                    "execução determinística", "sem calibração de temperatura", baseline["accuracy"],
                    baseline["f1_macro"], baseline["ce"], SOURCES[3]))
    for r in sequences["groups"]:
        label = r["label"] if r["group"] == "encoder" else "CNN — " + r["label"]
        assert r["seeds"] == [17, 29, 43] and len(r["run_ids"]) == 3
        rows.append(row("sequencias", label, r["group"], 2048, "média de três sementes",
                        "sem calibração de temperatura", r["accuracy_mean"],
                        r["f1_macro_mean"], r["ce_mean"], SOURCES[3]))
    bpe = read_json(SOURCES[4])
    assert bpe["audit_n"] == 2048
    for r in bpe["groups"]:
        assert r["seeds"] == [17, 29, 43]
        rows.append(row("bpe", "CNN — " + r["label"], "tokenização", 2048,
                        "média de três sementes", "sem calibração de temperatura",
                        r["accuracy_mean"], r["f1_macro_mean"], r["ce_mean"], SOURCES[4]))
    assert len(rows) == 23
    return rows


def table(rows):
    lines = ["| Modelo / condição | Acurácia ↑ | F1 macro ↑ | Loss CE ↓ |",
             "|---|---:|---:|---:|"]
    for r in rows:
        values = [f"{100 * r['acuracia']:.2f}%", f"{r['f1_macro']:.4f}", f"{r['loss_ce']:.4f}"]
        lines.append("| " + r["condicao"] + " | " + " | ".join(v.replace(".", ",") for v in values) + " |")
    return "\n".join(lines)


def render(rows):
    sections = {s: [r for r in rows if r["estudo"] == s] for s in ["original", "sequencias", "bpe"]}
    document = f"""# Acurácia, F1 macro e loss em um lugar

[Início](../README.md) · [Notebook único](../notebooks/seminario_completo.ipynb) ·
[CSV consolidado, com as 23 linhas](../results/resumo_metricas.csv)

Estas tabelas leem os resultados congelados; não executam novas inferências nem
treinos. O CSV reúne as três comparações com uma coluna `estudo`. **Compare dentro
de cada estudo:** o teste oficial e as duas auditorias internas são conjuntos distintos.
Os exemplos de operações em CPU do notebook não são resultados deste benchmark.

## O que significa cada coluna

| Métrica | Como ler |
|---|---|
| Acurácia | Percentual de rótulos corretos; maior é melhor |
| F1 macro | Média do F1 de cada classe, entre 0 e 1; maior é melhor |
| Loss CE | Entropia cruzada, também chamada de log loss; menor é melhor |

O CSV conserva os números sem arredondar para apresentação. Nas linhas de três
sementes, cada métrica é a média das métricas individuais, não a métrica de um
ensemble. As faixas por semente e demais métricas estão nos arquivos de origem.

## 1. Comparação original — teste oficial, 25.000 resenhas

Uma execução por modelo, com probabilidades calibradas por temperatura em uma
divisão separada. CNN e BERT têm pesos ajustados; o Qwen usa pesos congelados e
escores de rótulos. A tabela também está na seção 12 do notebook único.

{table(sections['original'])}

Fontes: [comparação dos quatro modelos](../results/qwen/evaluation/model_comparison.csv),
[resultados originais por divisão e calibração](../results/comparison.csv) e
[métricas Qwen](../results/qwen/classification/metrics.json).

Os comprimentos de entrada variam: texto completo no TF-IDF, até 400 tokens na CNN,
192 WordPieces no BERT e 768 tokens de fonte no Qwen. A extensão Qwen reutiliza um
teste já examinado e é exploratória. Esses números não isolam arquitetura ou escala.

## 2. CNNs e encoders — auditoria interna de 2.048 resenhas

Mesmos prefixos de até 128 palavras; limite de 256 tokens nas redes. Ajuste com
6.144 exemplos e seleção com 1.536 de validação. As linhas neurais são médias de
três sementes (17, 29, 43); o TF-IDF tem uma execução determinística. As
probabilidades não têm calibração de temperatura nesta extensão.

{table(sections['sequencias'])}

**Linhas reutilizadas:** CNN + Adam, a arquitetura base 3/4/5 com cem filtros e
dropout 0,5 são os mesmos três checkpoints, mostrados em blocos diferentes.
As 16 linhas neurais desta tabela não correspondem a 16 novos conjuntos de treinos.
Ao todo, a grade dessa etapa contém 87 ajustes, incluindo as taxas candidatas.

Fontes: [CSV de comparação](../results/sequences/comparison.csv),
[resumo com escolhas e sementes](../results/sequences/summary.json) e
[protocolo](PROTOCOLO_SEQUENCIAS.md).
As receitas congelada e completa têm orçamentos e otimizadores diferentes.
Entre famílias também mudam pré-treinamento e tokenização.

## 3. Palavras versus BPE — outra auditoria interna de 2.048 resenhas

Mesmos checkpoints por palavras, três CNNs novas com BPE e uma avaliação reservada
diferente da tabela anterior. Médias das mesmas três sementes, sem calibração de
temperatura. BPE aprende regras apenas no texto de treino e fica congelado antes
do treino neural; a taxa Adam é herdada da receita por palavras, sem retuning.

{table(sections['bpe'])}

A proporção de tokens desconhecidos caiu de 3,27% para zero, sem ganho de acurácia
nesta receita fixa. Janelas da CNN passam a contar subpalavras. Não concluímos
superioridade universal de uma tokenização.

Fontes: [comparação BPE](../results/bpe/comparison.csv),
[resumo](../results/bpe/summary.json) e [protocolo BPE](PROTOCOLO_BPE.md).

## 4. Onde ficam os outros experimentos

| Estudo | Avaliação e resultados |
|---|---|
| MLPs: otimizadores, profundidade e regularização | Outra auditoria interna de 2.048 resenhas; [métricas dos 42 checkpoints](../results/didactic/audit_metrics.csv), [resumo por condição](../results/didactic/summary.json) e [guia](CONEXOES_DISCIPLINA.md) |
| Double descent | Regressão sintética: erro quadrático, norma e condicionamento; [880 condições](../results/didactic/double_descent.csv) e seção 10 do notebook |
| Steering na geração | 64 fontes e 11 condições: [resumo dos diagnósticos](../results/qwen_improved/test_summary.json), [todas as saídas e escores](../results/qwen_improved/test/scored.json) e [galeria de trios](../relatorio_qwen3.html) |

O suplemento MLP estuda trajetórias de acurácia e CE; F1 macro não foi registrado
como métrica desse estudo. Seus resultados estão nas fontes acima. Na geração do
Qwen, os diagnósticos usam o próprio modelo como juiz, sem rótulos humanos
independentes.

## Conferir ou regenerar estas tabelas

```bash
python tools/build_results_overview.py --check
# Para reconstruir os arquivos derivados a partir das mesmas fontes:
python tools/build_results_overview.py
```

O gerador cruza a comparação original com as métricas de origem e preserva os
valores das extensões. [Proveniência e hashes](../results/resumo_metricas_proveniencia.json)
identificam os arquivos usados. Os verificadores científicos descritos na
[reprodução](REPRODUCAO.md) fazem a conferência independente das previsões salvas.
"""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    provenance = {"purpose": "Reader overview derived from frozen result files, without model selection or inference",
                  "rows": len(rows), "studies": {k: len(v) for k, v in sections.items()},
                  "sources_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}}
    return {"docs/RESULTADOS.md": document, "results/resumo_metricas.csv": stream.getvalue(),
            "results/resumo_metricas_proveniencia.json": json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check existing tables without changing files")
    args = parser.parse_args()
    outputs = render(collect())
    for name, value in outputs.items():
        path = ROOT / name
        if args.check:
            assert path.read_text() == value, f"Outdated overview: {name}"
        else:
            path.write_text(value)
    print("Metric overview verified: 23 rows, 3 identified evaluation cohorts, accuracy/F1 macro/CE linked to frozen sources.")


if __name__ == "__main__":
    main()
