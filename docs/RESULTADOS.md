# Acurácia, F1 macro e loss em um lugar

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

| Modelo / condição | Acurácia ↑ | F1 macro ↑ | Loss CE ↓ |
|---|---:|---:|---:|
| TF-IDF + regressão logística | 90,14% | 0,9014 | 0,2396 |
| CNN original | 86,78% | 0,8678 | 0,3037 |
| BERT-base ajustado | 90,94% | 0,9094 | 0,2278 |
| Qwen3-8B congelado | 95,26% | 0,9526 | 0,1367 |

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

| Modelo / condição | Acurácia ↑ | F1 macro ↑ | Loss CE ↓ |
|---|---:|---:|---:|
| TF-IDF + regressão logística | 83,30% | 0,8330 | 0,3849 |
| CNN — SGD | 76,53% | 0,7631 | 0,4798 |
| CNN — SGD + momentum | 79,35% | 0,7934 | 0,4698 |
| CNN — Adam | 81,95% | 0,8194 | 0,4094 |
| CNN — 3/4/5, 100 filtros | 81,95% | 0,8194 | 0,4094 |
| CNN — Janela 3, 100 filtros | 81,27% | 0,8126 | 0,4152 |
| CNN — Janela 5, 100 filtros | 80,94% | 0,8094 | 0,4153 |
| CNN — 3/4/5, 32 filtros | 81,64% | 0,8164 | 0,4089 |
| CNN — Dropout 0,5 | 81,95% | 0,8194 | 0,4094 |
| CNN — Dropout 0 | 81,80% | 0,8180 | 0,4112 |
| CNN — Dropout 0,8 | 80,44% | 0,8043 | 0,4217 |
| BERT-base congelado | 84,52% | 0,8451 | 0,3655 |
| BERT-base ajuste completo | 88,31% | 0,8831 | 0,3027 |
| DistilBERT congelado | 83,35% | 0,8334 | 0,3839 |
| DistilBERT ajuste completo | 86,80% | 0,8678 | 0,3382 |
| ModernBERT-base congelado | 81,30% | 0,8129 | 0,4103 |
| ModernBERT-base ajuste completo | 91,24% | 0,9124 | 0,2458 |

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

| Modelo / condição | Acurácia ↑ | F1 macro ↑ | Loss CE ↓ |
|---|---:|---:|---:|
| CNN — Palavras | 82,00% | 0,8200 | 0,3964 |
| CNN — BPE 20 mil | 81,41% | 0,8140 | 0,4052 |

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
