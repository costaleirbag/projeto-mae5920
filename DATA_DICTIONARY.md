# Dicionário resumido dos dados

## Dados preparados: `data/*.parquet`

| Coluna | Tipo | Significado |
|---|---|---|
| `id` | string | Identificador local determinístico: split oficial e posição na fonte. |
| `text` | string | Texto original da avaliação; não inclui nome do arquivo nem nota como campo separado. |
| `label` | inteiro 0/1 | Rótulo: 0 negativo; 1 positivo. |
| `official_split` | string | `train` ou `test`, conforme a distribuição oficial. |
| `clean_text` | string | Texto após decodificar HTML, retirar tags e normalizar espaços; caixa preservada para o BERT. |
| `text_sha256` | string | SHA-256 do texto limpo com casefold; usado na auditoria de duplicação. |
| `word_count` | inteiro | Número de segmentos separados por espaço no texto limpo. |

Os arquivos `train`, `validation`, `calibration` e `test` possuem papéis distintos.
Validação e calibração derivam exclusivamente do treino oficial. Teste é mantido
integralmente. As exclusões e hashes dos arquivos aparecem em `manifest.json`.

## Previsões: `results/<modelo>/predictions.parquet`

| Coluna | Significado |
|---|---|
| `id`, `label`, `word_count` | Chaves e atributos dos dados preparados. |
| `split`, `model` | Divisão de avaliação e modelo responsável. |
| `logit` | Logit positivo menos negativo; regressão logística usa sua função de decisão. |
| `p_raw` | Sigmoide do logit, antes de calibrar. |
| `p_positive` | Sigmoide do logit dividido pela temperatura ajustada na calibração. |
| `prediction` | Classe prevista no limiar 0,5. |
| `outcome` | TN, FP, FN ou TP; a classe positiva é a avaliação positiva. |

## Segmentos e taxas

`length_group` usa contagem de palavras, não WordPieces. `semantic_cluster` é um
grupo de similaridade atribuído por K-means aprendido no treino sem rótulos.
Os indicadores lexicais são regras heurísticas, sobrepostas e sujeitas a erro.

`false_positive_rate = FP/(FP+TN)`; `false_negative_rate = FN/(FN+TP)`.
Intervalos são de Wilson a 95%. Valores ausentes indicam denominador zero.
`small_group` identifica grupos com menos de 50 textos. Associação semântica não
é uma explicação causal automaticamente validada.

O cálculo das métricas usa probabilidades recortadas para
`[1e-7, 1-1e-7]`, preservando os valores originais nos arquivos de previsões.
Esse recorte limita a penalidade de log loss em erros extremamente confiantes;
as tabelas seguem a mesma convenção entre modelos. A temperatura é ajustada
diretamente nos logits, sem esse recorte.

## Intervenções

`alpha` é intensidade assinada, com escala comum para todas as direções
unitárias. `p_positive_raw` é o escore bruto sob intervenção, sem alegação de
calibração nesse novo regime. `input_text_unchanged=True` identifica o experimento
que só altera ativações. `edited` identifica a etapa separada que modifica palavras.
Textos neutros de demonstração são sintéticos; não há classe neutra no dataset.

## Extensão Qwen3-8B

`results/qwen/classification/predictions.parquet` usa as mesmas definições de
logit, probabilidade e outcome, sem coluna `model`. Contém validação,
calibração e todas as 25.000 avaliações oficiais de teste. Não contém previsões
do treino inteiro, porque o backbone Qwen não foi ajustado.

| Campo adicional | Significado |
|---|---|
| `allowed_label_mass` | Soma das probabilidades dos tokens `negative` e `positive` no vocabulário inteiro, antes da normalização binária. |
| `original_review_tokens` | Quantidade de tokens Qwen da avaliação antes de truncar, sem as instruções. |
| `used_review_tokens` | Quantidade nominal de tokens da fonte usados no prefixo, no máximo 768. |
| `review_truncated` | A avaliação excedeu o limite de tokens da fonte. |

`features/subset_manifest.json` fixa os IDs das amostras de treino (512),
validação (256) e teste (256). `features/activations.npz` contém essas chaves,
rótulos, logits dos dois tokens e ativações de 4.096 dimensões por bloco.

`rfm/rfm_layer_*.npz` salva centros padronizados, média, escala escalar,
coeficientes kernel e fatores da métrica/AGOP. `metric_is_identity=True` indica
a geometria euclidiana completa; caso contrário,
`M = metric_factor.T @ metric_factor`.
`AGOP = agop_factor.T @ agop_factor`.
`steering/selected_rfm.npz` guarda a RFM do bloco escolhido em validação.

`steering/interventions.parquet` tem uma linha por divisão, ID, método e alfa.
`base_logit` é o escore sem intervenção; `steered_logit` vem do mesmo texto
com a soma nas ativações. A norma nominal por token é
`abs(alpha) * scale`, comum aos métodos no mesmo bloco. O teste de
classificação com intervenção não tem o mesmo tamanho do teste completo.

### Geração e avaliação

`generation/sources.json` registra 24 avaliações oficiais e oito descrições
neutras sintéticas. O campo `source_label` é ausente/nulo nas fontes sintéticas.
`generation/generations.json` mantém as 544 saídas e seus tokens, sem exclusão
por qualidade. `evaluation/scored_generations.*` acrescenta os diagnósticos.

| Campo | Significado |
|---|---|
| `condition`, `method`, `alpha`, `target` | Condição fixa da geração; `target` descreve o sinal pretendido, mas só entra no prompt quando `method=prompt_only`. |
| `source`, `generated` | Fonte limpa integral e texto realmente gerado. |
| `uses_hook` | Se houve contexto de intervenção; RFM em intensidade zero também passa pelo contexto, sem alterar os valores. |
| `output_token_ids`, `output_tokens` | Sufixo gerado sem EOS/padding e seu comprimento. |
| `stopped_with_eos`, `hit_token_limit` | Terminação natural ou alcance do orçamento de 96 novos tokens. |
| `prompt_sha256` | Hash do prompt completo, incluindo o template do modelo. |
| `prefill_hook_calls_in_batch`, `decode_hook_calls_in_batch` | Chamadas da intervenção no prompt e nos novos tokens; são contagens do lote, repetidas nas linhas de suas fontes. |
| `<juiz>_p_positive` | Probabilidade calibrada do juiz TF-IDF, CNN ou BERT, avaliado sem hooks. Calibração no IMDb não garante calibração no domínio gerado. |
| `<juiz>_delta_from_unsteered` | Escore da saída menos escore da reescrita sem steering da mesma fonte. |
| `semantic_cosine_to_source`, `semantic_cosine_to_unsteered` | Cosseno MiniLM com fonte e reescrita de referência; não comprova fatos preservados. |
| `repeated_trigram_fraction` | Fração de trigrama repetidos nas palavras da saída. |
| `required_entities_retained` | Quantidade de entidades indicadas na fonte sintética que aparecem por correspondência textual; é uma proxy limitada. |
| `source_numeric_tokens_retained` | Quantos números da fonte reaparecem na saída; números podem ser notas de opinião. |

`generation_summary.csv` separa todas as fontes, avaliações oficiais e fontes
sintéticas. ICs são percentis de 2.000 reamostragens pareadas de fontes por
condição, sem correção para comparações múltiplas.
`generation_method_contrasts.csv` compara a separação
`[P(positivo,+4)-P(positivo,-4)]/2` da RFM com cada controle nas mesmas fontes;
no controle por instrução, usa os prompts positivo/negativo. Diferença positiva
favorece RFM somente nesse diagnóstico de polarização automática.

`qualitative_review_blind.csv` é uma planilha vazia para anotação humana
posterior; `qualitative_review_key.csv` contém a chave. As notas em
`assistant_qualitative_review.json` declaram leitura pelo assistente, sem
validação humana independente, e fixam hashes das saídas lidas.
