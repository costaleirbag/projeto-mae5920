# Pooling e ordem na TextCNN

O [primeiro ensaio de posição](diagnosticos.md#o-que-acontece-quando-guardamos-a-posição-dos-picos)
reajustou a camada final com os filtros congelados. Aqui treinamos embeddings,
filtros e camada final de ponta a ponta para cada forma de pooling. A pergunta
principal é se a CNN acerta as duas versões de um par quando a troca dos
adjetivos muda o veredito que vem no fim do texto.

## Comparação

| Pooling | Valores por filtro | Resumo |
|---|---:|---|
| Máximo global | 1 | Maior ativação de todas as janelas válidas |
| Máximos por metade, ordenados pelo valor | 2 | Menor e maior entre os máximos das duas metades |
| Máximos na ordem das metades | 2 | Máximo da primeira metade, seguido do máximo da segunda |
| Média global | 1 | Média das ativações nas janelas válidas |
| Médias na ordem das metades | 2 | Média da primeira metade, seguida da média da segunda |
| K-max ordenado | 2 | As duas maiores ativações, na ordem de ocorrência |

Todos usam o vocabulário por palavras e o recorte do estudo interno. São
6.144 resenhas de treino, 1.536 de validação e 2.048 de auditoria. Mantivemos
os três ramos de convolução com larguras 3, 4 e 5, cem filtros por ramo,
Adam a 0,001 e doze épocas. Cada treino escolhe a época de menor entropia
cruzada na validação. Em cada uma das sementes 17, 29 e 43, os pesos iniciais
dos embeddings e filtros e a ordem dos minibatches são iguais. As camadas
finais com o mesmo número de entradas começam com os mesmos pesos. Os dois
formatos com máximos por metade têm 600 entradas e controlam o tamanho da
camada final ao comparar o efeito de indicar a metade de origem.

A janela excedente vai para a primeira metade quando a contagem é ímpar.
Uma metade vazia vale zero. No K-max, empates favorecem a ocorrência mais
cedo, e um segundo pico ausente vale zero. [Código](treinar_pooling.py) ·
[Divisões e vocabulário](../results/bpe/data_manifest.json).

## Avaliações

- **Veredito no fim.** Em 128 pares sintéticos, as duas versões têm as mesmas
  palavras e vereditos opostos. O par só conta se ambas estiverem corretas.
- **Veredito no começo.** O mesmo critério em outros 128 pares sintéticos.
- **Mesmo veredito em outra posição.** Em 256 pares, o veredito fica no
  começo ou no fim. As frases usam conectivos e moldes diferentes para
  manter o mesmo rótulo. Contamos o acerto nas duas versões.
- **Edições humanas.** São 344 pares do IMDb Contrast Sets. Oito ficaram
  idênticos após o recorte. As predições também permitem conferir o acerto
  nos 336 pares com entradas distintas.
- **Auditoria.** Acurácia em 2.048 resenhas do IMDb. O controle do slide 10
  mede a mudança de margem em A e B, duas versões com as mesmas janelas.

Os checkpoints são selecionados pela validação das resenhas comuns. Os pares
sintéticos e editados já foram usados no ensaio com filtros congelados. Os
512 textos sintéticos cabem no recorte e têm sequências de IDs distintas
das usadas no treino e na validação. Os resultados aqui descrevem esse
conjunto de moldes, adjetivos e edições.
[Construção dos pares](medir_posicao.py) · [IMDb Contrast Sets](https://github.com/allenai/contrast-sets/tree/e2b87316ba7b30093bc9ffbc340e6550cab79f67/IMDb/data).

## Resultados

| Pooling | Acurácia na auditoria | Veredito no fim | Veredito no começo | Mesmo veredito em outra posição | Edições humanas |
|---|---:|---:|---:|---:|---:|
| Máximo global | 82,00% | 0,00% | 80,47% | 19,14% | 40,21% |
| Máximos por metade, ordenados pelo valor | 81,85% | 0,00% | 79,95% | 12,37% | 38,08% |
| Máximos na ordem das metades | 82,11% | 0,00% | 79,17% | 13,28% | 39,24% |
| Média global | 82,18% | 0,00% | 96,88% | 21,09% | 30,62% |
| Médias na ordem das metades | 81,79% | 0,00% | 95,05% | 18,75% | 31,01% |
| K-max ordenado | 82,63% | 0,00% | 92,19% | 9,24% | 37,60% |

A tabela mostra médias de três sementes. Depois da coluna de auditoria,
cada percentual exige acertar os dois textos do par. O máximo global
reproduziu exatamente os logits da referência publicada na auditoria.

Cada um dos 18 treinos acertou zero dos 128 pares com veredito no fim. O
K-max ordenado inverteu as duas classes na direção errada em 81,5% desses
pares, em média. Nos demais, previu a mesma classe para as duas versões.
No controle do slide 10, os máximos na ordem das metades diferenciaram
os escores de A e B nas três sementes.

Entre os 344 pares editados, os máximos na ordem das metades tiveram 39,24%
de pares corretos, ante 38,08% ao ordenar os mesmos picos por valor e 40,21%
com o máximo global. Nos 336 pares com entradas distintas após o recorte,
as taxas foram 40,18%, 38,99% e 41,17%, respectivamente. O K-max chegou à
maior acurácia da auditoria comum, com 37,60% de acerto nos pares editados.
[Métricas por semente](../results/pooling_order/summary.json) ·
[Logits e rótulos](../results/pooling_order/predictions.npz) ·
[Seleção de épocas](../results/pooling_order/selection.json).

## Cobertura do recorte e embeddings

As contagens usam os tokens da CNN por palavras antes e depois do recorte,
sobre as resenhas limpas. Cada ocorrência conta, inclusive as que recebem
o ID de desconhecido.

| Avaliação | Recorte | Resenhas cortadas | Tokens conservados |
|---|---|---:|---:|
| Teste oficial | Primeiros 400 tokens | 3.118 de 25.000 (12,47%) | 5.120.969 de 5.715.860 (89,59%) |
| Auditoria interna | Primeiras 128 palavras, com teto de 256 tokens | 1.535 de 2.048 (74,95%) | 248.422 de 484.775 (51,24%) |

Na auditoria interna, todos os prefixos tinham até 152 tokens por palavras.
O recorte controla o custo das convoluções. O comprimento da entrada pode
variar, pois o pooling produz um vetor fixo por resenha. A tabela aprendida
tem **20.000 linhas por 128 colunas**, com 19.998 entradas do vocabulário mais
`<pad>` e `<unk>`. Um lote com `B × T` IDs busca `B × T × 128` valores. A
transposição produz `B × 128 × T` para a convolução. [Configuração original](../config.json) ·
[Vocabulário](../results/sequences/vocabulary.json) · [Implementação](../src/imdb_lab/models.py#L35-L54).

## Reproduzir

Use o [ambiente do projeto e o corpus original](../docs/REPRODUCAO.md#5-restaurar-o-imdb--somente-quando-a-operação-precisar-do-corpus).
Baixe `test_original.tsv` e `test_contrast.tsv` da revisão ligada acima.
O código verifica os hashes do corpus e das edições. Use um diretório
novo para os pesos e passe o mesmo diretório às duas etapas.

```bash
python seminario/treinar_pooling.py train --source data/train.parquet --output tmp/pooling
python seminario/treinar_pooling.py audit --source data/train.parquet --contrast-dir /caminho/contrast --output tmp/pooling
```
