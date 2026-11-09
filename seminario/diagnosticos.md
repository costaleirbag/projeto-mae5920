# Discordâncias e janelas da CNN

## CNN e TF-IDF nas mesmas resenhas

Cruzamos as previsões originais pelo ID e conferimos os rótulos no teste
oficial. Uma resenha entra em “com recorte” quando a expressão regular da
CNN encontra mais de 400 tokens. Os percentuais usam o total de cada linha.
[Previsões da CNN](../results/cnn/predictions.parquet) · [Previsões do TF-IDF](../results/tfidf/predictions.parquet).

| Entrada original da CNN | Resenhas | Só TF-IDF acertou | Só CNN acertou | Ambos acertaram | Ambos erraram |
|---|---:|---:|---:|---:|---:|
| Sem recorte | 21.882 | 1.376 (6,29%) | 765 (3,50%) | 18.364 | 1.377 |
| Com recorte | 3.118 | 339 (10,87%) | 110 (3,53%) | 2.457 | 212 |
| Total | 25.000 | 1.715 (6,86%) | 875 (3,50%) | 20.821 | 1.589 |

Os acertos exclusivos do TF-IDF ficaram mais frequentes no grupo com recorte.
Os da CNN tiveram frequência semelhante nos dois grupos. Comprimento e
conteúdo variam juntos nesse cruzamento. Abrimos dois casos para ver o que
cada modelo fez. [Cruzamentos originais, incluindo BERT](../results/analysis/error_overlap.csv).

### Só TF-IDF acertou na avaliação original

`imdb_test_03531` tem 780 tokens e rótulo negativo. A resenha termina com
uma crítica explícita após o limite de 400 tokens. Mantivemos os pesos, os
tokenizadores e as temperaturas originais e trocamos a parte do texto que
cada modelo recebeu.

| Entrada | Escore positivo do TF-IDF | Escore positivo da CNN |
|---|---:|---:|
| Primeiros 400 tokens da CNN | 71,7% | 94,7% |
| Texto inteiro | 6,5% | 12,1% |

No texto inteiro, `bad`, `awful` e `waste` são os três termos que mais
empurram o logit do TF-IDF para o negativo. Na CNN com o texto inteiro,
as janelas `hate to waste` e `so bad i left` dão as maiores contribuições
negativas à margem. A troca de classe nos dois modelos mostra o efeito do
recorte nesse caso. O TF-IDF também recalcula a normalização quando recebe
o prefixo. [Decomposição e escores](dados/discordancias.json).

### Só a CNN acertou

`imdb_test_22831` tem 61 tokens e rótulo positivo. A resenha elogia a
virada da trama e a filmagem e critica o final. Ambos os modelos tiveram
acesso ao texto inteiro. A CNN deu 96,6% de escore positivo, e o TF-IDF
deu 45,5%.

Na CNN, `part is brilliant` contribui com +0,257 para a margem positiva,
enquanto `the ending doesn't make much` contribui com −0,167. No TF-IDF,
`brilliant` contribui com +0,503. `sadly` e `not one` contribuem com −0,359
e −0,337. No texto, `not one` contrasta um enredo com dois. As contribuições
do TF-IDF são coeficientes vezes valores das features. As da CNN são os
máximos dos filtros vezes os pesos da margem na camada final. Elas explicam
como os termos entram no escore de cada caso.
[Contribuições registradas](dados/discordancias.json).

## Largura e passo

A largura `k` diz quantos tokens um filtro lê. Com passo `s=1`, uma
janela começa nos índices 0, 1, 2 e seguintes. Com `s=2`, começa em 0,
2, 4 e seguintes. Para `T ≥ k`, são `⌊(T − k)/s⌋ + 1`
janelas completas. A implementação mantém uma janela com padding para
textos mais curtos que o filtro.
[Implementação da CNN](../src/imdb_lab/models.py).

### Larguras treinadas

Treinamos larguras isoladas 3 e 5 e a combinação 3/4/5. A taxa Adam 0,001,
o orçamento e a regra de seleção por validação permaneceram fixos. Cem
filtros por largura produzem 100 filtros nas arquiteturas de uma largura e
300 na de três larguras. Também variamos filtros por largura e dropout.
[Grade do estudo](../sequence_config.json) · [Curvas e resultados](../results/sequences/summary.json).

| Larguras | Filtros totais | Acurácia média na auditoria sequencial |
|---|---:|---:|
| 3 | 100 | 81,27% |
| 5 | 100 | 80,94% |
| 3, 4 e 5 | 300 | 81,95% |
| 3, 4 e 5 com 32 filtros por largura | 96 | 81,64% |

### Passo com pesos congelados

Na auditoria do estudo BPE, usamos as três CNNs por palavras, treinadas
com passo 1. Comparamos passos 1, 2 e 3 selecionando as posições válidas
antes do máximo global. Essa seleção é equivalente a aplicar os mesmos
filtros com esses passos. O passo 1 reproduziu as classes da auditoria
original, com diferença máxima de cerca de 0,000005 nos logits. As três
linhas usam as mesmas 2.048 resenhas.

| Passo | Janelas retidas | Acurácia | F1 macro | Entropia cruzada |
|---|---:|---:|---:|---:|
| 1 | 100,0% | 82,00% | 0,8200 | 0,3964 |
| 2 | 50,2% | 81,45% | 0,8140 | 0,4014 |
| 3 | 33,6% | 80,97% | 0,8085 | 0,4137 |

No slide 5, o filtro 89 com largura 3 responde 0,98 à janela
`movie was terrible`. Ao mudar para passo 2, a varredura pula essa posição
e o máximo passa a 0,40. O ensaio de inferência mede o efeito de remover
janelas de uma rede treinada com passo 1. Os filtros foram mantidos fixos.
[Slide interativo](https://costaleirbag.github.io/projeto-mae5920/#slide-5) · [Métricas por semente](dados/passo_resultados.json) · [Logits e rótulos](dados/passo_predicoes.npz).

## Repetir os diagnósticos

Use os Parquets originais e os pesos gerados pelo projeto. O cruzamento
espera os artefatos originais `tfidf/` e `cnn/`. O ensaio de passo usa os
checkpoints do estudo pareado por palavras e BPE. Os hashes publicados
identificam todos os insumos. [Restauração de dados e pesos](../docs/REPRODUCAO.md).

~~~bash
python seminario/comparar_erros.py --test /caminho/para/test.parquet --artifacts /caminho/para/artefatos-originais
python seminario/medir_passo.py --train /caminho/para/train.parquet --artifacts /caminho/para/artefatos-pareados
~~~
