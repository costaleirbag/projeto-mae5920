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
cada modelo recebeu. Dentro do recorte, nos tokens 230–239, já aparece
“This film is as crassly manipulative as it is vapid…”. Só no trecho
final, nos tokens 716–724, aparece “I had to leave because it was so bad.”
[Abrir o recorte interativo](https://costaleirbag.github.io/projeto-mae5920/#recorte).

| Entrada | Escore positivo do TF-IDF | Escore positivo da CNN |
|---|---:|---:|
| Primeiros 400 tokens da CNN | 71,7% | 94,7% |
| Texto inteiro | 6,5% | 12,1% |

O experimento acrescentou de uma vez todos os 380 tokens restantes. Os
trechos mostram onde havia pistas críticas. No teste original, valem a
célula de TF-IDF com o texto inteiro e a célula de CNN com o recorte.

No texto inteiro, `bad`, `awful` e `waste` são os três termos que mais
empurram o logit do TF-IDF para o negativo. Na CNN com o texto inteiro,
as janelas `hate to waste` e `so bad i left` dão as maiores contribuições
negativas à margem. A troca de classe nos dois modelos mostra o efeito do
recorte nesse caso. O TF-IDF também recalcula a normalização quando recebe
o prefixo. [Decomposição e escores](dados/discordancias.json).

### Como o TF-IDF e a CNN representam a resenha

O baseline transforma a resenha limpa num vetor de até 80.000 palavras e
pares de palavras consecutivas do vocabulário aprendido no treino. Por
exemplo, *so bad* ativa `so`, `bad` e `so bad` quando constam no vocabulário.
A frequência usa escala logarítmica. O IDF dá mais peso a termos
encontrados em menos resenhas de treino, e o vetor é normalizado. Uma
regressão logística treinada com os rótulos soma as contribuições dos
termos e um viés. A ordem entra nos bigramas. A posição de cada ocorrência
e a ordem entre trechos distantes ficam de fora da representação.
[Código do baseline](../src/imdb_lab/baseline.py).

No par em que trocamos *terrible* e *wonderful*, as contagens de palavras
isoladas coincidem e o bigrama *terrible but* dá lugar a *wonderful but*.
Ambos os bigramas estão no vocabulário salvo. Reaplicamos o baseline
original e o escore positivo sem calibração foi de 47,93% em A e 47,86% em
B, com classe negativa nas duas. [Frases do par](casos.json) ·
[Escores do baseline](dados/discordancias.json) · [Código da comparação](comparar_erros.py).

A TextCNN aprende embeddings e filtros sobre janelas ordenadas de 3, 4 e
5 tokens. Cada filtro fornece sua maior ativação à camada linear. O
limite de 400 pertence à entrada que usamos no teste original. Ao
estender a entrada para 780, a mesma convolução percorre mais janelas.
[Configuração original](../config.json) · [TextCNN](../src/imdb_lab/models.py).

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
[Varredura interativa](https://costaleirbag.github.io/projeto-mae5920/#slide-5-varredura) · [Métricas por semente](dados/passo_resultados.json) · [Logits e rótulos](dados/passo_predicoes.npz).

## Repetir os diagnósticos

Use os Parquets originais e os pesos gerados pelo projeto. O cruzamento
espera os artefatos originais `tfidf/` e `cnn/`. O ensaio de passo usa os
checkpoints do estudo pareado por palavras e BPE. Os hashes publicados
identificam todos os insumos. [Restauração de dados e pesos](../docs/REPRODUCAO.md).

~~~bash
python seminario/comparar_erros.py --test /caminho/para/test.parquet --artifacts /caminho/para/artefatos-originais
python seminario/medir_passo.py --train /caminho/para/train.parquet --artifacts /caminho/para/artefatos-pareados
~~~

## O que acontece quando guardamos a posição dos picos?

A CNN original guarda o pico de cada filtro. No par da troca de blocos, o
pico fica igual quando a ativação muda de lugar. Mantivemos os embeddings
e as convoluções das três CNNs por palavras congelados. Para cada filtro,
dividimos as janelas válidas em duas metades pelo índice de início, deixando
uma janela a mais na primeira metade quando a contagem é ímpar. Guardamos
o máximo de cada metade. Se a segunda estiver vazia, seu pico vale zero.

Treinamos duas cabeças logísticas com 600 entradas e o mesmo número de
coeficientes. A primeira recebe os dois picos ordenados por valor, **menor
e maior**. A segunda recebe os mesmos picos na ordem do texto, **primeira
e segunda metade**. Cada cabeça escolhe a regularização por entropia
cruzada nas mesmas 1.536 resenhas de validação, após ajustar os coeficientes
em 6.144 resenhas de treino. A grade de C foi 0,001, 0,01, 0,1, 1 e 10.
Todas as cabeças selecionaram 0,01. Reajustamos também uma cabeça sobre
o máximo global, com 300 entradas, para separar o efeito de reajustar a
saída. [Código e protocolo](medir_posicao.py).

| Cabeça | Auditoria, acurácia em 2.048 resenhas | 256 pares sintéticos, ambos corretos | 344 pares de edição humana, ambos corretos |
|---|---:|---:|---:|
| Máximo global, cabeça original | 82,00% | 40,23% | 40,21% |
| Máximo global, cabeça reajustada | 81,90% | 39,19% | 40,02% |
| Dois picos por valor | 82,28% | 39,32% | 39,83% |
| Dois picos por posição | 82,29% | 39,97% | 39,15% |

Os valores são médias das sementes 17, 29 e 43. Nos pares, o acerto exige
classificar corretamente a versão positiva e a negativa. Os 256 pares
sintéticos cruzam oito adjetivos positivos, oito negativos e quatro
moldes em inglês. Em metade dos pares o veredito vem no começo, na outra
metade vem no fim. No grupo com veredito no fim, nenhuma cabeça acertou
as duas frases de um mesmo par em qualquer uma das sementes.

Para a edição humana, usamos os dois arquivos de teste do IMDb Contrast
Sets. Cada linha original vem acompanhada de uma edição que muda o
rótulo. Excluímos 144 dos 488 pares porque pelo menos uma versão coincidia
com uma entrada de treino ou validação depois da limpeza, tokenização
e recorte. Entre os 344 pares restantes, oito têm a mesma entrada visível
nas duas versões e foram mantidos. O recorte pode esconder uma edição,
e as alterações humanas envolvem vários fenômenos linguísticos.
[Gardner et al. (2020)](https://aclanthology.org/2020.findings-emnlp.117/) · [Dados na revisão avaliada](https://github.com/allenai/contrast-sets/tree/e2b87316ba7b30093bc9ffbc340e6550cab79f67/IMDb/data).

No controle com as mesmas janelas, trocar os blocos troca exatamente
os dois picos de posição em cada filtro. As três cabeças com ordem
deram margens diferentes entre A e B, preservando a classe em ambos.
As cabeças por valor deram margens iguais. A posição fica disponível
à camada final. Nesta rede congelada e com essa cabeça linear, o
acerto nos pares editados ficou semelhante, com variação entre as
sementes. Uma arquitetura treinada de ponta a ponta com outro
pooling responderia a outra pergunta.

Baixe test_original.tsv e test_contrast.tsv na revisão acima e passe a
pasta ao script junto com os dados e checkpoints do estudo pareado.
O NPZ usa a ordem do manifesto para a auditoria e a ordem dos moldes
e adjetivos do código para os pares sintéticos. Em cada par sintético,
a positiva vem primeiro. Em cada par de edição humana, vem a original
e depois a edição. contrast_tsv_rows guarda as linhas dos TSV
começando pelo cabeçalho na linha 1. Os logits das regressões têm
zero na primeira coluna e a margem na segunda. [Resultados por semente](dados/posicao_resultados.json) · [Logits e rótulos](dados/posicao_predicoes.npz).

~~~bash
python seminario/medir_posicao.py --train /caminho/para/train.parquet --contrast-dir /pasta/dos/tsv --artifacts /caminho/para/artefatos-pareados
~~~
