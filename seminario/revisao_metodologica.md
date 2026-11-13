# Revisão metodológica para o seminário

## Por que ambos os vocabulários têm 20 mil entradas?

A CNN por palavras mantém os 19.998 itens mais frequentes do treino, além de
`<pad>` e `<unk>`. No BPE, configuramos 20.000 como alvo. Ele parte de caracteres
e aprende a fundir pares que aparecem pelo menos duas vezes. Neste treino,
atingiu 20.000 entradas, com 19.924 fusões. Os dois modelos têm a mesma
tabela de **20.000 × 128**, com 2.560.000 parâmetros. Esse controle permite
comparar a segmentação com a mesma capacidade no embedding. O BPE pode usar
outro alvo. As janelas com a mesma largura em tokens podem cobrir quantidades
diferentes de palavras. [Protocolo BPE](../docs/PROTOCOLO_BPE.md) ·
[Configuração](../bpe_config.json) · [Fusões registradas](../results/bpe/tokenizer_training.json).

## Como funciona o TF-IDF com regressão logística?

1. Extraímos palavras e pares de palavras consecutivas da resenha inteira.
   Em *not good*, aparecem `not`, `good` e `not good`. Em *good not*,
   as palavras isoladas são as mesmas e o bigrama muda. A ordem dentro de
   cada par fica registrada. A localização do par na resenha se perde.
2. O TF cresce com o número de ocorrências na resenha, em escala \(1+\log f\).
   O IDF reduz o peso de termos presentes em muitos textos de treino. Os
   produtos são normalizados pela norma L2 do vetor. A maioria dos valores
   é zero em cada texto. As 80.000 colunas formam o vocabulário de atributos.
3. A regressão aprende um coeficiente por coluna. Soma cada valor de TF-IDF
   vezes seu coeficiente, mais um viés. A sigmoide transforma esse escore
   em uma probabilidade para a classe positiva. A penalização L2 limita
   os pesos, com \(C=8\) selecionado entre 0,5, 2 e 8 pelo F1 macro na validação.

O par sintético da apresentação dá **47,93%** de escore positivo em A e
**47,86%** em B para o TF-IDF. A troca alterou o vetor, mas as duas classes
ficaram negativas. A presença de bigramas dá acesso a pistas locais. A
regressão soma pistas distribuídas no texto, sem uma regra que identifique
a oração que contém o veredito. [Código](../src/imdb_lab/baseline.py) ·
[Par de contraste e contribuições](dados/discordancias.json) ·
[Slide interativo](https://costaleirbag.github.io/projeto-mae5920/#tf-idf).

## Como uma regressão acertou quase 100% no treino?

A regressão é linear no vetor de atributos. Esse vetor tem 80.000 colunas
para 19.823 resenhas de treino e, na mediana, 206 valores diferentes de zero por
resenha. A fronteira linear tem flexibilidade suficiente para ajustar
muitas combinações raras de termos. Os **99,72%** foram medidos nos próprios
dados usados para ajustar os coeficientes. São 56 erros no treino, ante
**91,53%** na validação e **90,14%** no teste. A regularização continuava ativa.

| C | Acurácia no treino | Acurácia na validação |
|---:|---:|---:|
| 0,5 | 93,25% | 89,39% |
| 2 | 97,34% | 90,72% |
| 8 | 99,72% | 91,53% |

No controle com \(C=8\), embaralhamos só os rótulos do treino. O modelo
acertou **99,75% dos rótulos embaralhados no treino** e **50,08% dos
rótulos reais na validação**. O controle demonstra a capacidade de ajuste.
O teste de duplicatas também confirmou zero hashes de texto limpo
compartilhados entre treino, validação, calibração e teste. Isso cobre
duplicatas exatas após normalização. [Métricas originais](../results/tfidf/metrics.json) ·
[Controle e valores de regularização](../results/tfidf_ablation/summary.json).

## Por que o baseline se saiu melhor?

Um controle reajustou TF-IDF e logística nas mesmas divisões originais com
quatro comprimentos. O recorte usa a contagem da CNN por palavras, e o
vetorizador continua usando o seu próprio tokenizador. A representação e a
logística são reajustadas para cada linha, com \(C\) escolhido na validação.

| Texto fornecido | Só unigramas | Unigramas e bigramas |
|---|---:|---:|
| Até 256 tokens | 87,28% | 88,51% |
| Até 400 tokens | 88,37% | 89,66% |
| Até 768 tokens | 88,66% | 90,10% |
| Texto inteiro | 88,66% | 90,15% |

Com o mesmo limite de 400 tokens, a receita com bigramas ficou 2,88 pontos
acima dos 86,78% da CNN original. A receita integral ganhou 1,48 ponto ao
incluir bigramas. Esse contraste muda também a dimensão dos atributos
(40.723 para 80.000 no texto inteiro) e o \(C\) selecionado. O reajuste
integral repetiu a acurácia de validação original e diferiu em uma decisão
no teste, em outra versão do scikit-learn. Os números da apresentação
usam o baseline original de 90,14%. [Código e protocolo do controle](auditar_baseline.py) ·
[Métricas](../results/tfidf_ablation/summary.json) ·
[Margens pareadas](../results/tfidf_ablation/predictions.npz).

O TF-IDF tem acesso a todo o documento e agrega vários indícios lexicais.
A TextCNN aprende embeddings e filtros do zero. Seus filtros de 3, 4 e 5
tokens observam trechos consecutivos, e o máximo global retém só a ativação
mais alta de cada filtro. Dois textos com as mesmas janelas em outra ordem
entregam o mesmo resumo. A mesma conclusão vale para um pico que aparece
uma ou várias vezes com a mesma intensidade, desde que os outros máximos
permaneçam iguais. No par com *but*, algumas janelas mudam e as classes
invertem na direção errada. O máximo explica a invariância do primeiro
controle. Os filtros aprendidos e seus pesos na saída também participam
da inversão no segundo caso. [Controles da CNN](diagnosticos.md) ·
[Resultados anteriores de pooling](pooling_ordem.md).

## O que o Qwen mostra?

No teste oficial de 25.000 resenhas, a acurácia foi **95,26%** para
Qwen3-8B, **90,94%** para BERT ajustado, **90,14%** para TF-IDF e
**86,78%** para a CNN original. O Qwen recebe um prompt com até 768 tokens
da resenha e classifica pela diferença dos logits de `positive` e
`negative`, com o backbone congelado. É uma vantagem de **5,12 pontos**
sobre TF-IDF neste teste. Na comparação resenha a resenha, o Qwen acertou
sozinho 1.914 casos e o TF-IDF acertou sozinho 635. Pré-treino, arquitetura,
escala, objetivo e entrada diferem. O protocolo do Qwen reutiliza o teste
e não audita eventual exposição ao IMDb no pré-treino.
[Protocolo Qwen](../QWEN_EXPERIMENTO.md#1-classificação-do-teste-inteiro) ·
[Cruzamento pelo ID](../results/tfidf_ablation/summary.json).

## Por que os slides 7 e 13 têm números diferentes?

| Slide | Treino e entrada | Avaliação | Minibatches de treino |
|---|---|---|---|
| 7 | Estudo original, 19.823 textos. TF-IDF lê o texto inteiro. CNN e BERT usam até 400 tokens por palavras e 192 WordPieces, respectivamente. | 25.000 resenhas do teste oficial. Uma execução por modelo. O BERT fez 90,94%. | CNN 64, BERT 32. A regressão logística usa LIBLINEAR. |
| 13 | Extensão, 6.144 textos e o mesmo prefixo de 128 palavras para todos os encoders. | 2.048 resenhas da auditoria interna, média de três sementes. BERT com cabeça nova fez 84,52%. BERT ajustado por inteiro fez 88,31%. | 128 ao ajustar só a cabeça, 32 no ajuste completo. |

O quadro do Qwen no slide 13 cita separadamente seu resultado no teste oficial.
As duas tabelas de BERT mudam amostra, comprimento e protocolo. A diferença
entre 90,94% e 88,31% não mede o efeito de trocar o tamanho do lote.
[Configuração original](../config.json) · [Configuração da extensão](../sequence_config.json).

## Que regularização usamos?

- **CNN original.** Dropout 0,5, AdamW com `weight_decay=0.01` e checkpoint
  escolhido pelo F1 macro da validação, com paciência 2. O treino tinha
  orçamento de cinco épocas e a época 5 foi selecionada.
- **CNNs dos ensaios internos.** Dropout padrão 0,5, além dos testes com 0
  e 0,8. Adam, SGD e SGD com momentum usaram decaimento zero. A época de
  menor entropia cruzada na validação é escolhida depois de rodar as 12
  épocas. O clipping de norma 1 nas duas receitas controla atualizações
  com gradientes grandes.
- **Logística e encoders.** A logística original usou penalização L2.
  Os encoders internos ajustados de ponta a ponta usaram AdamW e
  decaimento 0,01 nos pesos elegíveis. As cabeças sobre vetores congelados
  usaram Adam com decaimento zero e dropout 0,1.

[Treino original](../src/imdb_lab/training.py) · [Treino dos ensaios](../src/imdb_lab/sequences.py).

## Por que limitar o comprimento?

Os 400 já estavam na configuração inicial. O teto limita o cálculo e a
memória, inclusive o padding até a sequência mais longa do minibatch.
O histórico guarda o valor, sem uma busca para justificar o número exato.
No teste oficial, a maior resenha tinha 2.247 tokens por palavras. O teto
de 400 cortou 3.118 de 25.000 resenhas (12,47%) e 10,41% dos tokens. O
efeito medido com os pesos originais foi de 0,50 ponto no teste completo
e 3,98 pontos nas resenhas cortadas.

O teto de 256 tokens pertence aos ensaios internos. Ele se aplica depois
de selecionar as primeiras 128 palavras, para comparar a mesma entrada
entre tokenizadores e encoders e limitar o custo. Nenhum dos prefixos
auditados passou dos 256 tokens de seu tokenizer. Na auditoria BPE, o
recorte inicial de 128 palavras afetou 1.535 de 2.048 resenhas (74,95%) e
conservou 51,24% dos tokens na contagem por palavras.
[Efeito do recorte original](diagnosticos.md#efeito-do-recorte-no-teste-oficial) ·
[Cobertura interna](pooling_ordem.md#cobertura-do-recorte-e-embeddings).

### Reproduzir o controle de TF-IDF

Com as quatro divisões preparadas do [protocolo de reprodução](../docs/REPRODUCAO.md),
rode a partir da raiz do projeto. O código confere os hashes das divisões e
usa os registros de predições publicados.

~~~bash
PYTHONPATH=src python seminario/auditar_baseline.py --data data --output tmp/tfidf_ablation
~~~
