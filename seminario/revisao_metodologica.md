# Revisão metodológica para o seminário

## Por que ambos os vocabulários têm 20 mil entradas?

A mesma regex separa palavras, contrações e números nas duas versões.
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

No treino real de 19.823 resenhas, *movie* aparece em 12.153 e *awful*
em 1.151. Os IDFs são 1,49 e 3,85. Se os termos ocorrem uma vez cada,
o segundo recebe 2,58 vezes o valor do primeiro. Isso reserva mais
espaço para termos que distinguem resenhas. Uma grafia rara também
pode ter IDF alto. O coeficiente que a logística aprende com os
rótulos determina o sinal da contribuição. Nosso vocabulário só
admite termos com pelo menos duas resenhas de treino.
[Frequências medidas](../results/tfidf_regularization/summary.json).

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

### Mais regularização ajudou?

Ampliamos a busca de \(C\) de 0,125 a 64. Um \(C\) menor penaliza mais
os coeficientes. Com \(C=0{,}125\), a acurácia no treino baixou para
89,35% e o F1 macro de validação para 0,8796, ante 99,72% e 0,9152
com \(C=8\). O melhor F1 na validação escolheu \(C=16\). No teste oficial,
essa escolha fez 90,04%, ante 90,15% na repetição de \(C=8\) no mesmo
ambiente. O resultado original de 90,14% continua sendo o principal.

Ao tirar só o IDF e manter \(C=8\), fizemos 89,45% no teste. Mesmo com
sua própria escolha na validação (\(C=64\)), a versão sem IDF fez 89,31%.
Os modelos mantêm palavras, bigramas, TF logarítmico e normalização.
A comparação apoia o uso do IDF nesta receita. O objetivo é melhorar
em dados novos. Uma acurácia menor no treino, por si só, pouco diz
sobre o desempenho fora dele.
[Busca ampliada](../results/tfidf_regularization/selection.json) ·
[Teste pareado](../results/tfidf_regularization/summary.json).

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

Em um controle com a mesma cabeça de 300 entradas, trocar o máximo pelo
máximo suave elevou a acurácia de 87,11% para 87,71% com teto de 400 tokens,
na média de três sementes. Com 768, foi de 87,63% para 88,13%. A posição
ponderada também foi testada e o acerto nos pares com veredito no fim
continuou perto de zero. Podemos atribuir o ganho observado à mudança de
agregação nessa receita. O controle deixa em aberto quanto do restante
vem dos filtros e do sinal de ordem nos dados de treino.
[Ensaio de pooling com capacidade fixa](ensaios_cnn.md#o-que-muda-ao-trocar-só-o-pooling).

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

## Por que o BERT tem dois resultados?

| Slide | Treino e entrada | Avaliação | Minibatches de treino |
|---|---|---|---|
| 9 | Estudo original, 19.823 textos. TF-IDF lê o texto inteiro. CNN e BERT usam até 400 tokens por palavras e 192 WordPieces, respectivamente. | 25.000 resenhas do teste oficial. Uma execução por modelo. O BERT fez 90,94%. | CNN 64, BERT 32. A regressão logística usa LIBLINEAR. |
| 16 | Extensão, 6.144 textos e o mesmo prefixo de 128 palavras para todos os encoders. | 2.048 resenhas da auditoria interna, média de três sementes. BERT com cabeça nova fez 84,52%. BERT ajustado por inteiro fez 88,31%. | 128 ao ajustar só a cabeça, 32 no ajuste completo. |

O quadro do Qwen no slide 16 cita separadamente seu resultado no teste oficial.
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

Acima de 500 palavras, a CNN original fez 81,52% de acurácia e 0,8146
de F1 macro. Com os mesmos pesos no texto integral, fez 86,18% e
0,8615. Nas duas faixas até 250 palavras, nenhuma entrada foi cortada e
o TF-IDF ainda ficou à frente. As faixas contam palavras por espaços.
O teto da CNN conta tokens por regex. [Resultados por faixa](dados/comprimentos.json).

O teto de 256 tokens pertence aos ensaios internos. Ele se aplica depois
de selecionar as primeiras 128 palavras, para comparar a mesma entrada
entre tokenizadores e encoders e limitar o custo. Nenhum dos prefixos
auditados passou dos 256 tokens de seu tokenizer. Na auditoria BPE, o
recorte inicial de 128 palavras afetou 1.535 de 2.048 resenhas (74,95%) e
conservou 51,24% dos tokens na contagem por palavras.
[Efeito do recorte original](diagnosticos.md#efeito-do-recorte-no-teste-oficial) ·
[Cobertura interna](pooling_ordem.md#cobertura-do-recorte-e-embeddings).

Os [novos ensaios da TextCNN](ensaios_cnn.md) comparam treinar de ponta a
ponta com tetos de 256, 400 e 768 tokens e com o texto inteiro. Os
resultados também separam escolhas de pooling, largura e regularização.

### Reproduzir o controle de TF-IDF

Com as quatro divisões preparadas do [protocolo de reprodução](../docs/REPRODUCAO.md),
rode a partir da raiz do projeto. O código confere os hashes das divisões e
usa os registros de predições publicados.

~~~bash
PYTHONPATH=src python seminario/auditar_baseline.py --data data --output tmp/tfidf_ablation
~~~
