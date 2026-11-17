# Roteiro de fala

O texto para falar e os cliques dos 15 slides estão na colinha de banca.
[Abrir o roteiro slide a slide](colinha.md#roteiro-slide-a-slide).

[Como ler as curvas](#como-ler-as-curvas) · [Configuração original](#configuracao-original) · [Como os tempos foram medidos](#como-os-tempos-foram-medidos)

## Como ler as curvas

- **Pico inicial do SGD.** A época 0 mostra os pesos da inicialização.
  A CE média sobe de 0,74 para 0,84 na validação.
  No treino, também sobe de 0,74 para 0,83, com acurácia em 50%. A taxa
  0,3 é a maior da grade do SGD. No controle com 0,03, a CE de validação
  cai para 0,70 na primeira época e termina em 0,68, ante 0,46 com 0,3.
  Isso é compatível com passos iniciais grandes em gradientes de
  minibatches. As curvas são médias por época e descrevem o resultado
  desses passos.
- **Momentum e Adam.** O momentum 0,9 acumula a direção de passos recentes.
  Isso pode reduzir o zigue-zague, e a direção acumulada também pode levar
  a ultrapassar um mínimo. A faixa sombreada é a variação entre sementes.
  Adam também usa uma média dos gradientes e ajusta a escala de cada
  parâmetro pela média dos gradientes ao quadrado. A trajetória mais
  suave vale para a taxa usada. Com Adam a 0,003, a melhor época foi a
  2 ou a 3, e a CE média de validação terminou em 0,85.
- **Escolha da época.** No Adam a 0,001, a CE média de treino cai de 0,054
  na época 6 para 0,003 na 12. A de validação sobe de 0,377 para 0,510.
  A CE penaliza especialmente erros cometidos com alta confiança. Como
  selecionamos pela menor CE de validação, guardamos a época 5 ou 6 de
  cada semente, mesmo tendo executado as 12. Isso ilustra a seleção de
  *checkpoint* pelo princípio do *early stopping*.
  [Curvas](../results/sequences/presentation_data.json) ·
  [Prince, cap. 6](https://github.com/udlbook/udlbook/tree/main/Notebooks/Chap06) ·
  [Livro, seções 6.2–6.4 e 9.3.1](https://udlbook.github.io/udlbook/).

<a id="configuracao-original"></a>

## Configuração do estudo original

| Modelo | Entrada | Ajuste |
|---|---|---|
| TF-IDF + logística | Resenha limpa inteira. Até 80 mil atributos, palavras e pares consecutivos. | L2, \(C=8\) escolhido na validação entre 0,5, 2 e 8. |
| CNN por regex | Até 400 tokens. Vocabulário de 20 mil IDs, embedding de 128, 100 filtros por largura 3, 4 e 5, máximo global. | AdamW, taxa 0,001, decaimento 0,01, dropout 0,5, lote 64. Cinco épocas, escolhida a quinta. |
| BERT-base ajustado | Até 192 WordPieces, incluindo tokens especiais. | AdamW, taxa 0,00002 com aquecimento e decaimento linear, decaimento de pesos 0,01, lote 32. Duas épocas, escolhida a segunda. |

O controle de 400 recorta o texto com a regex da CNN e depois aplica o
analisador de unigramas e bigramas do TF-IDF ao prefixo. Reajustamos o
vetorizador e a regressão no treino e escolhemos \(C\) na validação.
As duas receitas desse controle escolheram \(C=8\). No teste, as
acurácias foram 90,148% com texto inteiro e 89,664% com recorte. A
repetição integral acertou uma resenha a mais que a versão original.

Nos ensaios internos dos slides 6 e 11, usamos o prefixo comum de até
128 palavras e um teto de 256 tokens por tokenizador. Os prefixos da
auditoria BPE já cabiam nesse teto adicional. Esse protocolo usa outras
divisões, separadas do teste oficial.
[Configuração original](../config.json) ·
[Controle de recorte](../results/tfidf_ablation/summary.json).

## Como os tempos foram medidos

- **Estudo original.** O cronômetro do TF-IDF inclui vetorização,
  transformação das quatro divisões, busca de três valores de \(C\) e
  seleção com a validação. Vetorizar consumiu 11,6 s dos 13,1 s. A
  regressão ajusta 80 mil coeficientes e um viés. A CNN ajusta 2,71 milhões
  de parâmetros e o BERT 109,48 milhões. Os tempos das redes incluem
  validação e gravação dos pesos por época quando melhora. Eles começam
  com os textos já tokenizados. São tempos dos dispositivos e receitas
  usados no trabalho. [TF-IDF](../results/tfidf/metadata.json) ·
  [CNN](../results/cnn/metadata.json) · [BERT](../results/bert/metadata.json).
- **Inferência original, para perguntas.** Foram 0,15 s para a
  regressão, 20,72 s para a CNN e 149,91 s para o BERT, somando as
  quatro divisões com as entradas já vetorizadas ou tokenizadas.
  Para o TF-IDF, cada novo texto ainda precisa virar um vetor.
  [Cronômetros do TF-IDF](../src/imdb_lab/baseline.py) ·
  [Cronômetros das redes](../src/imdb_lab/training.py).
- **Comparação interna do BERT.** Usamos os mesmos 6.144 trechos no
  treino, 1.536 na validação e 2.048 na auditoria. Cada trecho tem
  até 128 palavras e 256 tokens. Na mesma NVIDIA GB10, o regime congelado
  extraiu os vetores de treino e validação em 15,2 s e treinou a camada
  linear em 2,35 s. O tempo do ajuste completo cobre os lotes de treino.
  Ambos os tempos excluem as avaliações ao longo das épocas. A camada
  linear teve 40 épocas com lote 128. O ajuste completo teve 3 épocas
  com lote 32. A diferença de tempo e acurácia vale para essas duas
  receitas. Para classificar os 2.048 trechos
  já tokenizados, ambos ainda passaram pelo encoder. Foram cerca de 4,3
  e 4,0 s, respectivamente. [Receitas e tempos](../results/sequences/summary.json) ·
  [Laço de treino e extração](../src/imdb_lab/sequences.py).
- **O que mede o pré-treino.** A cabeça isolada mostra até onde podemos
  aproveitar os pesos publicados com um classificador linear. Para
  medir o ganho causado pelo pré-treino, seria preciso comparar o mesmo
  BERT inicializado aleatoriamente e pré-treinado. O ajuste de apenas
  alguns blocos finais seria uma medição intermediária.
  [Políticas implementadas](../src/imdb_lab/training.py).
- **Qwen3-8B.** Classificou com prompt e pesos congelados na NVIDIA GB10.
  Foram 95,26% no teste oficial. O tempo registrado foi 63,3 min para
  o processo de classificação que cobre as 29.956 resenhas da
  validação, calibração e teste, com entrada de até 768 tokens da
  resenha. O cronômetro inclui carregamento, preparação dos prompts,
  inferência e escrita de resultados. A temperatura foi ajustada em
  uma amostra separada. [Metadados](../results/qwen/classification/metadata.json) ·
  [Código de classificação](../src/imdb_lab/qwen_experiment.py).
