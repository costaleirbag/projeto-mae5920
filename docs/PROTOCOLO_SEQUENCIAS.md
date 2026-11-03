# CNN e encoders: protocolo da extensão

Este estudo acrescenta experimentos com texto sequencial ao estudo anterior
com MLPs. Ele preserva os resultados anteriores, inclusive o teste oficial.
As definições executáveis estão em [sequence_config.json](../sequence_config.json).

## Perguntas

1. Como SGD, momentum e Adam aprendem a mesma TextCNN? O resultado muda
   quando cada método recebe uma taxa escolhida na validação?
2. Como a largura dos filtros, as janelas de convolução e o dropout afetam
   a CNN no orçamento fixo?
3. O que se aprende ajustando somente uma cabeça linear sobre um encoder
   congelado? O que muda com o ajuste integral de BERT, DistilBERT e ModernBERT?

## Dados e seleção

Usamos somente `data/train.parquet`, o treino oficial limpo do trabalho original.
Os 6.144 exemplos de treino e os 1.536 de validação coincidem com esses papéis
na extensão MLP. O vocabulário da CNN e o TF-IDF usam apenas o treino.

Uma nova auditoria interna tem 2.048 exemplos balanceados, excluindo todos
os exemplos de treino, validação e auditoria da extensão MLP. Ela recebe o
mesmo tratamento de texto, mas seu arquivo de entrada fica separado durante
o treinamento. Esta auditoria é interna ao projeto e não constitui um novo
benchmark externo. Não voltamos a abrir o teste oficial do IMDb.

Todos os modelos recebem o mesmo prefixo de até 128 palavras separadas por
espaço do texto limpo. A CNN e os encoders limitam a entrada a 256 tokens.
O baseline TF-IDF lê o prefixo completo, sem esse limite adicional.
Como palavras e subpalavras diferem, registramos a truncagem adicional por
tokenizador. A classificação usa o rótulo da resenha completa, portanto um
prefixo pode omitir a conclusão que justifica o rótulo.

Cada execução escolhe a época com menor entropia cruzada de validação entre
as épocas treinadas. A época zero é apenas diagnóstico. A taxa de cada
método é a que minimiza a média dessa perda nas três sementes 17, 29 e 43.
Empates favorecem a menor taxa. Não selecionamos a melhor semente.

Antes de avaliar a auditoria, congelamos o código, as revisões dos modelos,
todos os checkpoints, as taxas e as épocas. As predições e seus hashes
permitem recalcular as métricas. O protocolo admite retomar uma execução
interrompida somente se código e entradas continuam idênticos.

## CNN

Embedding treinável de 128 dimensões. Convoluções paralelas com janelas
3, 4 e 5, cem filtros por janela, ReLU, máximo global com máscara para o
padding, concatenação de 300 valores, dropout 0,5 e camada linear com duas
saídas. Embeddings começam em N(0; 0,1), convoluções usam He e a saída usa Xavier.

São 27 execuções de otimização: três métodos, três taxas e três sementes,
com os mesmos pesos iniciais e a mesma ordem de minibatches por semente.
O painel com taxa comum usa 0,001. O painel ajustado usa a escolha de
validação. O orçamento é de doze épocas, batch 64, com norma do gradiente
limitada a 1 em todos os métodos.

Nove execuções adicionais usam apenas janela 3, apenas janela 5 ou 32 filtros
por janela. Outras seis usam dropout 0 ou 0,8. Esses quinze treinos reutilizam
a taxa escolhida para Adam, sem ajuste por arquitetura ou dropout. A condição
multiescala com cem filtros e dropout 0,5 reutiliza o treino do estudo de
otimizadores. As comparações com dropout compartilham os pesos iniciais.

Há 42 treinos distintos de CNN. Não tentamos demonstrar double descent
nessa CNN. O exemplo sintético da extensão anterior continua identificado
como outro experimento.

## Encoders

Partimos dos checkpoints públicos de pré-treinamento, sem ajuste prévio no
IMDb. As revisões exatas ficam em `results/sequences/models_pinned.json`.
O encoder produz um vetor por token. Uma média com máscara exclui padding
e tokens especiais. Sobre o vetor médio de 768 dimensões acrescentamos
dropout 0,1 e uma camada linear 768 por 2, com 1.538 parâmetros.

No modo congelado, o encoder fica em avaliação e produz os vetores uma única
vez. Só a cabeça é treinada, por quarenta épocas, Adam e três taxas. São
27 treinos de cabeça. No ajuste completo, todas as camadas do encoder e a
cabeça recebem gradientes. São dezoito treinos: três modelos, duas taxas e
três sementes. Cada treino tem três épocas, batch 32, AdamW, decaimento 0,01
nos pesos matriciais, warmup de 10%, redução linear da taxa e clipping 1.
Os vieses e parâmetros de normalização não recebem decaimento.

As cabeças congelada e completa começam com os mesmos pesos para cada
semente. O modo congelado mantém o dropout do encoder desativado e reutiliza
vetores. O ajuste completo ativa o dropout do encoder durante o treino.
As políticas têm orçamentos e otimizadores diferentes, escolhidos de forma
explícita para cada regime. Esta comparação avalia duas receitas práticas de
transferência, sem isolar o efeito de destravar pesos como único fator.

BERT tem doze blocos, DistilBERT seis e ModernBERT-base 22. Os modelos diferem
em pré-treinamento, tokenização, arquitetura e número de parâmetros. Um
resultado melhor não identifica qual dessas diferenças causou o ganho.
Usamos SDPA e BF16 para os encoders, com parâmetros e estado do otimizador
em FP32. A CNN usa FP32. TF32 fica desativado em treino e auditoria.

## Custo e interpretação

O cronômetro de treino inclui a passagem de treino com os minibatches e as
atualizações, mas exclui avaliações entre épocas, download e carregamento.
O tempo total da execução inclui avaliações e seleção de checkpoint. Para
encoders congelados, a extração dos vetores é um custo separado, a ser
considerado além do treino da cabeça. Resultados de tempo refletem esta
implementação e este hardware. Não representam o máximo de desempenho
possível de cada arquitetura.

As figuras mostram todas as três sementes, usando média e mínimo/máximo.
Essas faixas descrevem as execuções, sem representar intervalos de confiança
populacionais. A auditoria serve para descrever as escolhas já congeladas.

## Referências primárias

- [Kim, 2014: convoluções para classificação de sentenças](https://aclanthology.org/D14-1181/).
- [Devlin et al., 2019: BERT](https://aclanthology.org/N19-1423/).
- [Sanh et al., 2019: DistilBERT](https://arxiv.org/abs/1910.01108).
- [Warner et al., 2024: ModernBERT](https://arxiv.org/abs/2412.13663).
- [Modelo e instruções do ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base).
- [Implementação oficial no Transformers](https://huggingface.co/docs/transformers/model_doc/modernbert).
- Simon J. D. Prince, *Understanding Deep Learning*: capítulos 6, 7, 8, 9,
  10, 11 e 12, conforme o catálogo de fontes da disciplina já documentado.
