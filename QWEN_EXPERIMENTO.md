# Qwen3-8B: protocolo executável

## Rodada atual: geração localizada na opinião

Os arquivos principais de leitura foram atualizados com os resultados de
`results/qwen_improved/`. O primeiro relatório e notebook estão preservados
em `results/qwen_improved/previous_didactic/`. O protocolo original aparece
na seção seguinte e continua válido para os artefatos em `results/qwen/`.

A direção atual é a diferença normalizada entre as médias de 256 estados
internos: 128 enredos de treino, cada um com uma instrução positiva e uma
negativa. Capturamos a última posição do prompt no bloco 28, antes de gerar.
Os rótulos identificam a instrução; não são novas anotações humanas do IMDb.
Não treinamos um scorer para essa direção nem alteramos pesos do Qwen.

Na inferência, o pedido de sentimento é removido. O prompt comum pede FACT
e REVIEW. A intervenção `h += alfa * s * v` começa somente após o marcador
REVIEW: realmente gerado, com alfa -2/0/+2 e escala de treino 100,851032349.
Todos os textos são gerados em uma passagem autoregressiva; não substituímos
o prefixo e não reescrevemos a saída por pós-processamento. A primeira
tentativa usava quebras de linha e falhou quando elas eram omitidas; suas
336 saídas estão preservadas. Os campos explícitos resolvem esse gatilho,
sem garantir uma única frase por campo ou um resumo perfeitamente neutro.

Seleção e intensidade usam somente a validação de geração (oito resenhas e
oito enredos). O teste posterior usa 32 resenhas e 32 enredos inéditos nesta
etapa de geração, em onze condições, totalizando 704 saídas. Todos os prompts
e fontes são iguais entre as condições de cada caso. A comparação com o
controlador original muda direção, bloco, escala e posições em conjunto;
as três direções aleatórias mantêm os demais fatores iguais aos da nova.
As 32 fontes antigas foram reavaliadas depois, em mais 160 saídas.

O juiz semântico é o próprio Qwen congelado sem hooks, avaliando a nova
opinião em REVIEW e fatos no texto completo. Identifica ambos os sentidos-alvo
em 55/64 fontes (24/32 reais e 31/32 sintéticas). TF-IDF, CNN e BERT avaliam
também o texto completo. Proporções usam intervalos de Wilson; diferenças
usam 5.000 reamostragens pareadas das fontes. São intervalos descritivos e
avaliação automática, sem referência humana independente.

Os tokens factuais são idênticos ao zero em todas as intervenções localizadas.
Isso não prova correção do resumo nem impede novas informações em REVIEW.
O relatório inclui uma invenção clara de cenário. Uma releitura do assistente
dos dez alertas factuais positivos sugere nove confusões entre opinião
diferente e erro factual; os rótulos automáticos não foram substituídos.
Os 12 trios qualitativos principais foram escolhidos antes do teste por
ordem e classe; a releitura dos alertas é explicitamente posterior.

O notebook reconstrói a direção original da AGOP e a nova diferença de médias
com os pequenos artefatos incluídos, sem carregar os 8B parâmetros.
`verify_qwen_improved.py` confere hashes, direção, tokens, fonte, prompt,
controles, leituras e entregáveis. `build_qwen_improved_report.py` atualiza os
arquivos de leitura; execute o notebook depois de reconstruí-los.

## Protocolo da primeira rodada

Extensão do projeto de Nestor, Letícia e Gabriel. O objetivo é verificar se um
LLM congelado classifica resenhas e se uma direção aprendida nas ativações
altera sua decisão e o tom de textos que ele gera. Os resultados são
exploratórios; o projeto não precisa demonstrar superioridade da RFM ou do
steering sobre prompting. A pergunta didática é: mantendo pesos e entrada
fixos, o que acontece ao deslocar uma representação interna?

O relatório e o notebook começam pelo mecanismo: prompt e leitura dos logits,
ativações contextuais de 4.096 dimensões, aprendizado de uma direção com um
preditor auxiliar e soma nas ativações do bloco 24. O Qwen não recebe uma nova
cabeça de classificação nem tem seus pesos ajustados. A analogia com GANs e
flows é a manipulação de representações; não pressupõe um único código
latente global nem uma transformação invertível no transformer.

`results/qwen/didactic_examples.json` registra as fontes, saídas integrais,
leituras do assistente, regra de escolha e hashes de origem. Na geração,
usamos `neutral_01` e a primeira resenha oficial de cada rótulo na ordem do
manifesto. Na classificação, usamos o primeiro exemplo de cada rótulo no
subconjunto reservado, com 40 a 150 palavras e sem truncamento. A escolha não
depende do efeito da intervenção. Os três sentidos usam a mesma fonte e o
mesmo prompt. As condições por instrução direta ficam como controles de onde
se intervém: entrada versus estado interno. Não integram o trio de prompt
idêntico. Esta revisão didática reaproveita saídas; não faz nova inferência.

## Configuração fixa e dados

`qwen_config.json` registra a semente, revisão exata do modelo, precisão BF16,
limites de entrada/saída, blocos candidatos, intensidades e controles.
`Qwen/Qwen3-8B` é carregado sem ajuste de seus pesos, com atenção SDPA,
`enable_thinking=False` e decodificação gulosa. A execução registrada usa
Transformers 4.57.6 e acelerador NVIDIA GB10. O modelo tem aproximadamente
8,2 bilhões de parâmetros; CUDA é exigida na implementação de inferência.

As divisões preparadas do IMDb são as do projeto original, com revisão fixa e
hashes em `data/manifest.json`. Treino, validação, calibração e teste não
compartilham textos normalizados idênticos. O teste mantém todas as 25.000
linhas oficiais, inclusive suas 201 repetições internas.

O backbone não usa ajuste supervisionado. A RFM e as direções simples usam
512 exemplos do treino. Largura e iteração do kernel são escolhidas por AUC
em 256 exemplos da validação, separadamente por bloco. O bloco é escolhido
pela maior resposta direcional média na mesma validação, com desempate por
AUC. As intervenções de classificação são medidas também em 256 exemplos
reservados do teste. A validação é pequena e reutilizada nas escolhas; suas
métricas não são estimativas finais sem viés de seleção.

Esta extensão dá continuidade a um projeto em que o teste oficial já havia
sido consultado para comparar os modelos menores e explorar erros. As
escolhas da extensão foram fixadas antes de suas próprias saídas de geração,
mas o mesmo teste reutilizado não é uma nova validação externa independente.
Trate os resultados como investigação exploratória.

## 1. Classificação do teste inteiro

O prompt cita a fonte como dados e pede um único rótulo. A fonte é truncada
antes de adicionar instruções. Os tokens `negative` e `positive` devem ser
unitários no tokenizer. O escore é a diferença de seus logits, normalizada
como probabilidade binária; a massa desses tokens no vocabulário inteiro é
registrada separadamente. Não se pede ao modelo que invente uma confiança.

São avaliados todos os 2.478 exemplos de calibração, 2.478 de validação e
25.000 de teste. Uma temperatura positiva é ajustada somente na calibração.
O limiar de decisão é 0,5. Acurácia, F1 macro, log loss, Brier, ECE, FP e FN
são recalculáveis a partir das previsões. Um audit com 256 exemplos da
validação mede inversão da ordem dos rótulos e redução do prefixo para 192
tokens do Qwen; esse número não equivale a 192 WordPieces do BERT.

No piloto, uma descrição neutra teve escore binário positivo de 0,953,
apesar de os dois tokens de rótulo somarem somente 0,01295% da probabilidade
no vocabulário inteiro. Essa normalização é condicional às duas opções e
não transforma um texto sem opinião em positivo, nem detecta neutralidade.
A massa média desses rótulos nos dados IMDb polarizados ficou próxima de 1.

TF-IDF recebe o texto completo; CNN, até 400 palavras; BERT, 192 WordPieces;
Qwen, até 768 tokens mais o prompt. BERT/CNN foram ajustados no IMDb. A
comparação não isola arquitetura ou tamanho. Não foi possível auditar uma
eventual exposição ao IMDb no pré-treinamento do Qwen.

## 2. RFM / AGOP sem substituir o método

São extraídas ativações do último token do prompt após os blocos 12, 24 e 32
do decoder, antes da RMSNorm final. A regressão usa kernel Laplaciano:

`K_M(x,z) = exp(-sqrt((x-z)^T M (x-z))/L)`.

A RFM alterna regressão kernel regularizada e atualização
`M = média de grad f(h) grad f(h)^T`. O centramento e uma única escala
escalar são aprendidos no treino. A geometria inicial é a identidade completa.
Depois, `M=B^T B`, com `B=G/sqrt(n)`, é uma representação exata, sem PCA ou
projeção aproximada. O gradiente é analítico. Testes com uma implementação
densa incluem validação fora do espaço gerado pelas amostras de treino.

A direção principal da AGOP é orientada para correlação positiva com o rótulo.
AGOP usa gradientes do preditor kernel: não é a Hessiana exata do Qwen, nem
uma atualização de seus pesos. A sonda RFM e a intervenção no LLM são
componentes distintos. Uma boa sonda não garante boa geração.

## 3. Intervenção e controles

Após o bloco escolhido:

`h_t <- h_t + alfa * escala * direção`, para cada posição sem padding.

A intervenção continua em cada novo token durante a geração com KV cache.
Todas as direções têm norma unitária e compartilham a mesma escala, o desvio
padrão da projeção RFM no treino. Portanto, a norma da perturbação por token
é a mesma entre métodos com o mesmo alfa. O número de posições pode mudar
entre condições de geração se os comprimentos gerados mudarem.

Controles: intensidade zero; diferença de médias; sonda logística; três
direções aleatórias; instruções diretas positiva/negativa sem intervenção.
As mesmas intensidades de classificação são executadas em validação e teste:
`-4, -2, -1, 0, 1, 2, 4`. A continuação em cache é comparada ao forward
completo, inclusive com intervenção não nula e padding à esquerda. Hooks
são removidos também em caso de exceção. Intensidade zero preserva a saída;
retirar os hooks restaura o comportamento original.

Mudar os escores com o texto idêntico não muda o sentimento do autor. As
probabilidades sob intervenção não são consideradas calibradas.

## 4. Geração de texto de fato

24 avaliações oficiais de teste são sorteadas, balanceadas por classe, entre
40 e 150 palavras. Oito descrições neutras sintéticas são acrescentadas;
IMDb não tem uma classe neutra neste projeto. Cada fonte recebe 17 condições:

- RFM em alfa `-4, -2, 0, 2, 4`;
- médias, logística e três direções aleatórias em `-4, 4`;
- instrução direta positiva ou negativa, sem hooks.

São 544 saídas. O mesmo pedido de duas frases, sem alvo de sentimento, vale
para todas as intervenções nas ativações. Somente o controle por instrução
pede explicitamente sentimento positivo/negativo. Até 96 novos tokens, sem
amostragem. Nenhuma saída é descartada, regenerada por qualidade ou
reranqueada por um classificador de sentimento. Limite de tokens e saídas
vazias são registrados como falhas possíveis. Posições de geração avançam
com a máscara de atenção; não se reutilizam posições fixas do prefill.

## 5. Avaliação e leitura

TF-IDF, CNN e BERT avaliam as saídas sem hooks. A mudança é pareada à
reescrita sem steering da mesma fonte. ICs descritivos de 95% usam 2.000
reamostragens de fontes por condição e não corrigem múltiplas comparações.
Resultados de avaliações oficiais e fontes sintéticas também são separados.

Similaridade MiniLM com fonte/saída sem steering, repetição, truncamento,
retenção de nomes e sobreposição de números são indicadores auxiliares.
Números podem ser notas de opinião; nomes retidos não provam fatos
preservados. Nenhum desses indicadores substitui leitura humana de
sentimento, fluência e fatos. A calibração dos juízes no IMDb não garante
calibração em textos gerados ou neutros.

Uma planilha cega de 60 casos (`qualitative_review_blind.csv`) está preparada
para revisão posterior. As notas do assistente, em arquivo separado, declaram
sua origem e não são apresentadas como anotação humana independente.
Elas registram os hashes das saídas efetivamente lidas. Uma nova execução que
produza textos diferentes exige nova leitura antes de reutilizar essas notas.

## Execução

Instale o ambiente conforme o README. O pacote de entrega omite os pesos
grandes, dados preparados e caches; as previsões, saídas e relatórios estão
salvos. O notebook de leitura usa apenas os resultados incluídos. Para a
auditoria de dados completa, restaure o dataset público com `prepare`.

```bash
python run.py prepare
# Para reconstruir os juízes e análises originais quando não há checkpoints:
python run.py baseline
python run.py cnn
python run.py bert
python run.py analyze
# Extensão Qwen:
python run_qwen.py pilot
python run_qwen.py features
python run_qwen.py rfm
python run_qwen.py classify
python run_qwen.py steer
python run_qwen.py generate
python run_qwen.py evaluate
python run_qwen.py report
python execute_notebook.py --notebook notebooks/qwen3_8b_steering.ipynb
python -m pytest -q
python verify_qwen.py
```

`python run_qwen.py all` encadeia as etapas da extensão. Classificação e
geração têm partes de retomada com assinaturas de configuração, dados e
direções; partes incompatíveis são recusadas. Não apague partes de outra
execução para esconder resultados. Salve uma nova configuração em outro
diretório de experimento caso pretenda explorar escolhas adicionais.

O relatório lê apenas artefatos existentes. Regerá-lo recria um notebook sem
execução; execute o notebook e a verificação final depois. Metadados de
ambiente e custo são registrados por etapa. `requirements-lock.txt` fixa o
ambiente local; versões de PyTorch/CUDA na execução remota estão nos
metadados. Não é necessário contratar serviço de API ou enviar textos a LLM
externo. Os pesos públicos ocupam aproximadamente 16 GB, além de caches,
memória de trabalho e KV cache.

O ZIP omite também as ativações Qwen e os arquivos RFM de cada bloco;
mantém a direção e a RFM selecionadas. O notebook de leitura não precisa
desses caches. A verificação científica estrita exige os artefatos completos
locais, ou sua reconstrução pelas etapas documentadas. O manifesto de entrega
lista os arquivos incluídos e as omissões.

## Fontes

- [Qwen3-8B oficial](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Beaglehole et al., Toward universal steering and monitoring of AI models](https://arxiv.org/abs/2502.03708)
- [Implementação dos autores](https://github.com/dmbeaglehole/neural_controllers)
- [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

Implementação educacional independente. A adaptação ao Qwen e o protocolo de
controle acima não são uma reprodução integral do artigo.
