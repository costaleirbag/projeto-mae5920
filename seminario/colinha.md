# Colinha de banca · MAE5920

**O que a CNN lê em uma resenha?** Material de ensaio para Nestor, Letícia e
Gabriel. A apresentação investiga o que uma TextCNN aprende para classificar
sentimento no IMDb e o que se perde entre o texto e a decisão. Usem as perguntas
como cartões. Uma pessoa lê, outra responde em voz alta e a terceira abre o
gabarito e pede o controle que sustenta a resposta.

[Revisão rápida](#revisão-rápida) · [Mapa dos modelos](#mapa-dos-modelos) · [Perguntas de banca](#perguntas-de-banca) · [Código para abrir na hora](#código-para-abrir-na-hora) · [Ensaio em grupo](#ensaio-em-grupo)

**Primeiro passe, em cinco minutos.** Cartões 02 (vazamento), 17 (rótulos
embaralhados), 19 (por que a CNN perdeu), 22 (sementes), 27 (máximo global)
e 45 (uso de IA). Abra o gabarito depois de responder.

## Revisão rápida

### A resposta de 30 segundos

> Partimos de uma tarefa binária, com resenhas positivas e negativas. No
> IMDb, o TF-IDF com regressão logística acertou 90,14% e a nossa TextCNN
> acertou 86,78%. Mostrar a resenha inteira à mesma CNN recuperou meio
> ponto no total e ajudou mais nas resenhas longas. Em outro controle,
> trocamos dois blocos preservando todas as janelas locais. Os máximos e
> os logits ficaram iguais, o que mostra exatamente onde a posição se
> perdeu. BERT e Qwen passaram do baseline com outras receitas e
> pré-treinamento. O trabalho explica o comportamento da nossa CNN por
> meio dessas comparações.

### Os números que a banca pode pedir

| Referência | Número para lembrar | Onde aconteceu |
|---|---|---|
| Teste principal | TF-IDF **90,14%**, CNN **86,78%**, BERT **90,94%** | Mesmas **25.000** resenhas do teste oficial |
| Qwen3-8B | **95,26%**, pesos congelados, classificação por logits de rótulos | Extensão no teste oficial já consultado |
| Recorte da CNN | **400 tokens por regex**, **3.118** resenhas cortadas. Com o mesmo checkpoint e texto inteiro, **87,28%** | Teste oficial |
| CNN curta | Até **128 palavras**, depois teto de **256 tokens**. **2 tokenizadores × 3 sementes** na comparação BPE | Auditoria interna de **2.048** resenhas |
| Vocabulário e rede | **20.000 × 128** no embedding. Janelas **3/4/5**, **100 filtros** por largura, **300 máximos**, **2 logits** | TextCNN base |
| Custo original | **13,1 s** TF-IDF, **139,6 s** CNN, **446,4 s** BERT | CPU do M1, GPU do M1 e NVIDIA GB10, respectivamente |
| BERT congelado / completo | **84,52% / 88,31%**. **17,6 s / 148,9 s** para extração + ajuste / ajuste | Mesma auditoria sequencial, três sementes |
| Posição | Mesmas janelas em outra ordem dão **logits idênticos**. Nos **344** pares editados, dois picos por valor deram **39,83%** e por posição **39,15%** | Troca de blocos sintética e teste da cabeça com os filtros congelados |

[Métricas oficiais e auditorias](../docs/RESULTADOS.md) · [Roteiro e cronômetros](roteiro.md#como-os-tempos-foram-medidos) · [Controles de recorte e posição](diagnosticos.md)

> **Antes de citar uma porcentagem**, diga em que conjunto ela foi medida,
> qual foi a entrada e quais pesos foram atualizados. A maior parte das
> perguntas difíceis começa quando uma dessas três informações fica implícita.

## Mapa dos modelos

### O que vai à apresentação

| Modelo ou ensaio | O que ajustamos | Por que entra no estudo |
|---|---|---|
| **TF-IDF + regressão logística** | Vocabulário, IDF e coeficientes de uma regressão L2. Até 80 mil unigramas e bigramas da resenha inteira. | Referência forte e rápida para medir o quanto pistas lexicais já resolvem. |
| **TextCNN original** | Embeddings, convoluções e saída do zero. Teto de 400 tokens. AdamW a 0,001, lote 64, cinco épocas, dropout 0,5 e decaimento 0,01. | Vê ordem local, aprende as próprias features e permite observar a varredura e o pooling. |
| **BERT-base original** | Reutilizamos os pesos públicos de pré-treino e ajustamos tudo, incluindo a cabeça binária. Teto de 192 WordPieces com especiais. AdamW a 0,00002, lote 32, duas épocas. | Referência de transferência com um encoder contextual. |
| **CNNs internas por palavras** | 42 treinos. São 27 de otimização (3 métodos × 3 taxas × 3 sementes), nove de arquitetura e seis de dropout. | Entender como taxa, checkpoint, largura e regularização mudam uma TextCNN. |
| **CNNs internas com BPE** | Três novos treinos usam um BPE ajustado no texto de treino. Reutilizam-se três checkpoints por palavras, pareados nas sementes 17, 29 e 43. | Separar, nesta receita, a troca de tokenização da variação entre treinos. Os mesmos seis fazem os testes com frases e comentários. |
| **BERT, DistilBERT e ModernBERT internos** | 27 cabeças lineares sobre vetores congelados e 18 ajustes completos. Também ajustamos TF-IDF + logística no prefixo comum como referência. | Medir custo e acerto ao reaproveitar uma representação pronta ou adaptar todo o encoder. |
| **Qwen3-8B** | Reutilizamos o modelo público com pesos congelados. A classificação lê logits e ajusta apenas uma temperatura fora do treino. Nas intervenções, estimamos direções separadamente. | Referência de LLM para sentimento e extensão didática sobre ativações internas. |

**Contagem da extensão sequencial.** Os 42 treinos de CNN por palavras, 27
cabeças e 18 ajustes completos totalizam **87**. Com as três CNNs BPE novas,
chegam a **90**. As linhas “CNN + Adam”, “arquitetura base” e “dropout 0,5”
podem mostrar os mesmos três pesos. As três sementes geram três métricas que
depois são promediadas. Um ensemble teria de combinar as saídas por resenha
antes de calcular a métrica. [Receitas e contagens](../docs/PROTOCOLO_SEQUENCIAS.md) ·
[Protocolo BPE](../docs/PROTOCOLO_BPE.md) · [Configuração original](../config.json)

### O que fica para perguntas

| Estudo ou componente | O que fizemos | Para que serve |
|---|---|---|
| **Cabeças logísticas de posição** | Com os filtros congelados, reajustamos a cabeça global de 300 entradas e duas cabeças de 600 entradas para os mesmos picos ordenados pelo valor ou pela metade do texto. | Controlar a quantidade de parâmetros da cabeça ao testar posição. |
| **Seis poolings de ponta a ponta** | 18 treinos, três sementes por pooling. Máximo global, máximos por metade ordenados por valor ou posição, média global, médias por metade e K-max ordenado. | Ver se o aprendizado dos filtros passa a aproveitar a posição. |
| **Outros 93 treinos de CNN** | 33 de comprimento/arquitetura/regularização, 15 de comprimento e ordem em pares, nove de embeddings aleatórios ou GloVe, 12 de treino contrastivo e 24 de pooling com cabeça do mesmo tamanho. | Investigar a diferença para TF-IDF e o caso em que a opinião final muda. Resultados exploratórios. |
| **CNN maior e mais épocas** | Texto inteiro, duas sementes, TextCNN de 2,71 mi e CNN residual piramidal de 11,13 mi com duas taxas iniciais. | Comparar a melhor época até a quinta com a melhor de toda a execução. |
| **MLPs do suplemento** | 51 treinos sobre TF-IDF reduzido por SVD a 128 dimensões e padronizado. 27 de otimização, 15 de profundidade/resíduos e nove de regularização. | Isolar mecanismos do curso em redes densas. São experimentos próprios, com outra auditoria interna. |
| **Inicialização e double descent sintéticos** | Uma rede ReLU de 32 camadas para inspecionar escalas antes do treino. Na regressão sintética, camada aleatória congelada e ajuste só da saída, por mínimos quadrados ou ridge, em 880 condições. | Observar escala de ativações e gradientes, e explorar erro perto da interpolação. |
| **MiniLM + K-means** | MiniLM pré-treinado e congelado para extrair vetores. K-means ajustado em 2.500 resenhas de treino, gerando oito grupos exploratórios. | Organizar a análise de erros por grupos de texto. MiniLM também fornece uma medida auxiliar de similaridade na geração. |
| **Sondas de steering** | Ajustamos RFM, uma regressão kernel sobre ativações congeladas, além de uma sonda logística de controle. A direção mais recente do Qwen usa diferença de médias de ativações pareadas. | Distinguir atributos decodificáveis da intervenção que soma um vetor na rede. |
| **BERT de edição mascarada** | Reutilizamos o encoder ajustado e uma cabeça de linguagem mascarada para preencher duas palavras selecionadas sob intervenção. | Primeira demonstração de alterar texto, antes dos experimentos de geração autoregressiva do Qwen. |

[Ensaios da CNN](ensaios_cnn.md) · [Ensaio de pooling](pooling_ordem.md) ·
[Suplemento de mecanismos](../docs/CONEXOES_DISCIPLINA.md) · [Análises e steering originais](../docs/EXPERIMENTO_ORIGINAL.md)

## Perguntas de banca

### Dados e métricas

<details>
<summary>01 · Por que separar treino, validação, calibração e teste?</summary>

**Resposta.** No estudo original, usamos 19.823 resenhas para aprender
vocabulário, IDF e pesos, 2.478 para escolher hiperparâmetros e checkpoints,
2.478 para ajustar a temperatura e 25.000 para medir desempenho. Nas extensões
principais, foram 6.144 de ajuste, 1.536 de validação e 2.048 de auditoria
interna, com as escolhas congeladas antes da abertura da auditoria.

**Se o professor perguntar por que há várias auditorias de 2.048.** É
o tamanho de cada conjunto reservado, e as amostras diferem entre
MLP, CNN/encoders e o comparativo BPE. Cite o nome da auditoria com o número.
[Manifesto das divisões](../data/manifest.json) · [Protocolos sequencial](../docs/PROTOCOLO_SEQUENCIAS.md) e [BPE](../docs/PROTOCOLO_BPE.md)

</details>

<details>
<summary>02 · Vocês olharam o teste para excluir duplicatas. Isso é vazamento?</summary>

**Resposta.** Lemos o texto do teste para remover do treino 123 linhas com
hash correspondente. Eliminamos outras 98 duplicatas internas no treino.
O ajuste do vocabulário, IDF, fusões e pesos usou a divisão de treino.
Os rótulos do teste ficaram reservados para avaliar e os quatro conjuntos
finais têm zero hashes de texto normalizado em comum. O teste manteve as
25.000 linhas oficiais, incluindo 201 ocorrências duplicadas.

**Se ele aprofundar.** Foi uma limpeza que conhecia o texto do teste,
aplicada para reduzir contaminação exata. SHA-256 do texto limpo cobre a
igualdade depois da normalização. Paráfrases, repetições dentro do teste,
possível exposição no pré-treino e decisões motivadas por inspeções
posteriores são questões distintas. As extensões que voltaram ao teste
oficial são exploratórias. Para confirmar uma receita nova, congele-a
antes de abrir outra amostra.
[Preparação](../src/imdb_lab/data.py) · [Manifesto](../data/manifest.json)

</details>

<details>
<summary>03 · Qual era a entrada do modelo? Havia classe neutra ou nota numérica?</summary>

**Resposta.** Usamos o texto limpo da resenha, com HTML decodificado, tags
removidas e espaços normalizados. O rótulo oficial é 0 negativo ou 1
positivo. ID, comprimento e hash são campos de auditoria. Um escore perto
de 50% indica margem pequena entre as duas classes do modelo. Comentários
ambíguos do YouTube ficaram sem rótulo de referência. Uma nota escrita no
próprio texto, como “9/10”, pode permanecer como pista lexical.
[Dicionário de dados](../DATA_DICTIONARY.md) · [IMDb original](https://ai.stanford.edu/~amaas/data/sentiment/)

</details>

<details>
<summary>04 · Por que acurácia, F1 macro e entropia cruzada?</summary>

**Resposta.** Acurácia conta a fração de acertos. F1 combina precisão e
revocação. Calculamos um F1 tratando positiva como alvo, outro tratando
negativa como alvo, e tiramos a média. A entropia cruzada mede
$-\log p(\text{classe correta})$ por exemplo e depois faz a média.
Dar 1% à classe correta custa cerca de 4,61. Dar 49% custa 0,71. Os dois
casos podem ser erros de classe, com confiança bem diferente.

**No quadro.** Em 100 exemplos com 90 positivos, prever tudo como positivo
dá 90% de acurácia. O F1 positivo é 0,947 e o negativo é 0, então o F1 macro
fica em 0,474. No IMDb oficial balanceado, as métricas ficaram próximas.
[Cálculo das métricas](../src/imdb_lab/metrics.py)

</details>

<details>
<summary>05 · Se calibrar melhora a loss, por que a acurácia fica igual?</summary>

**Resposta.** Dividimos a margem $z_{+}-z_{-}$ por uma temperatura positiva
escolhida pela menor CE na calibração. A margem muda de escala e conserva
o sinal. Com o limiar de 50%, a classe fica igual, assim como a acurácia
e o F1 macro. No TF-IDF, a CE do teste vai de **0,2531 para 0,2396**.
Um escore calibrado no IMDb ainda precisa ser avaliado para uso em outro
domínio.
[Código da temperatura](../src/imdb_lab/metrics.py) · [Métricas brutas e calibradas](../results/comparison.csv)

</details>

<details>
<summary>06 · Três sementes dão um intervalo de confiança? O ganho é significativo?</summary>

**Resposta.** As faixas dos gráficos vão do mínimo ao máximo em três
treinos. Elas descrevem a variação observada e dependem da mesma amostra
de avaliação. Para quantificar a incerteza ao comparar dois modelos nas
mesmas resenhas, eu usaria uma comparação pareada por exemplo, como
McNemar para acerto/erro, ou reamostraria os pares de previsões para medir
a diferença de F1. Para o teste de ordem, a unidade é o par inteiro.
“Estatisticamente detectável” e “grande o bastante para justificar o custo”
são perguntas distintas.
[Protocolo das faixas](../docs/PROTOCOLO_SEQUENCIAS.md) · [Predições e métricas](../docs/RESULTADOS.md)

</details>

### Tokenização e a TextCNN no quadro

<details>
<summary>07 · BPE substituiu a regex? Quem aprendeu o vocabulário?</summary>

**Resposta.** A regex primeiro separa palavras em inglês, contrações e
números, depois de converter para minúsculas. Na CNN por palavras,
buscamos a palavra inteira no vocabulário. No BPE, partimos das palavras
já separadas e aprendemos fusões frequentes de caracteres e peças a partir
dos 6.144 textos de treino. O tokenizer fica fixo antes do treino
supervisionado. A CNN aprende os vetores que correspondem aos IDs, os filtros
e a camada final pela loss de sentimento.

**No quadro.** A palavra rara *pallbearer* vira um `<unk>` na versão por
palavras. O BPE da execução devolve `p`, `all` e `bearer</w>`. O marcador
`</w>` faz parte do último token. Nosso BPE opera sobre caracteres e pode
chegar a `<unk>` se encontrar caracteres fora do alfabeto aprendido.
[Tokenizer](../src/imdb_lab/bpe_study.py) · [Fusões salvas](dados/animacoes.json)

</details>

<details>
<summary>08 · Os dois vocabulários terem 20 mil entradas foi coincidência?</summary>

**Resposta.** Foi uma escolha de controle. Reservamos 19.998 itens lexicais,
`<pad>` e `<unk>` na versão por palavras. O BPE tem 20 mil como alvo e
atingiu esse valor com 19.924 fusões. Assim ambas as CNNs têm a mesma
quantidade de parâmetros no embedding. Uma janela de cinco tokens BPE
pode cobrir menos palavras do que a janela da CNN por palavras. Parte da
comparação é justamente essa mudança de unidade.
[Configuração](../bpe_config.json) · [Treino do BPE](../results/bpe/tokenizer_training.json)

</details>

<details>
<summary>09 · A lookup table tem tamanho B × T × 128? E os últimos dez IDs?</summary>

**Resposta.** A tabela de pesos mede **20.000 × 128**, uma linha por ID.
Para um lote de IDs de tamanho **B × T**, o lookup gera **B × T × 128**.
Transpomos para **B × 128 × T** porque a Conv1d espera canais antes do eixo
que ela percorre. Na CNN original, `T` segue o maior comprimento do lote,
com piso 5. No treino interno, as sequências de cada divisão são montadas
em uma matriz já preenchida. Em ambos, o preenchimento é à direita.
Se o exemplo tem comprimento `T − 10` na matriz, os últimos dez IDs valem
zero, o `<pad>`. O código também passa o comprimento real para a máscara.

**Se perguntar por que vetor de padding zero é insuficiente.** Uma janela
mista ainda pode conter tokens reais, e a convolução tem viés. A máscara
retira as janelas incompletas do máximo. Para textos com menos tokens que
a largura, o código mantém uma janela preenchida à direita.
[Arquitetura e padding](../src/imdb_lab/models.py) · [Empacotamento no estudo interno](../src/imdb_lab/sequences.py)

</details>

<details>
<summary>10 · O que dá 2.714.502 parâmetros? Mostre a conta.</summary>

**Resposta.** São **2.560.000** valores na tabela de embeddings
(`20.000 × 128`), **153.900** nas convoluções
(`100 × [(3 × 128 + 1) + (4 × 128 + 1) + (5 × 128 + 1)]`) e **602**
na saída (`300 × 2 + 2`). Os `+1` das convoluções contam um viés por
filtro. A linha de padding integra a contagem da tabela, embora fique
zerada. Os três ramos recebem os mesmos embeddings em paralelo. Para um
texto de dez tokens, com passo 1, cada um produz 100 mapas com **8, 7 e 6**
posições, respectivamente. O máximo deixa 100 valores por ramo.
[Implementação](../src/imdb_lab/models.py) · [Montagem no slide 5](https://costaleirbag.github.io/projeto-mae5920/#slide-5)

</details>

<details>
<summary>11 · Por que “terrible” precisa cair numa posição específica? Onde está “wonderful”?</summary>

**Resposta.** Cada posição dentro de um filtro tem seu próprio vetor de
pesos. O filtro soma contribuições de todos os embeddings naquela janela.
No filtro 89 de largura 3, na frase A, `movie was terrible` ativa **0,98** e
`was terrible but` ativa **0,40** após a ReLU. A terceira posição da janela
recebe pesos diferentes da segunda, e os vizinhos também mudam. Deslizar
a mesma janela inteira para outro ponto do texto preserva a resposta do
mesmo filtro. Selecionar largura 4 ou 5 no slide troca o ramo e o filtro.

**Se perguntar pelo positivo.** O filtro 6, também de largura 3, tem seu pico
de **0,45** em `it was wonderful` e contribui para a classe positiva. A
camada final soma os 300 máximos com seus coeficientes. A maior ativação
de um único filtro é apenas uma contribuição. ReLU é `max(0, z)`. Uma
ativação positiva ainda pode receber peso negativo na saída.
[Varredura do slide 5](https://costaleirbag.github.io/projeto-mae5920/#slide-5-varredura) · [Ativações medidas](dados/animacoes.json)

</details>

<details>
<summary>12 · A CNN precisava de uma MLP ou uma sigmoid no treino?</summary>

**Resposta.** A implementação combina 300 máximos com uma camada
`Linear(300, 2)`. Convolução com ReLU e pooling já tornam o modelo
não linear em relação à entrada. A entropia cruzada recebe os dois
logits diretamente e trata a normalização de forma numericamente estável.
Para exibir a classe, escolhemos o maior logit. Para exibir a probabilidade
positiva, o softmax de dois logits equivale à sigmoide da margem
`z_positivo − z_negativo`. Outra implementação possível usa um único
logit com `BCEWithLogitsLoss`.
[Saída da CNN](../src/imdb_lab/models.py) · [Laço de treino](../src/imdb_lab/training.py)

</details>

<details>
<summary>13 · Por que cortar em 400? E de onde vieram 128 e 256?</summary>

**Resposta.** O 400 já estava na configuração original. Um teto limita a
memória, o padding e a quantidade de janelas, mas a arquitetura aceita
sequências variáveis. No teste, cortou **3.118/25.000** resenhas (**12,47%**),
removendo **10,41%** dos tokens. A maior tinha **2.247 tokens por regex**.
Nas extensões, usamos primeiro um prefixo de **128 palavras por espaços**
e depois um teto de **256 tokens** para as redes. Na auditoria BPE, 1.535
de 2.048 perderam texto pelo prefixo, e todos os prefixos já cabiam em 256.

**No quadro.** A resenha do teste tem média de **226 palavras**, mediana
**170** e P90 **439**, contando por espaços. As frases têm média de **18**
palavras e mediana **16**, com segmentação aproximada por pontuação.
Especifique a unidade antes de comparar esses números a um teto de tokens.
[Distribuições](dados/comprimentos.json) · [Cobertura e recortes](revisao_metodologica.md#por-que-limitar-o-comprimento)

</details>

<details>
<summary>14 · Zero tokens desconhecidos quer dizer que o BPE foi melhor?</summary>

**Resposta.** Na auditoria BPE, os desconhecidos caíram de 3,27% para zero
e a sequência média passou de 121,3 para 129,1 tokens. A acurácia média
foi 82,00% com palavras e 81,41% com BPE. Recuperar peças melhora a
cobertura. Embeddings ainda precisam aprender o valor dessas peças e um
filtro de largura fixa passa a cobrir menos palavras onde há segmentação.
A taxa veio da escolha na CNN por palavras. Esse resultado mede as
duas tokenizações sob a receita usada.
[Slide 11](https://costaleirbag.github.io/projeto-mae5920/#slide-11) · [Resultados pareados](../results/bpe/summary.json)

</details>

### TF-IDF, acertos e erros

<details>
<summary>15 · “Movie was awful” vira um vetor de cinco posições? Como sai a classe?</summary>

**Resposta.** As cinco entradas ativas são `movie`, `was`, `awful`,
`movie was` e `was awful`. O vetor continua com **80 mil atributos**,
um por item do vocabulário fixo. Atributos ausentes têm valor zero
no vetor esparso. Para cada termo presente, o TF vale $1+\log f$ e
o IDF suavizado vale $1+\log[(N+1)/(df+1)]$. Multiplicamos e normalizamos
o vetor pela norma L2. A regressão aprende um peso por atributo e soma
`peso × TF-IDF` mais o viés. No slide, o escore bruto é **−7,302** e a
sigmoide dá **0,067%** para positivo.

**Por que “awful” pesa tanto?** No treino, aparece em 1.151 resenhas e
`movie` aparece em 12.153. Com uma ocorrência de cada no mesmo texto,
os IDFs de 3,85 e 1,49 criam pesos TF-IDF diferentes. A raridade só muda
a magnitude da feature. O sinal negativo vem do coeficiente aprendido
com os rótulos. `was awful` tem TF-IDF maior e contribuição negativa
menor do que `awful` porque seu coeficiente tem menor magnitude.
[Conta com os pesos reais](dados/tfidf_slide.json) · [Código](../src/imdb_lab/baseline.py) · [Definição no scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)

</details>

<details>
<summary>16 · Então o TF-IDF ignora a ordem e é imune ao comprimento?</summary>

**Resposta.** Na nossa receita, os bigramas preservam a ordem de duas
palavras consecutivas. `not good` e `good not` ativam colunas
diferentes. Depois de contar, o vetor perde a localização dos termos e a
ordem entre trechos distantes. Seu tamanho depende do vocabulário e o
vetorizador original recebe a resenha inteira. Acrescentar texto pode
introduzir pistas e mudar a normalização L2 de todas as features
presentes. O modelo varia pouco por faixa de tamanho no IMDb medido.

**Prova com o par do slide.** Trocar os adjetivos muda alguns bigramas
e o escore do baseline passa de **47,93% para 47,86%** de positivo.
A classe continua negativa nas duas versões. Bigramas representam
ordem local, com alcance de duas posições.
[Escores e contribuições](dados/discordancias.json) · [Métricas por faixa](dados/comprimentos.json)

</details>

<details>
<summary>17 · Uma logística com rótulos embaralhados acertou 99,75%. É vazamento?</summary>

**Resposta.** Eram os mesmos textos e os rótulos embaralhados usados
no ajuste. Com 80 mil atributos e 19.823 textos de treino, uma fronteira
linear em um espaço esparso pode separar quase todo o conjunto. O teste
mostra ajuste a rótulos arbitrários, uma forma de memorização. Nos
textos novos com rótulos reais, a validação caiu para **50,08%**.
Com rótulos reais no ajuste, o TF-IDF fez **99,72%** no treino,
**91,53%** na validação e **90,14%** no teste.

**Se pedirem uma checagem de vazamento.** A curva sozinha mede capacidade.
Para auditar vazamento, verifique o papel usado em cada `fit`, as
duplicatas entre divisões e os artefatos que a escolha de hiperparâmetros
podia acessar.
[Controle de rótulos e matriz](../results/tfidf_ablation/summary.json) · [Auditoria do baseline](auditar_baseline.py)

</details>

<details>
<summary>18 · Diminuir a acurácia de treino com L2 teria melhorado o teste?</summary>

**Resposta.** Na logística, reduzir $C$ aumenta a penalização. Na grade
original, $C=0{,}5$ deu 93,25% no treino e 89,39% na validação, e $C=8$
deu 99,72% e 91,53%. Selecionamos pelo F1 macro de validação. Numa
busca posterior, $C=16$ venceu na validação e fez **90,04%** no teste.
Repetir $C=8$ naquele ambiente fez **90,15%**. A avaliação em dados
novos decide se a regularização ajudou. Escolher de novo a partir
desse teste exigiria outra avaliação reservada.
[Busca ampliada](../results/tfidf_regularization/summary.json) · [Seleção](../results/tfidf_regularization/selection.json)

</details>

<details>
<summary>19 · Por que a CNN perdeu? Dá para atribuir o gap a uma causa?</summary>

**Resposta.** O baseline agrega pistas lexicais e bigramas do texto
inteiro. A CNN aprende as features do zero e resume cada filtro pelo
maior pico. Medimos o efeito do recorte reaplicando os mesmos pesos
à resenha inteira. A acurácia passou de **86,78% a 87,28%** no teste
completo e de **82,33% a 86,31%** nas resenhas cortadas. Quando reajustamos
o TF-IDF com a mesma fronteira de 400 tokens por regex, ele fez **89,66%**.
Já um máximo suave, comparado ao máximo global em uma receita posterior,
ganhou **0,60 ponto** a 400 tokens, mantendo os 300 valores da cabeça.
São controles de intervenções diferentes. Eles mostram efeitos locais
de comprimento e agregação. Somar os efeitos como uma decomposição
única exigiria um desenho fatorial, com a mesma receita e as mesmas
amostras, inclusive para medir interações.
[Recorte da CNN](diagnosticos.md#efeito-do-recorte-no-teste-oficial) · [Controle do TF-IDF](../results/tfidf_ablation/summary.json) · [Pooling com capacidade fixa](ensaios_cnn.md#o-que-muda-ao-trocar-só-o-pooling)

</details>

<details>
<summary>20 · Mostre uma resenha que só um deles acertou. A frase do fim causou a mudança?</summary>

**Resposta.** No teste original, só o TF-IDF acertou **1.715** resenhas
e só a CNN, **875**. Em uma negativa de **780 tokens**, a CNN deu
94,7% de positivo com os primeiros 400 e 12,1% com texto inteiro. O
TF-IDF, com os mesmos pesos em cada passagem, caiu de 71,7% para 6,5%.
O contraste altera todos os 380 tokens adicionais. O trecho de crítica
ajuda a interpretar o caso. Para medir o efeito isolado de uma frase,
precisaríamos editar só aquela frase e repetir a inferência.

**Exemplo no sentido inverso.** Em uma resenha positiva de 61 tokens,
os dois leram tudo. A CNN captou `part is brilliant` e fez 96,6% de
positivo. O TF-IDF fez 45,5%. O bigrama `not one` entrou com coeficiente
negativo, embora ali comparasse um enredo com dois. A decomposição da
margem explica a soma naquele texto. Alterá-lo pode mudar outros
termos ou outros máximos ao mesmo tempo.
[Resenhas e contribuições](diagnosticos.md#cnn-e-tf-idf-nas-mesmas-resenhas) · [Recorte interativo](https://costaleirbag.github.io/projeto-mae5920/#recorte)

</details>

### Inicialização, treino e regularização

<details>
<summary>21 · Como a rede aprende em um minibatch? Onde entra o clipping?</summary>

**Resposta.** Buscamos os embeddings, aplicamos os filtros, agregamos
as ativações e calculamos dois logits por texto. A loss compara esses
logits com os rótulos. Zeramos os gradientes acumulados antes de
começar o passo e usamos retropropagação para obter os gradientes.
Se a norma global ultrapassa 1, reescalamos o vetor de gradientes antes
de o otimizador atualizar pesos. Isso controla a magnitude do gradiente
naquela atualização. O otimizador ainda escolhe como transformá-lo em
passo. Uma época é uma passagem por todos os exemplos.

**No quadro.** O max pooling devolve o maior valor de cada mapa. Na
retropropagação de um exemplo, o gradiente desse máximo chega à posição
que ganhou, e os demais filtros podem ter ganhado em outras janelas.
Ao longo de outros exemplos e épocas, janelas diferentes podem receber
gradiente.
[CNN](../src/imdb_lab/models.py) · [Treinador original](../src/imdb_lab/training.py) · [Treinador interno](../src/imdb_lab/sequences.py)

</details>

<details>
<summary>22 · As seis CNNs são seis inferências aleatórias? A semente só muda os pesos iniciais?</summary>

**Resposta.** São seis conjuntos de pesos treinados, três por palavras e
três com BPE. A semente controla os **pesos iniciais**, a **ordem dos
exemplos nos lotes de cada época** e os **sorteios do dropout durante o
treino**. Dentro de cada par palavras/BPE, com a mesma semente,
repetimos os três sorteios e usamos as mesmas resenhas. Os IDs têm
significados próprios em cada vocabulário e os pesos aprendidos evoluem
de modo diferente. Em avaliação, cada modelo dá uma previsão por
texto com dropout desligado.

**Se pedirem um exemplo.** Em *Suicide Squad*, as probabilidades positivas
da CNN por palavras foram **64,71%, 72,64% e 62,58%**. No BPE, foram
**69,34%, 71,77% e 27,86%**, na ordem 17, 29 e 43. O último valor
produziu o único voto negativo entre as CNNs.
[Treino e sorteios](../src/imdb_lab/sequences.py) · [Pareamento](../docs/PROTOCOLO_BPE.md) · [Predições](dados/fora_dominio_resultados.json)

</details>

<details>
<summary>23 · Vocês usaram He em tudo? Por que a escala interessa?</summary>

**Resposta.** Nos seis treinos, os embeddings começam numa normal de
média zero e desvio 0,1, com a linha de padding zerada. As convoluções
usam **He normal** para ReLU. A saída usa **Xavier uniforme** e os vieses
começam em zero. He usa variância $2/\mathrm{fan\_in}$ sob as hipóteses
usuais, para compensar a redução do segundo momento após a ReLU.
No kernel 3, cada neurônio combina $3 \times 128=384$ entradas. A mesma
regra dá desvio $\sqrt{2/384}$ para os seus pesos. Alterar a semente
muda o sorteio com essa escala. A CNN original de 400 tokens foi criada
com a inicialização padrão das camadas do PyTorch.
[Inicialização interna](../src/imdb_lab/sequences.py) · [Criação da CNN original](../src/imdb_lab/training.py) · [He et al.](https://arxiv.org/abs/1502.01852)

</details>

<details>
<summary>24 · Por que o SGD sobe no início? Momentum é mais instável? Adam é sempre mais suave?</summary>

**Resposta.** No slide, a taxa do SGD é **0,3**, escolhida na validação
dentro da grade. Na primeira época, a CE de validação passa de cerca
de **0,74 para 0,84**. O gradiente de um lote é uma estimativa e um
passo grande pode piorar até a loss média de treino, que também subiu
ali. Com SGD a 0,03, o pico inicial desapareceu, mas a CE ao fim do
orçamento ficou pior. Isso sustenta a hipótese de passos iniciais grandes
naquela taxa, sem medir a geometria exata da superfície.

**Se pedirem a teoria.** Momentum carrega parte da direção anterior.
Pode reduzir oscilações entre passos e também ultrapassar regiões boas.
Adam usa médias dos gradientes e de seus quadrados para ajustar a escala
por parâmetro. Com Adam a 0,003, a validação piorou bem mais no fim.
Forma da curva, taxa e orçamento devem ser lidos juntos. As faixas do
slide descrevem diferenças entre sementes.
[Curvas e leituras](roteiro.md#como-ler-as-curvas) · [Dados das curvas](../results/sequences/presentation_data.json) · [Kingma e Ba](https://arxiv.org/abs/1412.6980)

</details>

<details>
<summary>25 · Vocês pararam na época 5 ou 6? E por que usar a loss de validação?</summary>

**Resposta.** Os ensaios internos rodaram as **12 épocas**. Guardamos
depois o checkpoint de menor CE de validação de cada execução. Para
Adam a 0,001, isso ocorreu na quinta ou na sexta época. No gráfico,
da época 6 à 12, a CE média de treino caiu de 0,054 para 0,003 e a
de validação subiu de 0,377 para 0,510. A CE de validação percebe
erros confiantes mesmo quando a classe de outros exemplos continua
igual. Já a CNN original usou até cinco épocas, critério de F1 macro
na validação e paciência dois. Guardou a quinta.
[Históricos internos](../results/sequences/presentation_data.json) · [Histórico original](../results/cnn/history.json)

</details>

<details>
<summary>26 · Houve regularização além de dropout? Weight decay é L2?</summary>

**Resposta.** A CNN original usou dropout 0,5, AdamW com
decaimento de 0,01 e seleção de checkpoint na validação. A grade interna
de CNN usou decaimento zero, experimentou dropout 0, 0,5 e 0,8 e
também selecionou o checkpoint. O TF-IDF tinha L2 na logística.
O AdamW aplica o encolhimento aos pesos separadamente da atualização
adaptativa. Adicionar L2 ao objetivo muda o gradiente que alimenta
Adam. No suplemento MLP, L2 explícita foi outra condição, com
`weight_decay=0` no Adam.

**Se perguntarem qual ajudou.** Dependeu da receita. No ensaio tardio
com máximo a 768 tokens, diminuir dropout de 0,5 para 0,2 elevou
a acurácia de treino em cerca de 1,5 ponto e mudou o teste em
**−0,09 ponto**. Em outra receita a 400, a mesma mudança ganhou
**0,60 ponto** no teste. A unidade de comparação é a receita pareada.
[Regularização das receitas](revisao_metodologica.md#que-regularização-usamos) · [Ensaios novos](ensaios_cnn.md) · [AdamW](https://arxiv.org/abs/1711.05101)

</details>

### Ordem, pooling e robustez

<details>
<summary>27 · O slide dos separadores representa uma resenha plausível? Por que repetir a frase?</summary>

**Resposta.** É um controle sintético para isolar um mecanismo. Cada
versão tem três cópias da frase separadora de oito palavras e dois
blocos de opinião, na ordem `S → elogio → S → crítica → S` ou
`S → crítica → S → elogio → S`. O separador é mais comprido que a
maior janela. Assim, qualquer janela que cruza a borda de uma opinião
encontra o mesmo contexto local nas duas versões. Nem mesmo a janela
da borda precisa ser descartada. As janelas aparecem em outra ordem
e em igual quantidade.

**No quadro.** O filtro 61 de largura 3 tem pico **0,47** em
`movie i loved`. A posição inicial vai de **8 para 20**. O maior valor
fica igual. Para cada um dos 300 filtros, mesma coleção de janelas
significa mesma coleção de respostas, mesmo máximo e mesmo valor
entregue à camada final. Os seis treinos produziram logits iguais entre
A e B. O teste em textos editados mede o alcance em material menos
artificial.
[Controle interativo](https://costaleirbag.github.io/projeto-mae5920/#maximo-global) · [Textos exatos](casos.json) · [Escores do controle](dados/pares_resultados.json)

</details>

<details>
<summary>28 · O máximo devia ser independente do kernel? Aumentar para sete resolve?</summary>

**Resposta.** O máximo é invariante à ordem dos valores **de um mesmo
filtro**. A largura muda quais tokens e pesos entram na resposta
antes da agregação, logo pode mudar o pico. No par com `but`,
uma janela de cinco já pode ver `terrible but it was wonderful`.
Uma de sete dá mais contexto e mais pesos por filtro. A rede ainda
precisa aprender como combinar esse padrão com os demais.

**Se pedirem evidência.** Na grade interna, janela 3 isolada fez
**81,27%**, janela 5 fez **80,94%** e 3/4/5 fez **81,95%**. A última
também tinha mais filtros totais. Reduzir de 100 para 32 filtros por
largura em 3/4/5 levou a **81,64%**. Em outro controle a 400 tokens,
3/5/7 fez **86,87%**, ante **87,11%** da referência 3/4/5.
Ao mudar o passo para 2 ou 3 na varredura, os pesos ficam fixos.
O passo 2 pulou a janela mais forte do filtro 89 e derrubou seu pico
de 0,98 para 0,40. Essas predições medem a omissão de posições depois
do treino com passo 1.
[Largura e passo](diagnosticos.md#largura-e-passo) · [Janela 7 e outras variantes](ensaios_cnn.md#o-efeito-de-treinar-com-mais-texto)

</details>

<details>
<summary>29 · No par com “but”, a culpa é toda do max pooling?</summary>

**Resposta.** Nas duas versões, trocar `terrible` e `wonderful` altera
as janelas e o vetor que chega à camada final. A leitura planejada
segue a segunda oração. As seis CNNs mudaram de classe na direção
da primeira oração. Na versão por palavras, o escore médio positivo
passou de **1,0% a 89,9%**. Com BPE, de **1,2% a 83,7%**.
O máximo pode descartar onde apareceram os sinais, e os filtros
aprendidos e os coeficientes finais também participam da inversão.
A igualdade exata de logits pertence ao controle dos separadores.

**Se perguntarem sobre negação.** Em `good, not bad` versus
`bad, not good`, a troca mudou a classe em duas sementes por palavras
e uma com BPE. Esses pares específicos testam a sensibilidade local a
ordem e escopo. Uma avaliação quantitativa pede uma coleção de pares
e rótulos definidos previamente.
[Par interativo](https://costaleirbag.github.io/projeto-mae5920/#contraste) · [Quatro pares e respostas por semente](dados/pares_resultados.json)

</details>

<details>
<summary>30 · No slide 14, vocês deram à CNN um vetor de posições?</summary>

**Resposta.** Usamos os mesmos embeddings e filtros congelados.
Por filtro, dividimos suas janelas válidas pela metade e extraímos
um pico de cada metade. Uma cabeça recebe `[menor, maior]`. A outra
recebe `[primeira metade, segunda metade]`. Cada cabeça recebe 600
valores. Com os mesmos números em canais de significado diferente,
a posição relativa pode chegar à saída pela ordem das duas entradas.

**Gabarito de leitura.** No controle dos separadores, os escores
mudaram nas três sementes com a cabeça posicional e as classes
continuaram iguais. Nos 344 pares editados, **39,83%** por valor e
**39,15%** por posição. O máximo global com cabeça reajustada fez
**40,02%**. Cada par conta somente se as duas versões forem classificadas
corretamente. Oito pares ficaram com entradas idênticas depois
do recorte e foram mantidos nessa contagem.
[Slide de posição](https://costaleirbag.github.io/projeto-mae5920/#posicao) · [Protocolo, controle e métricas](diagnosticos.md#o-que-acontece-quando-guardamos-a-posição-dos-picos)

</details>

<details>
<summary>31 · E treinando as alternativas de pooling de ponta a ponta, deu mais robustez?</summary>

**Resposta.** Com seis formas de pooling e três sementes cada, todos
os **18 treinos** acertaram zero dos **128 pares com veredito no fim**,
exigindo acerto nas duas versões. O K-max ordenado teve mais acurácia
na auditoria comum (**82,63%**) e menos acerto nos pares editados
(**37,60%**) que o máximo global (**82,00%** e **40,21%**).
Em outro ensaio, ler o texto inteiro com máximo global aumentou
o acerto em pares humanos de **36,34% para 43,41%**, e nos pares
sintéticos com veredito no fim todas as variantes ficaram entre
zero e 1,04%. Guardar posição e aprender qual posição decide são
duas exigências.

**Se pedirem um teste mais isolado.** No máximo suave com pesos
iguais nas posições, a saída ainda é invariante à ordem das janelas.
Em 768 tokens, ele ganhou cerca de meio ponto no IMDb sobre máximo
global, mas foi de **2 para 1 acerto em 384 avaliações de pares**
com veredito no fim. São os mesmos 128 pares avaliados nas três sementes.
A versão com peso maior no fim teve 4/384. A melhora média em resenhas
comuns mede outra coisa.

**Se perguntarem pelo ensaio anterior de comprimento.** Na bateria de comprimento,
o máximo global fez **85,89%** com 256 tokens, **87,11%** com 400,
**87,63%** com 768 e **87,81%** com texto inteiro. Máximo + média com
texto inteiro chegou a **88,18%** naquela bateria. Na bateria posterior,
o máximo suave a 768 ganhou na validação e fez **88,13%** no teste.
Dar peso extra ao fim chegou a **88,28%**, mas teve CE de validação
ligeiramente maior: 0,258514 ante 0,258467. Isso explica qual receita
selecionamos em vez de escolher pela coluna do teste.
[Poolings de ponta a ponta](pooling_ordem.md) · [Texto integral e pooling com cabeça fixa](ensaios_cnn.md)

**E uma CNN maior treinada por mais tempo?** No ensaio novo, a
TextCNN com texto inteiro chegou a **89,60%** e a melhor receita da
residual de 11,13 milhões a **88,37%**. Todos os seis checkpoints
foram escolhidos entre as épocas **2 e 4**, mesmo após treinar
de **8 a 10**. Mais épocas serviram para observar o sobreajuste.
A quantidade de dados ficou fixa, então a hipótese de que ela
é o gargalo precisa de outro controle.
[Ensaio com a CNN maior](ensaios_cnn.md#uma-cnn-maior-e-mais-épocas-ajudam)

</details>

<details>
<summary>32 · Se pudesse rodar um experimento a mais para explicar a diferença, qual seria?</summary>

**Resposta possível.** Eu cruzaria **comprimento** (400 tokens e texto
inteiro), **agregação** (máximo global e duas metades com capacidade
equivalente) e, numa segunda etapa, **supervisão contrastiva**. Fixaria
dados, arquitetura restante, orçamento e regra de seleção. Escolheria
pela validação e reservaria uma avaliação nova, com pares de ordem
anotados e exemplos dos dois sentidos. Reportaria acerto por resenha e
por par. Assim dá para medir se o efeito da posição muda com o recorte
ou com o sinal de treino.

**Se pedirem o que já tentamos.** Acrescentar 384 pares sintéticos ao
treino, com uma condição de repetição de resenhas de mesmo orçamento,
mudou pouco o resultado. No máximo global, houve só um par correto a
mais somando as sementes no veredito final. A transferência para
outros moldes e edições humanas foi limitada.
[Aumento contrastivo e controle](ensaios_cnn.md#treino-com-exemplos-contrastivos) · [Contrast Sets de Gardner et al.](https://aclanthology.org/2020.findings-emnlp.117/)

</details>

### Transformers, custo e mudança de domínio

<details>
<summary>33 · Por que o BERT caiu nas resenhas longas? Eram 192 palavras?</summary>

**Resposta.** Eram **192 posições de WordPiece**, incluindo `[CLS]`
e `[SEP]`, até 190 peças para o texto. O teto cortou **14.747/25.000**
resenhas. Na faixa acima de 500 palavras, todas as **1.845** resenhas
foram cortadas. Repetimos a inferência nesse mesmo grupo com o
checkpoint fixo e o teto de **512 tokens**. A acurácia passou de
**82,60% a 89,70%** e o F1 macro de **82,46% a 89,69%**. Mesmo com
512, todas ainda perderam alguma parte do texto. O ganho mede
mostrar mais texto na inferência nessa faixa, mantendo o ajuste
feito a 192.
[Controle a 192 e 512](dados/recorte_bert/resumo.json) · [Métricas por tamanho](dados/comprimentos.json)

</details>

<details>
<summary>34 · O que o BERT aprendeu antes do IMDb? Por que reutilizar o pré-treino?</summary>

**Resposta.** O BERT-base publicado aprendeu com BooksCorpus e
Wikipédia em inglês. Previa tokens selecionados para mascaramento
e distinguia trechos consecutivos de pares aleatórios. Isso dá ao
encoder pesos que já produzem representações contextuais. Nós
carregamos o checkpoint e o adaptamos aos rótulos de sentimento.
Os autores reportaram **quatro dias em 16 chips TPU** para
o pré-treino original. Nossas duas épocas de ajuste levaram
**446,4 s na DGX Spark**. São tarefas, volumes de dados e máquinas
diferentes. Para isolar o ganho do pré-treino nesse projeto, seria
necessário comparar com o mesmo BERT inicializado aleatoriamente.
[Artigo do BERT, §3.1 e apêndice A.2](https://aclanthology.org/N19-1423.pdf) · [Metadados do ajuste](../results/bert/metadata.json)

</details>

<details>
<summary>35 · “Só a cabeça” quer dizer os últimos blocos do BERT? Por que o batch muda?</summary>

**Resposta.** No modo congelado, todos os blocos ficam fixos. Na extensão,
fazemos a média dos vetores dos tokens válidos, excluindo padding e
especiais, e treinamos uma camada `Linear(768, 2)` de **1.538
parâmetros**. Usamos 40 épocas e lote 128 sobre vetores já calculados.
No modo completo, o gradiente passa pelo encoder, com três épocas e
lote 32. São duas receitas práticas. Na mesma auditoria, BERT fez
**84,52%** congelado e **88,31%** completo. DistilBERT fez
**83,35% / 86,80%** e ModernBERT **81,30% / 91,24%** nos mesmos regimes.

**Se mostrarem o 90,94% no slide 9.** Ali está o ajuste original, com
19.823 resenhas de treino, 192 WordPieces, a cabeça de sequência do
BERT e 25.000 de teste oficial. Na extensão são 6.144 de treino e
2.048 prefixos de auditoria. Lote, orçamento, cabeça e dados
mudam entre esses resultados. O ajuste de só alguns blocos finais
seria uma condição intermediária ainda a medir.
[Regimes, camadas e calibração](../docs/PROTOCOLO_SEQUENCIAS.md) · [Configuração original](../config.json)

</details>

<details>
<summary>36 · Congelar deixa o BERT mais rápido em produção? E o baseline é mesmo mais barato?</summary>

**Resposta.** Congelar acelerou a adaptação. Na mesma GB10, extrair
os vetores para treino e validação levou 15,2 s e treinar a cabeça
levou 2,35 s. A soma foi **17,6 s**, ante **148,9 s** de ajuste
completo. Ambos os tempos excluem as passagens de avaliação por
época. Com novos textos, os dois modelos ainda executam o encoder.
Para inferir nos mesmos 2.048 prefixos tokenizados, levaram
cerca de **4,3 s e 4,0 s**, respectivamente.

**Se pedirem o custo do teste original.** O TF-IDF, com vetorização
e busca de $C$, levou **13,1 s na CPU do M1** e teve 90,14% no teste.
A CNN levou **139,6 s na GPU do M1**, em cinco épocas, e fez 86,78%.
O BERT levou **446,4 s na GB10**, em duas épocas, e fez 90,94%.
Os cronômetros das redes incluem a validação por época e partem das
entradas já tokenizadas. O custo de 0,15 s de inferência logística
registrado é sobre matrizes já vetorizadas. Para comparar latência
de uso, mediríamos texto bruto até classe, nas mesmas condições de
hardware e lote.
[Cronômetros e escopo](roteiro.md#como-os-tempos-foram-medidos) · [Slide de custo](https://costaleirbag.github.io/projeto-mae5920/#custo)

</details>

<details>
<summary>37 · O Qwen fez 95,26%. Isso prova que uma rede maior sempre ganha? Ganhou em todos os casos?</summary>

**Resposta.** O Qwen ganhou no agregado do mesmo teste. Acertou
**1.914** resenhas que o TF-IDF errou, e o TF-IDF acertou **635**
que o Qwen errou. O Qwen recebe até 768 tokens da fonte e combina
outra arquitetura, outro pré-treino, outro tamanho e outro formato
de predição. O teste já havia sido consultado neste projeto e a
exposição no pré-treino do Qwen segue desconhecida. Isso é evidência de
desempenho de uma receita em uma tarefa, com custo e condições próprios.
[Comparação por ID](revisao_metodologica.md#o-que-o-qwen-mostra) · [Protocolo Qwen](../QWEN_EXPERIMENTO.md)

</details>

<details>
<summary>38 · O YouTube mede acurácia fora do domínio? O Qwen com raciocínio acertou mais?</summary>

**Resposta.** O slide do YouTube mostra dois casos qualitativos,
sem rótulo de referência. Em *The Room*, `No one` vira um elogio
citado em crítica na leitura da equipe. Os seis checkpoints da CNN
e as seis gerações do Qwen deram positiva. Em *Suicide Squad*,
cinco CNNs deram positiva e o Qwen deu negativa nas três gerações
de cada modo. Os modelos viram só o texto, e a plateia tem o
contexto do trailer. O rótulo final coincidiu entre os modos para
esses comentários. O teste rotulado usa outras fontes, 1.000
frases da Amazon e 1.000 do Yelp, com cerca de **68–70%** de
acurácia nas CNNs.
[Slide do YouTube](https://costaleirbag.github.io/projeto-mae5920/#slide-15) · [CNNs fora do domínio](dados/fora_dominio_resultados.json) · [Gerações do Qwen](dados/qwen_youtube_resultados.json)

</details>

### Qwen, steering e conexões com a disciplina

<details>
<summary>39 · O Qwen virou classificador treinado? Onde entra o raciocínio?</summary>

**Resposta.** Para as 25.000 resenhas, o Qwen3-8B congelado recebe uma
instrução para responder com um dos dois rótulos. Lemos os logits
dos tokens `negative` e `positive` na primeira posição da resposta,
com raciocínio desativado. Aplicamos um softmax restrito às duas opções.
Isso equivale a uma sigmoide da diferença dos logits. Calibramos a
margem em outra amostra. Se o modelo quiser emitir outro token, o
escore restrito a esses dois ainda pode ser alto. Por isso também
registramos a massa dos dois rótulos no vocabulário completo.

**No YouTube foi outra leitura.** Ali geramos a resposta final em
três amostras por modo, com a mesma instrução. O modo com raciocínio
permite ao modelo gerar texto interno antes do rótulo. A comparação
avalia o rótulo extraído ao fim, sem mudar os pesos e sem usar steering.
[Classificação por logits](../QWEN_EXPERIMENTO.md#1-classificação-do-teste-inteiro) · [Código das gerações do YouTube](medir_qwen_youtube.py)

</details>

<details>
<summary>40 · Como vocês descobriram a direção? É embedding, gradiente ou mudança de pesos?</summary>

**Resposta.** Na rodada recente, para cada um de **128 enredos de
treino**, passamos uma instrução de avaliação positiva e outra
negativa. Capturamos a ativação **contextual de 4.096 dimensões** na
última posição do prompt, no bloco **28**. Tiramos a média dos 128
vetores positivos, subtraímos a média dos 128 negativos e
normalizamos para norma 1. O pareamento mantém o enredo de cada
exemplo. As instruções dão o sinal do contraste.

**Se perguntarem pela RFM.** A primeira rodada usou uma sonda kernel
treinada em ativações de 512 resenhas rotuladas. A AGOP é a média de
produtos externos dos gradientes **do preditor auxiliar**, e usamos o
autovetor principal orientado para a classe positiva. A sonda aprende
como ler um atributo. O hook intervém somando um vetor à ativação de
uma camada. Os parâmetros do Qwen ficam fixos nos dois processos.
[Rodada atual e primeira RFM](../QWEN_EXPERIMENTO.md) · [Referência do método](https://arxiv.org/abs/2502.03708)

</details>

<details>
<summary>41 · Se o prompt fica igual, o que muda entre negativo, zero e positivo?</summary>

**Resposta.** Na geração recente, o prompt comum pede um trecho
`FACT` e um trecho `REVIEW`. O hook espera o marcador `REVIEW:`
ser gerado e então soma $\alpha s v$ às ativações do bloco 28
durante a geração da opinião. $v$ é a direção unitária e $s$ a
escala aprendida no treino. Mantemos fonte, prompt e pesos do
modelo. Em $\alpha=-2$ subtraímos $2sv$, em zero temos o controle e em
$+2$ somamos. A geração é autoregressiva em todos os casos. A
analogia com GANs e flows é a intervenção em uma representação
interna. Aqui ela é por camada e posição de token.

**Um trio para falar em voz alta.** No mesmo enredo sintético,
*Signal 2040*, o controle negativo critica a coerência e o ritmo.
Com zero, a opinião elogia o andamento da história. O positivo
elogia a narrativa e as atuações. O texto do campo `FACT` sai
igual nas três. O zero é a resposta padrão, que pode ter opinião.
[Trio registrado](../results/qwen_improved/presentation_examples.json) · [Galeria das saídas](../relatorio_qwen3.html)

</details>

<details>
<summary>42 · Como vocês sabem que o steering funciona? Ele é melhor que pedir no prompt?</summary>

**Resposta.** A pergunta do seminário é se uma mudança em ativações
altera a geração com a mesma entrada. Testamos o controle zero e
direções aleatórias. A direção e intensidade escolhidas na validação
foram aplicadas a **64 fontes** da rodada de geração. O próprio Qwen
sem hooks, como juiz automático, marcou os dois sentidos-alvo em
**55/64** fontes. Isso inclui 24/32 resenhas reais e 31/32 enredos
sintéticos. Os tokens de `FACT` coincidiram com o zero nas
intervenções localizadas. `REVIEW` ainda pode introduzir
informação inventada. Validação humana independente seria
necessária para confirmar sentimento e fidelidade.

**Se perguntarem pelo prompt direto.** É um controle em que muda a
instrução. A demonstração mantém a instrução fixa para identificar
o efeito de uma intervenção interna. Comparar eficácia prática com
prompting envolveria alinhar tarefa, critérios de qualidade e custo.
[Protocolo e limites](../QWEN_EXPERIMENTO.md) · [Diagnósticos por condição](../results/qwen_improved/test_summary.json)

</details>

<details>
<summary>43 · Onde apareceu double descent? Qual é a relação com o gráfico de loss da CNN?</summary>

**Resposta.** No suplemento, usamos **64 amostras de treino**
de regressão sintética, ruído de desvio 0,3 e features aleatórias
ReLU congeladas. Variamos o número $p$ de coeficientes da saída.
Perto de $p=64$, a solução passa a poder ajustar até o ruído e
pode ter coeficientes grandes quando o sistema está mal
condicionado. Com mais features, a solução de norma mínima
pode voltar a generalizar melhor. Ridge atenuou o pico. Foram
880 condições em 20 sementes, resumidas por mediana e intervalo
interquartil.

**Se apontarem a curva da CNN.** Ela varia épocas de uma rede com
capacidade fixa. Mostra a perda de treino caindo e a de validação
voltando a subir. É o caso que motiva guardar a melhor época.
A evidência de uma segunda queda com capacidade vem do experimento
sintético de regressão.
[Desenho e curvas do suplemento](../docs/CONEXOES_DISCIPLINA.md) · [Belkin et al.](https://doi.org/10.1073/pnas.1903070116)

</details>

<details>
<summary>44 · Para que fazer MLP, conexões residuais e GloVe se a CNN era o centro?</summary>

**Resposta.** Nas MLPs, fixamos a representação de 128 dimensões
TF-IDF/SVD e variamos o treinamento. Isso deixa comparar profundidade,
atalhos, escala da parte residual, otimizador e regularização com
a mesma entrada. As três redes de 12 camadas começam com as mesmas
matrizes e o mesmo número de parâmetros. A rede profunda pode
encaixar melhor o treino e generalizar pior. O diagnóstico separado
de 32 camadas inspeciona escala de ativações e gradientes antes de
qualquer atualização.

**Sobre GloVe.** É um teste de transferência com embeddings estáticos
numa TextCNN. Em 100 dimensões e com a mesma receita, obtivemos
**87,49%** com tabela aleatória, **87,63%** com GloVe embaralhado
entre palavras e **87,83%** com o vetor associado à palavra certa.
O embaralhamento ajuda a controlar ter aqueles valores iniciais sem
preservar a associação semântica. Ajustamos os embeddings nos três
casos. No BERT, o vetor de saída de cada posição já depende do contexto.
[Suplemento MLP](../docs/CONEXOES_DISCIPLINA.md) · [Ensaio GloVe](ensaios_cnn.md#embeddings-pré-treinados)

</details>

### Autoria, verificação e perguntas de encerramento

<details>
<summary>45 · O que exatamente a IA fez? Como vocês sabem o que está acontecendo no notebook?</summary>

**Resposta.** A declaração do relatório registra assistência do
Codex na implementação, execução, análise e preparação dos materiais.
O Qwen3-8B também foi usado como modelo experimental. Para mostrar
domínio, cada pessoa pode abrir o trecho de código e refazer no
quadro um cálculo que conferiu pessoalmente. Por exemplo, explicar
como saem os 300 máximos ou distinguir o resultado do recorte
reavaliado de um novo treino.

**No ensaio.** Cada pessoa completa em voz alta “Eu conferi ___ a
partir de ___ e encontrei ___”. Preencham com o que de fato
verificaram. Escolham uma hipótese que os dados contrariaram,
como “BPE vai subir a acurácia” ou “guardar a metade vai
resolver os pares”. Expliquem o que o controle mostrou e qual
conclusão vocês tiraram dele.
[Declaração de IA no relatório](partes/01-pergunta.tex) · [O que o notebook executa](../docs/NOTEBOOK_UNICO.md)

</details>

<details>
<summary>46 · Se eu pedir para conferir um número agora, de onde ele sai?</summary>

**Resposta.** Por exemplo, os escores de `Movie was awful` vêm do
vetorizador e dos coeficientes salvos. Somo os cinco produtos da
tabela com o viés e chego a −7,302. Para os **86,78%** do modelo
original, uso as predições por ID do teste e comparo classes aos
rótulos. Para as CNNs do YouTube, abro os seis pares de logits
e comparo as classes por semente. As evidências vêm com metadados e
hashes que identificam os artefatos.

**Se pedirem para reproduzir.** O notebook único executa seus
exercícios em CPU e abre resultados preservados. Há verificadores
para recalcular métricas e testes como invariância a padding e à
temperatura na classe prevista. Repetir os treinos grandes exige o
corpus, o ambiente e, quando aplicável, restaurar os pesos históricos.
[Conta do TF-IDF](dados/tfidf_slide.json) · [Métricas e metadados da CNN](../results/cnn/metrics.json) · [Reprodução](../docs/REPRODUCAO.md) · [Testes científicos](../tests/test_scientific_invariants.py)

</details>

<details>
<summary>47 · E se a implementação da CNN estiver errada? Qual limite vocês conseguem demonstrar?</summary>

**Resposta.** Um resultado baixo justifica investigar o código e a
receita. Conferimos dimensões, padding, modo de avaliação e predições
pareadas. O teste de padding verifica que aumentar apenas o
preenchimento à direita preserva o resultado. A igualdade nos logits
no par de janelas iguais é uma propriedade esperada do máximo
global, e tem um teste com os pesos reais. No par com `but`,
contudo, as janelas mudam, e atribuir a falha exclusivamente ao
pooling ultrapassaria o que esse caso mostra. Juntas, as evidências
localizam uma limitação da arquitetura e quantificam efeitos de
algumas escolhas de treino e entrada.
[CNN e máscara](../src/imdb_lab/models.py) · [Teste de padding](../tests/test_scientific_invariants.py) · [Troca de blocos](dados/pares_resultados.json)

</details>

<details>
<summary>48 · Qual modelo vocês escolheriam e qual conclusão tiram sobre Deep Learning?</summary>

**Resposta possível.** Para classificação em massa de resenhas
semelhantes às do IMDb e com orçamento de CPU, escolheríamos o
TF-IDF com logística a partir desta comparação. Ele foi rápido,
forte e permite inspecionar as contribuições de termos. Para
estudar contexto e adaptação a outras tarefas, o BERT é uma outra
receita, e o Qwen teve o maior acerto no teste deste trabalho.
Em uso real, eu pediria uma avaliação anotada do domínio-alvo,
com latência de ponta a ponta e um tratamento explícito para
comentários ambíguos.

**Conclusão do trabalho.** A TextCNN aprende ordem dentro da
janela. Global max pooling conserva a força do melhor sinal de
cada filtro e perde onde ele ocorreu. Recorte tira evidências de
algumas entradas. Nos controles, conseguimos medir esses efeitos
mesmo quando eles explicam só parte da diferença para o baseline.
[Teste oficial](../docs/RESULTADOS.md) · [Diagnósticos](diagnosticos.md)

</details>

## Código para abrir na hora

| Se pedirem para ver… | O que procurar | Arquivo |
|---|---|---|
| **A ordem das operações na CNN** | `TextCNN.forward` e `PadBatch.__call__`. Passe do comprimento real à máscara e aos dois logits. | [Modelo e padding](../src/imdb_lab/models.py#L35-L80) |
| **Onde o baseline aprende** | `fit_transform` no treino, `transform` nas demais divisões, seleção de `C` na validação. | [TF-IDF e regressão](../src/imdb_lab/baseline.py#L13-L32) |
| **De onde vêm os sorteios** | `orders`, `initialize_cnn` e `train_one`. Aponte para a permutação, os inicializadores e a semente do dropout. | [Lotes](../src/imdb_lab/sequences.py#L128-L133) · [Inicialização](../src/imdb_lab/sequences.py#L190-L199) · [Treino](../src/imdb_lab/sequences.py#L235-L280) |
| **Como atribuímos classe ao comentário** | `load_models` carrega dois checkpoints por semente. `predict` usa o `argmax` dos logits. | [Inferência das CNNs](demo.py#L41-L102) |
| **Onde entra a posição** | Compare a cabeça com picos por valor e a cabeça com picos por metade. Localize a avaliação que exige dois acertos por par. | [Experimento do slide 14](medir_posicao.py) |
| **Qual token dá a classe do Qwen** | `forward_labels` resolve os IDs de rótulo, lê a última posição do prompt e guarda os dois logits e sua massa total. | [Inferência por logits](../src/imdb_lab/qwen_core.py#L139-L166) |

## Ensaio em grupo

### Rodada de 12 minutos

1. **Quatro minutos de quadro.** Sem abrir as respostas, desenhem os
   tensores de um texto de dez tokens até os dois logits. Depois
   desenhem as cinco colunas ativas do TF-IDF de `Movie was awful`.
   Usem os cartões 09, 10, 12 e 15 para conferir.
2. **Cinco minutos de banca.** Cada pessoa pega uma pergunta difícil
   dos cartões 02, 17, 24, 27, 30, 33, 39 e 42. Dê a conclusão em
   uma frase e defenda com um controle e um limite da evidência.
3. **Três minutos de demonstração.** Abram a varredura da CNN e mudem
   para o filtro de `wonderful`. Depois mostrem A e B no máximo
   global e expliquem por que a posição muda sem mudar o pico.
   Terminando, abram um resultado salvo pelo link do gabarito.

[Varredura](https://costaleirbag.github.io/projeto-mae5920/#slide-5-varredura) · [Controle da troca](https://costaleirbag.github.io/projeto-mae5920/#maximo-global)

### Três contas sem computador

| Exercício | Resposta para conferir |
|---|---|
| Uma frase de dez tokens, janela 3 e passo 2. Quantas janelas completas? | Quatro, começando nos tokens 1, 3, 5 e 7. Com passo 1 seriam oito. A janela que começa no token 4 fica de fora quando o passo é 2. |
| Um filtro dá as respostas 0,20 e 0,95 na primeira metade e 0,10 e 0,30 na segunda. E se trocarmos as metades? | O máximo global fica em 0,95. Os máximos por metade passam de (0,95 e 0,30) para (0,30 e 0,95). A média global também permanece igual. |
| A margem é +2. O que acontece se a temperatura for 2? | A margem vira +1. A probabilidade positiva vai de aproximadamente 88% a 73%. A classe positiva se mantém. |

### Se faltar uma resposta na hora

Use o resultado que vocês efetivamente mediram e proponha o controle
que responderia à pergunta. Por exemplo, “O teste de 344 pares inclui
edições diferentes. Para estimar a robustez só à ordem em resenhas
naturais, eu coletaria novos pares nos quais a edição só troca a ordem
e a classe esperada foi validada por anotadores.”

**Para ligar ao livro.** Prince, caps. 6–10 para otimização,
inicialização, generalização, regularização e convoluções. Caps. 11–12
para residuais e Transformers. Os capítulos e os experimentos
relacionados estão reunidos no guia do projeto. [Livro](https://udlbook.github.io/udlbook/) · [Mapa das conexões com a disciplina](../docs/CONEXOES_DISCIPLINA.md)
