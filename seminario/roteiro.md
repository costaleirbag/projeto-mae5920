# Roteiro de fala

A apresentação termina no slide 17, **Conclusão**. Na versão de 12 minutos,
vá do slide 14 direto para **Conclusão** pelo link no topo. Na de 15, passe
também pelos slides 15 e 16. A explicação do TF-IDF entra na sequência
principal antes da comparação no IMDb. Os slides 4 e 7 são pontos
naturais para trocar de pessoa.

| Slide | Relógio | Assunto |
|---:|---|---|
| 1 | 00:00–00:20 | Pergunta do seminário |
| 2 | 00:20–01:15 | Duas frases e a hipótese de ordem |
| 3 | 01:15–02:00 | Amostras e comprimentos de resenha e frase |
| 4 | 02:00–02:50 | Palavras e BPE |
| 5 | 02:50–04:15 | Arquitetura da TextCNN |
| 6 | 04:15–05:25 | Taxa, época e generalização |
| 7 | 05:25–06:40 | Como TF-IDF e regressão logística classificam |
| 8 | 06:40–07:10 | Controle com rótulos embaralhados |
| 9 | 07:10–07:50 | Teste oficial e controle de recorte |
| 10 | 07:50–08:30 | Acurácia e F1 macro por comprimento |
| 11 | 08:30–09:05 | Auditoria de regex e BPE |
| 12 | 09:05–10:05 | A predição do par inicial |
| 13 | 10:05–11:00 | O limite do máximo global |
| 14 | 11:00–11:35 | Dois picos por filtro e acerto em pares |
| 15 | 11:35–13:00 | Comentários do YouTube ou demonstração |
| 16 | 13:00–14:30 | Encoders e geração com o Qwen |
| 17 | 14:30–15:00 ou 11:35–12:00 | Conclusão e perguntas |

## 1 · Pergunta do seminário

“A tarefa do nosso projeto é classificar resenhas de filmes do IMDb em positivas
ou negativas. Vamos usar uma CNN para acompanhar o que acontece entre ler uma
frase e devolver uma classe. A pergunta é quanto da ordem da frase chega
à decisão.”

## 2 · O contraste na resenha

Leia as duas frases e faça uma pausa breve antes de avançar.

“Na primeira, eu achei que o filme era terrível, mas terminei dizendo que foi
maravilhoso. Na segunda, a opinião final é terrível. Trocamos só os lugares de
*terrible* e *wonderful*. Se eu representasse a resenha apenas pela contagem
de palavras, as duas teriam a mesma representação. Nosso TF-IDF inclui bigramas,
então já tem algum acesso à ordem local. A CNN olha janelas de vários tokens e
pode reagir à troca. Guardem a expectativa sobre A e B. Vamos voltar ao que
os modelos fizeram depois de ver como eles foram treinados.”

## 3 · Treino, validação e avaliação

“Temos 25 mil resenhas em inglês no teste oficial. As resenhas têm, em
média, 226 palavras no teste e 231 no nosso treino. Uma frase típica tem
mediana de 16 palavras, contando pontuação como separador de frases.
Uma em cada dez resenhas do teste passa de 439 palavras. Antes de olhar
o teste, escolhemos a configuração na validação e a temperatura na
calibração. Os ensaios de tokenização usam uma auditoria própria.”

## 4 · Fusões do BPE

No HTML, use **Próxima etapa** ou a seta direita para mostrar a primeira
fusão, `r + e → re`. Avance no ritmo da fala ou arraste até a última etapa
ao falar das três peças. No PDF, aponte as três linhas da sequência.

“Treinamos a mesma arquitetura duas vezes por semente, com tokenizações
diferentes. As duas passam pela mesma expressão regular para separar
palavras, contrações e números. Na primeira, buscamos a palavra inteira no
vocabulário. *Pallbearer* ficou de fora e vira desconhecido. Na outra, o BPE
recebe *pallbearer* escrita, parte dos caracteres e aplica fusões aprendidas
no treino. Saem *p*, *all* e *bearer*. Uma janela de largura três pode cobrir
só essa palavra. Fixamos 20 mil IDs para igualar o tamanho dos embeddings.”
[Pré-processamento e treino do BPE](../src/imdb_lab/bpe_study.py#L31-L43).

## 5 · Arquitetura da TextCNN

Comece em **Montagem** e use a seta direita para avançar junto com a fala.
Aponte para o trecho de código que aparece em cada etapa. Reserve cerca
de um minuto para a montagem e o restante para a varredura.

“Cada ID busca um vetor aprendido de 128 dimensões. As janelas de três,
quatro e cinco tokens leem esses mesmos vetores em paralelo. Cada ramo tem
cem filtros e a ReLU zera respostas negativas. A máscara exclui as janelas
criadas pelo padding. O máximo de cada filtro conserva a força e descarta
a posição. Juntamos os 300 máximos, usamos dropout no treino e aprendemos
uma camada linear com dois logits. A entropia cruzada recebe os logits no
treino. Para exibir a previsão, o softmax dá as probabilidades. A positiva
também pode ser calculada pela sigmoide da diferença dos logits.”

Abra **Varredura** na última etapa. Em A, com largura 3 e passo 1, salte
pela barra até *movie was terrible*. O filtro chega a 0,98. Troque para
passo 2 e vá ao máximo. A janela do pico fica de fora e o valor cai para
0,40. No PDF, use o diagrama completo e o gráfico do mesmo filtro.

Se perguntarem por que o pico acompanha *terrible*, cada posição da
janela tem seu próprio vetor de pesos. O filtro 89 responde 0,98 a
`movie was terrible`, 0,40 a `was terrible but` e zero a `terrible but it`.
A janela inteira entra na conta. Ao mudar a largura, o slide também
troca de filtro. O padrão na frase inteira pode aparecer em qualquer
posição desde que a janela chegue até ele.

Use **Ver o filtro 6 com wonderful** para mostrar o pico de 0,45 em
`it was wonderful`. Esse filtro favorece a classe positiva na frase A.
Nos três ramos, 57 filtros atingiram um pico positivo na última janela
da frase A. Cada pico ainda é multiplicado pelo peso da camada linear.
Somados todos os filtros e o viés, a frase A ficou com classe negativa.
[Ativações medidas](dados/animacoes.json).

## 6 · Escolha da taxa e da época

“Com a arquitetura fixa, variamos SGD, momentum e Adam. Cada método teve três
taxas candidatas. Para cada semente, repetimos a partida e a ordem dos minibatches.
Primeiro escolhemos a época de menor perda de validação de cada treino.
Depois escolhemos a taxa pela média desse critério nas três sementes.
À esquerda estão as curvas das taxas escolhidas. À direita vemos uma
curva de treino que segue caindo depois que a de validação começa a subir.
Isso dá um motivo concreto para guardar os pesos das épocas cinco ou seis
no Adam, mesmo depois de executar as doze épocas. Também testamos larguras,
número de filtros e dropout. As grades completas estão no relatório.”

Se perguntarem sobre regularização, a CNN original usou AdamW com
decaimento de pesos de 0,01 e dropout de 0,5. As CNNs deste slide usaram
decaimento zero e guardaram o checkpoint escolhido na validação. O clipping
de norma 1 protege a atualização contra gradientes grandes.

### Como ler as curvas

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

## 7 · Da resenha à previsão com TF-IDF

No HTML, comece em **Contar** e avance com **Próxima etapa** ou a seta
direita. No PDF, leia as colunas da esquerda para a direita.

“Em *Movie was awful*, são três palavras e dois pares consecutivos.
Cada um ocupa uma coluna. Todos aparecem uma vez, então o TF é 1.
*Awful* aparece em bem menos resenhas de treino que *movie*, por isso
o IDF dele é maior. Multiplicamos TF por IDF em cada coluna e dividimos
os cinco produtos pelo comprimento do vetor, a norma L2. Isso gera
os cinco valores de entrada da logística.

A logística tem um peso aprendido para cada coluna. O TF-IDF de *awful*,
0,451, vezes o peso −10,744 contribui com −4,846. Fazemos o mesmo para as
outras quatro colunas, somamos os cinco produtos e adicionamos o viés.
Dá −7,302. A sigmoide transforma esse escore em cerca de 0,067% de
positivo. A frase foi escrita para o slide e todas as contas usam o
modelo original.” [As cinco contribuições](dados/tfidf_slide.json).

## 8 · Ajuste com rótulos embaralhados

“Embaralhamos só os rótulos do treino. As mesmas resenhas recebem
rótulos aleatórios. Com 80 mil atributos, a regressão conseguiu ajustar
99,75% deles, mesmo com L2. Os coeficientes podem usar combinações de
termos que distinguem os textos conhecidos. Na validação, com os
rótulos reais, fez 50,08%. Com rótulos reais no treino, a mesma receita
fez 91,53% na validação. A primeira coluna mostra que havia um sinal
que se repetia em textos novos.”

Se perguntarem se a gente devia regularizar mais, \(C\) menor significa
penalização L2 mais forte. Isso baixou acurácia no treino e F1 de
validação. Na busca ampliada, \(C=16\) venceu por pouco na validação e
ficou em 90,04% no teste. O controle com \(C=8\) fez 90,15% no mesmo
ambiente. [Busca de regularização](../results/tfidf_regularization/summary.json).

## 9 · Teste oficial do IMDb

“No estudo original, o TF-IDF leu resenhas inteiras e fez 90,14%. A
CNN leu até 400 tokens por regex e fez 86,78%. O BERT leu até 192
WordPieces e fez 90,94%. Depois fizemos um controle à parte. Na repetição,
o TF-IDF fez 90,15% com texto inteiro e 89,66% quando cortamos o texto aos
400 tokens da CNN e reajustamos o modelo. Mesmo com o prefixo comum,
a CNN original fica atrás.”

Acurácia e F1 macro usam as classes previstas. A CE usa as probabilidades
calibradas numa amostra separada.

<a id="configuracao-original"></a>

### Configuração do estudo original

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

## 10 · Desempenho por tamanho

“Nas primeiras duas faixas, a CNN leu todas as resenhas inteiras. O TF-IDF já
fica à frente. A maior diferença acontece acima de 500 palavras, em
1.845 resenhas. Com o teto, a CNN faz 81,52%. Ao deixar os mesmos pesos
lerem o texto inteiro, sobe para 86,18%. O BERT também melhora com mais
texto. Na aba F1 macro, o prefixo de 512 tokens leva o mesmo BERT de
82,46% a 89,69%, ainda com recorte.”

Para o BERT, o teto de 192 inclui `[CLS]` e `[SEP]`, então sobram
190 posições para o texto. Todas as
resenhas acima de 250 palavras passam do teto. O resultado com 512 foi
medido na faixa acima de 500 palavras.
[Controle do BERT](dados/recorte_bert/resumo.json).

No HTML, marque **Mostrar CNN com texto inteiro** para mostrar o controle.
Se faltar tempo, leia só a última linha e a comparação em destaque.

## 11 · Auditoria de regex e BPE

“Aqui os dois tokenizadores enfrentam a mesma auditoria interna. Usamos
a mesma arquitetura, orçamento e taxa de Adam, com três sementes pareadas.
A figura liga a versão por palavras à BPE dentro de cada semente. BPE
reduziu de 3,27 por cento para zero a fração de tokens desconhecidos.
No mesmo texto, a entrada passou de 121 para 129 tokens em média.
Na mesma comparação, a acurácia média foi de 82 por cento para 81,41.
Uma janela de cinco subpalavras pode cobrir menos palavras do que a de
cinco palavras completas. E a taxa foi escolhida originalmente na CNN por
palavras. Os números mostram o que ocorreu sob esse controle. Os escores
dos pares ajudam a ver o que mudou em frases específicas.”

## 12 · Dois adjetivos trocam de lugar

Comece na frase **A** e aponte a conclusão. Depois clique em **B · Trocar
adjetivos** ou pressione a seta direita. Volte a A se quiser refazer a troca.

“É o mesmo par do início. Em A a conclusão diz que o filme foi maravilhoso.
A CNN por palavras dá 1% de escore positivo e a de BPE, 1,2%. Os seis
treinos deram negativa. São dois tokenizadores, cada um com três sementes.
Agora vejam os dois adjetivos trocarem de lugar. A conclusão de B diz que
o filme foi terrível. Os escores pulam para 89,9% e 83,7%. Todos os
treinos passam para positiva. Neste par, a classe prevista coincidiu
com o sentimento da primeira oração nas duas versões.”

Se perguntarem sobre uma janela maior, em ambas as tokenizações a janela de
cinco tokens já contém *terrible but it was wonderful*. Uma de sete incluiria
também *movie was*. A troca muda as janelas, então esse teste mede a rede
inteira. [Tokens e classes](dados/pares_resultados.json).

## 13 · O máximo global descarta a posição

Mostre A e depois pressione **B** ou a seta direita. No PDF, compare as duas
linhas e aponte a posição e o valor do pico.

“Montamos dois textos de controle. A mesma frase vem antes, entre e depois
das opiniões. Assim, cada opinião tem os mesmos vizinhos quando troca de
lugar. A janela *movie i loved* começa no token 8 em A e no 20 em B. O
filtro dá 0,47 nos dois casos. Cada um dos 300 filtros guarda seu próprio
máximo. Como as janelas são as mesmas, os 300 valores e a saída coincidem.”
[Código do controle](medir_pares.py).

## 14 · Guardar a posição ajudou a classificar?

“Pegamos um pico por metade. Ordenados pelo valor, A e B dão 0,25 e
0,47. Na ordem do texto, o primeiro número vem do início e o segundo vem
do fim. Isso mudou os escores do controle, e a classe coincidiu. Cada
regressão recebe 600 valores. Nos 344 pares editados por pessoas, o
acerto ficou perto de 40% nas duas opções. O par conta quando acertamos
as duas versões.” [Código do teste de posição](medir_posicao.py).

Se perguntarem sobre o teste sintético, há quatro moldes com veredito no
começo ou no fim. Nenhuma das cabeças acertou as duas frases de qualquer
par com veredito no fim. [Método e resultados por semente](diagnosticos.md#o-que-acontece-quando-guardamos-a-posição-dos-picos).

## 15 · Comentários fora do IMDb

Mostre os prints e aponte o destaque em *No one*. O modelo recebeu o
texto de cada comentário. Os vídeos e a interface ajudam a plateia a
interpretá-lo. [Trailer de The Room](https://www.youtube.com/watch?v=cG_e8k9PILM&t=2s) · [Friday](https://www.youtube.com/watch?v=kfVsfOSbJY0).

“Em *The Room*, *No one* faz a citação soar como crítica. A CNN e o Qwen
previram positiva. Em *Friday*, a frase pode soar como admiração ou
ironia. Os dois deram negativa. No mesmo Qwen, pedimos o rótulo direto
ou deixamos o modelo gerar primeiro um bloco de raciocínio. Com o mesmo
texto e a mesma instrução, fizemos três gerações por modo e por comentário.
As classes coincidiram. A avaliação com rótulos fora do IMDb usa 2.000
frases de Amazon e Yelp. As CNNs ficaram entre 68 e 70 por cento.”

[Resultados das gerações](dados/qwen_youtube_resultados.json) · [Código](medir_qwen_youtube.py).

Se optarem pela demonstração, abram o navegador antes da fala e usem até
30 segundos para obter uma resenha curta em inglês da plateia. Mostrem a
classe, o escore e os tokens das duas CNNs. Se a resposta demorar, voltem
aos resultados salvos no slide.

## 16 · Encoders e geração com o Qwen

“Aqui são 2.048 resenhas internas, cortadas em 128 palavras. Os vetores
já trazem contexto dos outros tokens. Fazemos uma média mascarada e uma
camada linear classifica. À esquerda, só essa camada aprende, com lote
128. À direita, ajustamos o encoder também, com lote 32. O BERT original
do slide 9 também usou lote 32, com mais dados e outro conjunto de teste.

O quadro do Qwen remete ao teste oficial. Com pesos congelados e até 768
tokens da resenha, usamos o prompt e a diferença dos logits dos dois
rótulos. Ele atingiu 95,26%. Pré-treino, arquitetura e entrada mudam
juntos. A geração com intervenção nas ativações fica no apoio.”

## 17 · Conclusão

“A nossa CNN ficou 3,36 pontos abaixo do TF-IDF. O texto inteiro recupera
meio ponto, com os pesos fixos. O máximo global perde a posição, como
mostra a troca de blocos. No par com *but*, os filtros responderam à
troca, mas a CNN seguiu a primeira opinião. BERT e Qwen mostram que outras
receitas de deep learning superaram o baseline no mesmo teste.”
Deixe o quadro de perguntas na tela.

## Cortes para o ensaio

- Na versão de 12 minutos, vá do slide 14 à conclusão pelo link no topo.
- Para 10 minutos, apresente só a última linha no slide 10 e resuma a auditoria BPE em uma frase. Mostre uma troca de posição no slide 13.
- Na seção de apoio, usem o primeiro slide para mostrar o recorte da resenha, o segundo para calibração e o quarto para largura e passo.
- Se quiserem mostrar a geração do Qwen nos 15 minutos, reservem os últimos 30 segundos do slide 16 para o terceiro slide de apoio e depois voltem à conclusão.

As tabelas de treino, validação e teste, os controles e os casos completos
estão no relatório. [Relatório do seminário](../output/pdf/relatorio_seminario.pdf).
