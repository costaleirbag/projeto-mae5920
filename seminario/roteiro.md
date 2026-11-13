# Roteiro de fala

A apresentação termina no slide 14, **Conclusão**. Na versão de 12 minutos,
vá do slide 11 direto para **Conclusão** pelo link no topo. Na de 15, passe
também pelos slides 12 e 13. Os slides de apoio ficam num grupo separado
e abrem pelo link na conclusão. Os slides 4 e 8 são pontos naturais para
trocar de pessoa.

| Slide | Relógio | Assunto |
|---:|---|---|
| 1 | 00:00–00:25 | Pergunta do seminário |
| 2 | 00:25–01:35 | Duas frases e a hipótese de ordem |
| 3 | 01:35–02:35 | Amostras e escolhas de treino |
| 4 | 02:35–03:45 | Palavras e BPE |
| 5 | 03:45–05:15 | Arquitetura da TextCNN |
| 6 | 05:15–06:40 | Taxa, época e generalização |
| 7 | 06:40–07:50 | Teste oficial |
| 8 | 07:50–09:00 | Auditoria de palavras e BPE |
| 9 | 09:00–10:10 | A predição do par inicial |
| 10 | 10:10–11:10 | O limite do máximo global |
| 11 | 11:10–11:40 | Dois picos por filtro e acerto em pares |
| 12 | 11:40–13:10 | Comentários do YouTube ou demonstração |
| 13 | 13:10–14:35 | Encoders e geração com o Qwen |
| 14 | 14:35–15:00 ou 11:40–12:00 | Conclusão e perguntas |

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

“Usamos o IMDb, que traz resenhas em inglês com dois rótulos. No estudo original,
separamos treino, validação e calibração antes da avaliação nas 25 mil resenhas
do teste oficial. Removemos do treino textos repetidos, incluindo os que
coincidiam com o teste por hash. A validação serve para escolher a configuração.
A calibração usa outra amostra. Nos ensaios internos, todos recebem o
mesmo prefixo de 128 palavras e cada comparação tem uma auditoria reservada
de 2.048 resenhas. O teto adicional de 256 tokens não cortou nenhum desses
prefixos. A CNN original usa 400 tokens, com os pesos ajustados em uma
amostra de treino maior.”

## 4 · Fusões do BPE

No HTML, use **Próxima etapa** ou a seta direita para mostrar a primeira
fusão, `r + e → re`. Avance no ritmo da fala ou arraste até a última etapa
ao falar das três peças. No PDF, aponte as três linhas da sequência.

“A CNN recebe índices. Na versão por palavras, a expressão regular preserva
palavras, contrações e números. Contamos a frequência só no treino e reservamos
um índice para padding e outro para desconhecidos. Nesta frase, *pallbearer*
fica com o índice de desconhecido. No BPE, aprendemos uniões de caracteres
frequentes nos mesmos textos, sem consultar os rótulos. O tokenizer congelado
reconhece *p*, *all* e *bearer* com marca de fim de palavra. Isso dá à rede
uma representação específica para a palavra rara e aumenta o comprimento
da sequência. Um filtro com a mesma largura passa a ver outro contexto.
Os dois vocabulários têm 20 mil entradas por escolha experimental. Assim
as duas tabelas de embeddings têm 20 mil linhas de 128 valores.”

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

## 7 · Teste oficial do IMDb

“O baseline conta palavras e pares consecutivos, como *not good*, na
resenha inteira. O TF-IDF pesa essas ocorrências pela frequência na
resenha e pela raridade no treino. A regressão logística aprende a somar
contribuições positivas e negativas. O bigrama guarda a ordem dentro
do par. O vetor perde a posição ao somar pela resenha toda.

No teste oficial, com 25 mil resenhas, o baseline chegou a 90,14%, a
CNN a 86,78% e o BERT a 90,94%. Acurácia e F1 usam a classe. A
entropia cruzada usa a probabilidade, calibrada numa amostra separada.
O ganho de 0,50 ponto ao dar o texto todo à mesma CNN explica parte da
diferença para o baseline.”

O link **Entenda o TF-IDF** abre um exemplo em que a troca de ordem muda
o bigrama. Mostrem a troca em até 20 segundos e usem **Voltar ao teste
oficial**. Se perguntarem sobre os 99,72% no treino, expliquem que o
modelo tem 80.000 atributos para 19.823 textos e regularização L2 com
\(C=8\) escolhido na validação. O teste mede a generalização.

## 8 · Auditoria de palavras e BPE

“Aqui os dois tokenizadores enfrentam a mesma auditoria interna. Usamos
a mesma arquitetura, orçamento e taxa de Adam, com três sementes pareadas.
A figura liga a versão por palavras à BPE dentro de cada semente. BPE
reduziu de 3,27 por cento para zero a fração de tokens desconhecidos.
Na mesma comparação, a acurácia média foi de 82 por cento para 81,41.
Uma janela de cinco subpalavras pode cobrir menos palavras do que a de
cinco palavras completas. E a taxa foi escolhida originalmente na CNN por
palavras. Os números mostram o que ocorreu sob esse controle. Os escores
dos pares ajudam a ver o que mudou em frases específicas.”

## 9 · Dois adjetivos trocam de lugar

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

## 10 · O máximo global descarta a posição

Mostre A e depois pressione **B** ou a seta direita. No PDF, compare as duas
linhas e aponte a posição e o valor do pico.

“Montamos dois textos de controle. A mesma frase vem antes, entre e depois
das opiniões. Assim, cada opinião tem os mesmos vizinhos quando troca de
lugar. A janela *movie i loved* começa no token 8 em A e no 20 em B. O
filtro dá 0,47 nos dois casos. Cada um dos 300 filtros guarda seu próprio
máximo. Como as janelas são as mesmas, os 300 valores e a saída coincidem.”
[Código do controle](medir_pares.py).

## 11 · Guardar a posição ajudou a classificar?

“Pegamos um pico por metade. Ordenados pelo valor, A e B dão 0,25 e
0,47. Na ordem do texto, o primeiro número vem do início e o segundo vem
do fim. Isso mudou os escores do controle, e a classe coincidiu. Cada
regressão recebe 600 valores. Nos 344 pares editados por pessoas, o
acerto ficou perto de 40% nas duas opções. O par conta quando acertamos
as duas versões.” [Código do teste de posição](medir_posicao.py).

Se perguntarem sobre o teste sintético, há quatro moldes com veredito no
começo ou no fim. Nenhuma das cabeças acertou as duas frases de qualquer
par com veredito no fim. [Método e resultados por semente](diagnosticos.md#o-que-acontece-quando-guardamos-a-posição-dos-picos).

## 12 · Comentários fora do IMDb

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

## 13 · Encoders e geração com o Qwen

“Aqui são 2.048 resenhas internas, cortadas em 128 palavras. Os vetores
já trazem contexto dos outros tokens. Fazemos uma média mascarada e uma
camada linear classifica. À esquerda, só essa camada aprende, com lote
128. À direita, ajustamos o encoder também, com lote 32. O BERT original
do slide 7 também usou lote 32, com mais dados e outro conjunto de teste.

O quadro do Qwen remete ao teste oficial. Com pesos congelados e até 768
tokens da resenha, usamos o prompt e a diferença dos logits dos dois
rótulos. Ele atingiu 95,26%. Pré-treino, arquitetura e entrada mudam
juntos. A geração com intervenção nas ativações fica no apoio.”

## 14 · Conclusão

“A nossa CNN ficou 3,36 pontos abaixo do TF-IDF. Cortar as resenhas explica
meio ponto, com os pesos fixos. O máximo global perde a posição, como
mostra a troca de blocos. No par com *but*, os filtros responderam à
troca, mas a CNN seguiu a primeira opinião. BERT e Qwen mostram que outras
receitas de deep learning superaram o baseline no mesmo teste.”
Deixe o quadro de perguntas na tela.

## Cortes para o ensaio

- Na versão de 12 minutos, vá do slide 11 à conclusão pelo link no topo.
- Se precisarem de cerca de 10 minutos, passem do slide 7 ao 9 depois de citar o resultado BPE em uma frase. Resumam a escolha de taxa e época do slide 6 em 35 segundos.
- Na seção de apoio, usem o primeiro slide para mostrar o recorte da resenha, o segundo para calibração e o quarto para largura e passo.
- Se quiserem mostrar a geração do Qwen nos 15 minutos, reservem os últimos 30 segundos do slide 13 para o terceiro slide de apoio e depois voltem à conclusão.

As tabelas de treino, validação e teste, os controles e os casos completos
estão no relatório. [Relatório do seminário](../output/pdf/relatorio_seminario.pdf).
