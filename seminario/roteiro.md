# Roteiro de fala

A versão principal termina aos **12:00**, dentro da faixa do enunciado original.
Os slides 12 e 13 ocupam três minutos adicionais se a turma tiver 15 minutos.
Os slides 14 a 17 ficam disponíveis para perguntas. Os tempos incluem pausas
curtas e passagens de fala. Os slides 4 e 8 são pontos naturais para trocar de pessoa.

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
| 10 | 10:10–11:30 | O limite do máximo global |
| 11 | 11:30–12:00 | Fechamento |
| 12 | 12:00–13:30 | Comentários do YouTube ou demonstração |
| 13 | 13:30–15:00 | Encoders e geração com o Qwen |

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
A calibração usa outra amostra. Para estudar as escolhas de arquitetura e
tokenização, trabalhamos com subconjuntos do treino limpo. Cada comparação
tem uma auditoria reservada de 2.048 resenhas. Os números para CNN mudam de uma
tabela para outra porque o conjunto e o comprimento de entrada mudam.”

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
da sequência. Um filtro com a mesma largura passa a ver outro contexto.”

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

## 7 · Teste oficial do IMDb

“Esta tabela pertence ao estudo original, com 25 mil resenhas de teste.
O baseline com bigramas chegou a 90,14 por cento. A CNN chegou a 86,78
e o BERT a 90,94. Acurácia conta os acertos, F1 macro dá o mesmo peso às
duas classes e a entropia cruzada mede a confiança no rótulo observado.
A tabela usa probabilidades calibradas em uma divisão separada. CNN e
TF-IDF discordaram em 2.590 resenhas. Entre as que a CNN precisou recortar,
foi mais comum só o TF-IDF acertar. No slide de apoio há um caso em que
os dois modelos passam de positivo para negativo quando leem o texto inteiro.”

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

## 9 · Predições no par de contraste

“Voltem às duas frases do início. A acaba em *wonderful* e B acaba em *terrible*.
Perguntamos duas coisas. A rede muda de classe quando trocamos a ordem? E
muda na direção que o sentido da frase indica? Na CNN por palavras, o escore
positivo médio vai de um por cento em A para quase 90 em B. O BPE vai de
cerca de um para 84 por cento. As seis CNNs mudaram de classe, todas na
direção oposta à leitura que propusemos.”

## 10 · Limite do máximo global

“Agora fazemos uma troca com um controle mais forte. Colocamos a mesma frase
separadora no início, entre os dois blocos e no fim. O separador tem tokens
suficientes para que uma janela de até cinco tokens só encontre um bloco
por vez. Quando trocamos os blocos, cada janela que existia continua
existindo. Confirmamos isso com os IDs das duas tokenizações. Se um filtro
recebe as mesmas janelas, vê o mesmo conjunto de respostas. Como o pooling
fica só com o máximo, o valor que ele manda para a saída é igual. Foi o
que encontramos numericamente nos seis checkpoints. Aqui a igualdade é
uma consequência da arquitetura.”

## 11 · Fechamento aos 12 minutos

“A CNN resume cada resenha pelos padrões locais que acionam seus filtros.
Essa escolha explica tanto a reação ao primeiro par quanto a igualdade no
segundo. Na extensão com o Qwen, passamos a observar a própria resenha que
o modelo escreve quando alteramos uma ativação interna.”

## 12 · Comentários fora do IMDb

“Em *Friday*, a frase sobre IA pode soar como admiração ou ironia. As CNNs
devolveram negativo. No primeiro comentário de *The Room*, a pessoa critica
o filme comparando com o trailer. Na semente 17, a CNN por palavras dá
positivo e a BPE dá negativo. No segundo, o elogio está entre aspas e é
seguido de *No one*. Mesmo assim, todas as seis redes deram positivo.
Esses comentários são casos de leitura. Para medir acurácia fora do IMDb,
usamos 2.000 frases rotuladas de Amazon e Yelp. A média fica entre 68 e 70
por cento, conforme a fonte e o tokenizer.”

Se optarem pela demonstração, abram o navegador antes da fala e usem até
30 segundos para obter uma resenha curta em inglês da plateia. Mostrem a
classe, o escore e os tokens das duas CNNs. Se a resposta demorar, voltem
aos resultados salvos no slide.

## 13 · Encoders e geração com o Qwen

“Testamos também BERT, DistilBERT e ModernBERT. Aqui o vetor de um token já
incorpora seu contexto. Fazemos uma média mascarada desses vetores e
colocamos uma camada linear de classificação. A tabela compara ajustar só
essa cabeça com propagar gradientes por todo o encoder. Na auditoria
sequencial, o ModernBERT passa de 81,30 para 91,24 por cento. O orçamento e
o otimizador também mudam.

Com o Qwen, usamos os escores dos tokens de rótulo para classificar. Também
usamos o modelo para escrever uma opinião token a token a partir da mesma
fonte. Mantivemos o prompt e deslocamos uma ativação durante a geração.
O slide de apoio mostra as três opiniões que saíram do mesmo enredo.”

## Cortes para o ensaio

- Aos 12 minutos, encerrem no slide 11. Os exemplos salvos dos slides 9 e 10 já sustentam a explicação.
- Se precisarem de cerca de 10 minutos, passem do slide 7 ao 9 depois de citar o resultado BPE em uma frase. Resumam a escolha de taxa e época do slide 6 em 35 segundos.
- Usem o slide 14 para os dois erros pareados, o 15 para calibração e o 17 para largura e passo. Esses quadros ficam reservados para perguntas.
- Para mostrar as gerações do Qwen durante a fala de 15 minutos, reservem os últimos 30 segundos do slide 13 e avancem direto ao slide 16.

As tabelas de treino, validação e teste, os controles e os casos completos
estão no relatório. [Relatório do seminário](../output/pdf/relatorio_seminario.pdf).
