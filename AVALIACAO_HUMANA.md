# Revisão cega das reescritas

O experimento produziu 544 saídas e uma amostra predefinida de 60 casos para
revisão posterior. Nenhuma anotação humana foi realizada nesta execução.

Abra `results/qwen/evaluation/qualitative_review_blind.csv`. A planilha mostra
a fonte e a saída, sem informar a condição. Mantenha o arquivo
`qualitative_review_key.csv` separado até terminar as notas. Se possível, dois
integrantes avaliam independentemente os mesmos casos; cada um salva sua
cópia. Não substitua o arquivo original vazio que acompanha o experimento.

Avalie o texto de saída e compare os fatos com a fonte. Uma opinião positiva
pode virar negativa sem alterar fatos do enredo, e isso não deve reduzir a
nota de preservação factual.

| Coluna | Como preencher |
|---|---|
| `sentiment_negative_neutral_positive` | `negative`, `neutral` ou `positive`, considerando a opinião global expressa sobre o filme. Neutralidade significa ausência de avaliação clara, não dúvida do classificador automático. |
| `fluency_1_to_5` | 1: incompreensível ou sem texto; 2: erros graves/repetição; 3: compreensível com problemas; 4: natural com pequenos problemas; 5: natural e coerente. |
| `facts_preserved_1_to_5` | 1: sem texto ou fatos centrais contraditos; 2: várias invenções/alterações; 3: alguma alteração factual relevante; 4: fatos centrais mantidos com pequena omissão/imprecisão; 5: nomes, eventos e cenário mantidos, sem invenção relevante. |
| `notes` | Indique a evidência concreta, como nome inventado, troca de evento, frase cortada ou repetição. |

Uma saída vazia recebe 1 em fluência/fatos e sentimento em branco; registre
`empty_output` nas notas. Uma frase interrompida pelo limite de tokens deve
ser avaliada como está, sem completar mentalmente o texto. O pedido foi de
duas frases: anote descumprimentos, mas avalie a fluência e os fatos
separadamente. Omissões inevitáveis de uma síntese de duas frases devem ser
distinguidas de contradições e invenções.

Depois das notas, use a chave para comparar condições sobre as mesmas fontes.
Apresente a concordância entre avaliadores e exemplos de divergência. A
amostra mistura avaliações oficiais e descrições neutras sintéticas; mostre
essa diferença. As taxas dos classificadores automáticos e o cosseno MiniLM
não devem ser renomeados como notas humanas.

As leituras qualitativas já incluídas no relatório foram feitas pelo
assistente e declaram essa origem. Elas não substituem esta revisão cega.
