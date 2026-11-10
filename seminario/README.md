# Relatório e apresentação do seminário

**O que a CNN lê em uma resenha?** O relatório acompanha a classificação do
IMDb, da tokenização às falhas de ordem. Os slides usam a mesma sequência.
A fala principal dura 12 minutos, com três minutos adicionais e quatro slides
de apoio. O relatório tem oito páginas, dentro da faixa do enunciado.

| Material | Para ler ou editar |
|---|---|
| Relatório | [PDF](../output/pdf/relatorio_seminario.pdf) · [LaTeX](relatorio.tex) · [seções](partes/) |
| Apresentação | [Abrir no navegador](https://costaleirbag.github.io/projeto-mae5920/) · [HTML para baixar](../output/slides/slides_15_min_interativos.html) · [PDF 16:9](../output/pdf/slides_15_min.pdf) · [Beamer](slides.tex) |
| Fala, tempos e cortes | [Roteiro](roteiro.md) |
| Experimentos e código explicado | [Notebook único](../notebooks/seminario_completo.ipynb) |
| Predições dos quatro pares e teste de cobertura | [Entradas](casos.json) · [Resultados por semente](dados/pares_resultados.json) |
| Avaliação fora do IMDb e exemplos do YouTube | [Protocolo](#avaliação-fora-do-imdb) · [Resultados por semente](dados/fora_dominio_resultados.json) |
| Discordâncias, largura e passo | [Diagnósticos e exemplos](diagnosticos.md) |
| Dois picos com e sem posição | [Protocolo e resultados](diagnosticos.md#o-que-acontece-quando-guardamos-a-posição-dos-picos) |

## Apresentação interativa

Abra a apresentação pelo GitHub Pages. Nos slides 4 e 5, use **Anterior**
e **Próxima etapa** ou as setas para passar uma etapa por vez. A barra pula
para qualquer etapa. No slide 9, as setas ou os botões A e B trocam os
adjetivos e atualizam a predição. Use `F` para tela cheia.
[Abrir a apresentação](https://costaleirbag.github.io/projeto-mae5920/).

No slide 4, percorra as sete fusões de *pallbearer*. No slide 5, a
**Montagem** revela a TextCNN dos IDs às probabilidades, com os motivos
das escolhas e links para as linhas do código. A última etapa abre a
**Varredura**, que acompanha um filtro nas frases A e B com larguras 3, 4
ou 5 e passos 1, 2 ou 3. Nela, o último quadro destaca a maior resposta
entre as janelas visitadas. BPE e Varredura oferecem reprodução automática.
[BPE](https://costaleirbag.github.io/projeto-mae5920/#slide-4) · [Montagem](https://costaleirbag.github.io/projeto-mae5920/#slide-5) · [Varredura](https://costaleirbag.github.io/projeto-mae5920/#slide-5-varredura).

O slide 9 retoma as frases do início. Mostra as duas orações, a leitura
esperada da conclusão e o escore positivo de cada tokenizador. Cada linha
resume três treinos, um por semente. A régua mantém o limiar de 50% e marca
onde estava A quando você mostra B. [Par de contraste](https://costaleirbag.github.io/projeto-mae5920/#slide-9).

No slide 10, troque A por B e acompanhe um pico real da CNN. A mesma
janela começa no token 8 em A e no 20 em B. O valor que o filtro entrega
continua igual. [Controle do máximo global](https://costaleirbag.github.io/projeto-mae5920/#slide-10).

Os textos A e B do slide 10 são construídos. A frase separadora aparece
três vezes em cada um, preservando as janelas quando os blocos trocam de
lugar. O slide 11 mostra os picos reais das duas metades desse mesmo
exemplo e separa esse controle do teste com pares de resenhas editadas.
[Dois picos por filtro](https://costaleirbag.github.io/projeto-mae5920/#slide-11).

O primeiro slide de apoio alterna entre o recorte de 400 tokens e o texto
inteiro de uma resenha do IMDb. **Conclusão** no topo vai ao encerramento.
Ao chegar lá, o avanço normal para e os slides de apoio ficam disponíveis
pelo link próprio. [Recorte](https://costaleirbag.github.io/projeto-mae5920/#recorte) · [Conclusão](https://costaleirbag.github.io/projeto-mae5920/#encerramento).

Para apresentar offline, use **Download raw file** no arquivo do GitHub e
abra a cópia no navegador. [HTML para baixar](../output/slides/slides_15_min_interativos.html).

A sequência BPE foi conferida com o tokenizer treinado. Para cada largura
da CNN por palavras, mostramos o filtro com maior amplitude na frase A e
mantemos o mesmo filtro ao mudar para B. Os valores antes e depois da ReLU
foram extraídos do checkpoint da semente 17. O PDF resume os mesmos passos
em quadros fixos. [Valores usados nas animações](dados/animacoes.json).

## Revisão em grupo

Editem uma seção de cada vez em `partes/`, ou um frame de cada vez em `slides.tex`.
No PR, indiquem qual afirmação mudou, qual evidência a sustenta e se a fala ainda
cabe no tempo. Atualizem `roteiro.md` quando mudarem a ordem ou a duração dos frames.

Para compilar com TeX Live e `latexmk`, rode `make -C seminario` na raiz do
repositório. O comando grava os dois PDFs em `output/pdf/` e coloca arquivos
intermediários em `tmp/seminario/`. Para usar Overleaf, importe a pasta `seminario`
e selecione `relatorio.tex` ou `slides.tex` como documento principal, com pdfLaTeX.
As figuras e tabelas do LaTeX estão nessa própria pasta.

Para recompilar apenas o relatório, rode `make -C seminario relatorio`.
Após atualizar os escores do contraste ou as métricas de posição, rode
`python seminario/preparar_relatorio.py` antes de compilar para regenerar
os dados das figuras estáticas e da tabela de posição.

Para reconstruir também o HTML, rode `make -C seminario interativo` com
`pypdf` e `pdftoppm` instalados. O HTML usa as páginas renderizadas do PDF
nos slides fixos. Os slides de BPE, CNN, contraste, máximo global e recorte usam os dados
salvos e os controles da apresentação. Os prints do YouTube também ficam
embutidos no HTML. [Fontes do HTML](apresentacao.html) · [estilo](apresentacao.css) · [controles](apresentacao.js).

O diagrama vetorial do slide 5 também aparece no PDF da apresentação. Após editá-lo,
rode `make -C seminario figura-arquitetura` com `rsvg-convert` antes de
recompilar os slides. Os trechos do código são lidos das fontes ao gerar
o HTML. [Diagrama](arquitetura_slide.svg) · [Gerador](preparar_apresentacao.py).

Inclua o HTML regenerado ao publicar a edição na `main`. O GitHub Pages
publica automaticamente a nova versão do arquivo. [Configuração](../.github/workflows/slides-pages.yml).

As tabelas e os gráficos vêm das métricas salvas. Para atualizá-los, rode
`python seminario/preparar.py` antes de compilar. O arquivo
`dados/fontes.json` registra os hashes de origem. Os textos de fala e o
diagrama podem ser editados diretamente.

## Demonstração opcional em CPU

A demonstração usa a CNN por palavras e a CNN com BPE, ambas da semente 17.
Ela lê os checkpoints congelados do estudo pareado. Coloque os pesos nos
caminhos abaixo ou passe a pasta que os contém por `--artifacts`.
O programa confere os hashes antes de carregar os pesos.

| Representação | Caminho a partir da pasta de artefatos |
|---|---|
| Palavras | `sequences/cnn_optimizer__adam__lr0.001__s17.pt` |
| BPE | `bpe/cnn_bpe20k__s17.pt` |

Com o ambiente Python do projeto instalado:

~~~bash
python seminario/demo.py --artifacts /caminho/para/artifacts
~~~

A página recebe uma resenha curta em inglês e exibe classe, escore positivo
e tokens. [Demonstração no computador da apresentação](http://127.0.0.1:8765).
O endpoint local `POST /predict` recebe JSON no formato
`{"text":"The movie was wonderful."}`. Para usar o terminal, acrescente
`--text 'The movie was wonderful.'`.

No dia, carreguem a página antes da fala e façam um teste rápido. O slide 12
traz os prints e as classes previstas para *Friday* e *The Room*. Esses
casos podem ocupar o tempo da demonstração. As animações dos slides 4 e 5 já funcionam no HTML baixado.

Para reproduzir os quatro pares com as três sementes, a pasta de artefatos
precisa ter também as versões `s29` e `s43` das duas CNNs. Rode:

~~~bash
python seminario/medir_pares.py --artifacts /caminho/para/artifacts
python seminario/medir_animacoes.py --artifacts /caminho/para/artifacts
python seminario/preparar.py
make -C seminario interativo
~~~

## Avaliação fora do IMDb

Aplicamos os seis checkpoints congelados do estudo de tokenização a 1.000
frases de Amazon e 1.000 de Yelp. Usamos somente esses dois arquivos da
coleção *Sentiment Labelled Sentences*, com 500 exemplos positivos e 500
negativos por fonte. A coleção privilegia frases de polaridade clara. As
medianas são 9 palavras em Amazon e 10 em Yelp. [Coleção UCI, CC BY 4.0](https://doi.org/10.24432/C57604) · [Kotzias et al., 2015](https://doi.org/10.1145/2783258.2783380).

Conferimos a igualdade exata, após limpeza e normalização, com os textos de
treino e validação das CNNs. Foram zero coincidências. Mantivemos os
tokenizadores, pesos e limiar originais e usamos escores sem calibração.
Acurácia, F1 macro e entropia cruzada são calculadas por execução e depois
agregadas pela média das três sementes de cada representação.

Os cinco comentários são casos qualitativos sem rótulos de referência.
Em *Baby*, os autores comentam a repercussão do artista e a reação do
público. Em *Friday*, “AI could never do this” admite admiração e ironia.
Os dois de *The Room* criticam o filme por comparação com o trailer ou
com um elogio citado. O comentário com “No one” recebeu classe positiva
nos seis checkpoints. [Tokens e escores por execução](dados/fora_dominio_resultados.json).

O Qwen3-8B recebeu o mesmo texto de cada print, com o prompt binário de
resenhas do projeto. No protocolo padrão, deu positiva para o comentário
de *The Room* e negativa para o de *Friday*. Inverter a ordem dos rótulos
no prompt manteve as classes. Em *Friday*, o escore positivo entre os dois
rótulos foi de 1,10% para 22,27%. O escore bruto é restrito aos rótulos
`negative` e `positive`. Os comentários seguem sem rótulo de referência.
[Qwen, logits e dois controles de sanidade](dados/qwen_youtube_resultados.json).


Também geramos três respostas por comentário com raciocínio desativado
e três com raciocínio ativado. Usamos a mesma instrução e a mesma
amostragem nos dois modos, com temperatura 0,6, top-p 0,95, top-k 20,
semente 20261020 e limite de 1.536 tokens novos. Os seis rótulos de
*The Room* foram positivos e os seis de *Friday* foram negativos.
As respostas com raciocínio chegaram ao rótulo final depois do bloco
de raciocínio. O arquivo guarda as classes e as contagens de tokens.
[Configuração recomendada pelo Qwen](https://huggingface.co/Qwen/Qwen3-8B#switching-between-thinking-and-non-thinking-mode).

Para reproduzir com CUDA e o ambiente Qwen do projeto, rode
`python seminario/medir_qwen_youtube.py`.
[Configuração da inferência](../QWEN_EXPERIMENTO.md).

Para repetir a inferência com os seis checkpoints, baixe o ZIP original e
passe seu caminho ao script. O arquivo de predições guarda os números de
linha do arquivo original, os rótulos `0`/`1` e os dois logits de cada
execução. As chaves, por exemplo `amazon_rotulos` e `amazon_bpe_s17`,
identificam a fonte, a representação e a semente.
[Download da UCI](https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip) · [Predições em NPZ](dados/fora_dominio_predicoes.npz).

~~~bash
python seminario/medir_fora_dominio.py --dataset /caminho/para/uci-331.zip --artifacts /caminho/para/artifacts
python seminario/preparar.py
make -C seminario
~~~

## Mapa das evidências

| Estudo | Material para conferir |
|---|---|
| TF-IDF, CNN e BERT originais | [Resultados por divisão](../results/comparison.csv) · [papéis e hashes](../data/manifest.json) · [dicionário](../DATA_DICTIONARY.md) |
| Calibração e análise de erros | [Comprimento e marcadores](../results/analysis/subgroup_metrics.csv) · [erros individuais](../results/analysis/error_examples.json) · [métricas CNN](../results/cnn/metrics.json) |
| CNN, SGD/momentum/Adam, janelas e dropout | [Grade](../sequence_config.json) · [históricos por execução](../results/sequences/runs/) · [sumário](../results/sequences/summary.json) |
| BERT, DistilBERT e ModernBERT | [Protocolo de treino](../docs/PROTOCOLO_SEQUENCIAS.md) · [relatório de arquiteturas e custo](../output/pdf/relatorio_cnn_encoders.pdf) |
| BPE, cobertura e sementes pareadas | [Protocolo](../docs/PROTOCOLO_BPE.md) · [tokenizer](../results/bpe/tokenizer.json) · [métricas por semente](../results/bpe/audit_metrics.json) |
| MLPs, inicialização e double descent | [Protocolo](../docs/CONEXOES_DISCIPLINA.md) · [resultados](../results/didactic/) · [suplemento](../output/pdf/relatorio_disciplina.pdf) |
| Classificação e steering do Qwen | [Protocolo](../QWEN_EXPERIMENTO.md) · [relatório](../relatorio_qwen3.pdf) · [diagnósticos automáticos](../results/qwen_improved/) |

As métricas de classificação têm uma tabela única com o estudo identificado.
[Resultados consolidados](../docs/RESULTADOS.md).
