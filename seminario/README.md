# Relatório e apresentação do seminário

**O que a CNN lê em uma resenha?** O relatório acompanha a classificação do
IMDb, da tokenização às falhas de ordem. Os slides usam a mesma sequência.
A fala principal dura 12 minutos, com três minutos adicionais e quatro slides
de apoio. O relatório tem oito páginas, dentro da faixa do enunciado.

| Material | Para ler ou editar |
|---|---|
| Relatório | [PDF](../output/pdf/relatorio_seminario.pdf) · [LaTeX](relatorio.tex) · [seções](partes/) |
| Apresentação | [PDF 16:9](../output/pdf/slides_15_min.pdf) · [Beamer](slides.tex) |
| Fala, tempos e cortes | [Roteiro](roteiro.md) |
| Experimentos e código explicado | [Notebook único](../notebooks/seminario_completo.ipynb) |
| Predições dos quatro pares e teste de cobertura | [Entradas](casos.json) · [Resultados por semente](dados/pares_resultados.json) |

## Revisão em grupo

Editem uma seção de cada vez em `partes/`, ou um frame de cada vez em `slides.tex`.
No PR, indiquem qual afirmação mudou, qual evidência a sustenta e se a fala ainda
cabe no tempo. Atualizem `roteiro.md` quando mudarem a ordem ou a duração dos frames.

Para compilar com TeX Live e `latexmk`, rode `make -C seminario` na raiz do
repositório. O comando grava os dois PDFs em `output/pdf/` e coloca arquivos
intermediários em `tmp/seminario/`. Para usar Overleaf, importe a pasta `seminario`
e selecione `relatorio.tex` ou `slides.tex` como documento principal, com pdfLaTeX.
As figuras e tabelas do LaTeX estão nessa própria pasta.

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
traz resultados salvos de dois textos para usar se o teste ao vivo sair do
roteiro ou se os pesos estiverem em outro computador.

Para reproduzir os quatro pares com as três sementes, a pasta de artefatos
precisa ter também as versões `s29` e `s43` das duas CNNs. Rode:

~~~bash
python seminario/medir_pares.py --artifacts /caminho/para/artifacts
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
