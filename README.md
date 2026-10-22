<p align="center">
  <img src="docs/assets/capa.svg" alt="IMDb e Qwen3-8B: classificação de sentimento e intervenção nas ativações" width="100%">
</p>

<p align="center">
  <strong>Projeto MAE5920 · Introdução a Deep Learning</strong><br>
  Nestor · Letícia · Gabriel
</p>

<p align="center">
  <a href="relatorio_imdb.pdf">Relatório principal</a> ·
  <a href="relatorio_qwen3.pdf">Relatório Qwen3</a> ·
  <a href="notebooks/">Notebooks</a> ·
  <a href="https://github.com/costaleirbag/projeto-mae5920/releases/latest">Downloads</a>
</p>

Este trabalho tem duas partes: comparar modelos de classificação de sentimento
em resenhas de filmes e mostrar, de forma didática, o que acontece quando
intervimos nas ativações internas de um modelo de linguagem congelado.

Na extensão com **Qwen3-8B**, a mesma entrada e o mesmo prompt produzem três
saídas: sem steering, com intervenção negativa e com intervenção positiva.
O objetivo é compreender o mecanismo, seus efeitos e suas falhas.

## Comece por aqui

| Material | O que você encontra |
|---|---|
| **[01 · Relatório principal — IMDb](relatorio_imdb.pdf)** | TF-IDF + regressão logística, CNN, BERT, análise de erros e primeira intervenção nas ativações. |
| **[02 · Relatório — Qwen3-8B e steering](relatorio_qwen3.pdf)** | Modelo congelado, construção das direções e exemplos completos da intervenção nos dois sentidos. |
| [Notebook dos modelos](notebooks/experimentos_imdb.ipynb) | Resultados, figuras e verificações do experimento principal; já executado. |
| [Notebook Qwen3 e steering](notebooks/qwen3_8b_steering.ipynb) | Reconstrução das direções, exemplos e diagnósticos; já executado e pode ser reexecutado sem GPU. |
| [Galeria Qwen em HTML](relatorio_qwen3.html) | Todas as 64 fontes, cada uma com as três saídas da demonstração. Baixe o arquivo e abra no navegador. |
| [Relatório principal em HTML](relatorio.html) | Versão ampliada, com figuras e exemplos. Baixe e abra no navegador. |

Para compartilhar com o grupo, os **dois PDFs** são a leitura principal.
Os notebooks e os arquivos de resultados permitem conferir como chegamos até eles.
O GitHub exibe os notebooks executados; o seletor interativo do Qwen funciona
em um ambiente Jupyter. A galeria HTML também permite percorrer os casos.

## Duas perguntas, dois experimentos

### 1. Como os modelos classificam resenhas?

Resultados no teste oficial do IMDb, com **25.000 avaliações**:

| Modelo | Ajuste dos pesos neste projeto | Acurácia | F1 macro |
|---|---|---:|---:|
| TF-IDF + regressão logística | Sim, no treino | 90,144% | 90,144% |
| CNN | Sim, no treino | 86,784% | 86,782% |
| BERT-base | Ajuste integral | 90,944% | 90,941% |
| Qwen3-8B | Pesos congelados | 95,260% | 95,260% |

O Qwen classifica comparando os escores dos tokens `negative` e `positive`.
Não acrescentamos uma cabeça de classificação nem ajustamos seus pesos.
A calibração de probabilidades é uma etapa separada.

**A comparação tem limites:** o Qwen recebe até 768 tokens da fonte e o BERT,
192 WordPieces; o baseline recebe o texto completo. A extensão com Qwen é
exploratória e reutiliza um teste já examinado. Esses números não isolam
o efeito da arquitetura ou do tamanho do modelo.

Fontes dos números: [modelos principais](results/comparison.csv) e
[Qwen](results/qwen/classification/metrics.json).

### 2. O que muda quando deslocamos uma ativação?

Na demonstração atual, capturamos estados internos de **128 enredos de treino**,
cada um apresentado com uma instrução positiva e outra negativa. No bloco 28,
a diferença entre as médias dos dois grupos define uma direção unitária:

```text
v = normalizar(média dos estados positivos − média dos estados negativos)
h' = h + α · escala · v             α ∈ {−2, 0, +2}
```

Na geração de teste, o prompt comum pede os campos `FACT` e `REVIEW`, sem
indicar um sentimento-alvo. A intervenção começa após o marcador `REVIEW:`
gerado pelo modelo. Os pesos continuam congelados. O texto factual anterior
ao marcador permanece idêntico entre as três condições.

A direção original, obtida por **RFM/AGOP** a partir de 512 resenhas rotuladas,
também está preservada e é reconstruída no notebook. São procedimentos
distintos; a direção escolhida para a demonstração atual é a diferença de médias.

Após congelar a configuração na validação, avaliamos 64 novas fontes de
geração: 32 resenhas reais e 32 enredos sintéticos, em 11 condições
(*704 saídas*, incluindo controles aleatórios). Outras 160 saídas reavaliam
as fontes da rodada anterior. Todas as saídas consolidadas são preservadas.

O juiz automático reconheceu os dois sentidos-alvo em **55/64 fontes**.
Esse resultado é um diagnóstico do próprio Qwen sem intervenção, sem uma
referência humana independente. Há falhas de estilo e uma invenção clara de
cenário preservada no relatório. Prefixo factual idêntico não garante que
toda a saída esteja correta. Demonstrar superioridade sobre prompting
**não é o objetivo do trabalho**.

Detalhes: [protocolo Qwen](QWEN_EXPERIMENTO.md),
[seleção congelada](results/qwen_improved/selection_frozen.json) e
[verificação da rodada atual](results/qwen_improved/verification.json).

## Explorar e reproduzir

**Para ler**, abra os PDFs ou os notebooks no GitHub. O
[pacote da entrega](https://github.com/costaleirbag/projeto-mae5920/releases/latest)
reúne o material para download. Não é preciso carregar o Qwen.

**Para reexecutar o notebook didático do Qwen**, use Python 3.11 ou posterior:

```bash
git clone https://github.com/costaleirbag/projeto-mae5920.git
cd projeto-mae5920
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python execute_notebook.py --notebook notebooks/qwen3_8b_steering.ipynb
```

O notebook usa os estados selecionados e as saídas salvas, sem GPU e sem
baixar os pesos do Qwen. O arquivo `requirements-lock.txt` registra o ambiente
original; em outra plataforma, a versão disponível de PyTorch pode variar.

**Para conferir os artefatos publicados e o protocolo atual**, antes de
reexecutar notebooks ou regenerar relatórios:

```bash
python tools/verify_publication.py
python verify_qwen_improved.py
python -m pytest -q
```

Reexecutar um notebook pode alterar seus metadados e o hash do arquivo;
o manifesto descreve os arquivos publicados. Novas execuções devem ficar
em uma cópia de trabalho.

Para restaurar os dados do IMDb e reexecutar o notebook principal:

```bash
python run.py prepare
python execute_notebook.py --notebook notebooks/experimentos_imdb.ipynb
```

Treinamento, geração com os 8 bilhões de parâmetros, artefatos maiores e
verificações completas estão descritos no
**[guia de reprodução](docs/REPRODUCAO.md)**. A inferência Qwen implementada
requer GPU CUDA; a leitura dos resultados não requer.

## Mapa do repositório

```text
.
├── relatorio_imdb.pdf / relatorio.html       # Experimento principal
├── relatorio_qwen3.pdf / relatorio_qwen3.html # Extensão didática atual
├── notebooks/                              # Dois notebooks executados
├── src/imdb_lab/                            # Implementação dos modelos e análises
├── tests/                                  # Invariantes científicos e hooks
├── results/                                # Previsões, figuras e saídas consolidadas
│   ├── qwen/                               # Primeira rodada Qwen
│   └── qwen_improved/                       # Rodada atual e histórico identificado
├── data/manifest.json                      # Origem e auditoria das divisões
├── docs/                                   # Guia completo, reprodução e publicação
└── tools/verify_publication.py              # Integridade da distribuição
```

Os arquivos na raiz são as **versões atuais**. O material didático anterior
permanece em [`previous_didactic/`](results/qwen_improved/previous_didactic/).
O corpus preparado, os checkpoints treinados e caches maiores são tratados
separadamente; veja o guia de reprodução e os downloads da release.

## Documentação e referências

- [Guia completo do experimento](docs/GUIA_COMPLETO.md)
- [Reprodução e ambientes](docs/REPRODUCAO.md)
- [Protocolo Qwen3-8B](QWEN_EXPERIMENTO.md)
- [Dicionário dos dados e resultados](DATA_DICTIONARY.md)
- [Rubrica para revisão humana](AVALIACAO_HUMANA.md)
- [Escopo e integridade da publicação](docs/PUBLICACAO.md)

Dados: [Large Movie Review Dataset — Maas et al.](https://ai.stanford.edu/~amaas/data/sentiment/),
obtidos pela distribuição [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb).
Modelos: [BERT-base](https://huggingface.co/google-bert/bert-base-uncased),
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) e
[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
Steering: [Beaglehole et al., *Toward universal steering and monitoring of AI models*](https://arxiv.org/abs/2502.03708)
e [implementação dos autores](https://github.com/dmbeaglehole/neural_controllers).

Esta é uma implementação educacional independente. Dados e modelos de
terceiros mantêm seus termos de uso e atribuição. Leituras qualitativas
produzidas com assistência de IA estão identificadas nos resultados;
não houve anotação humana independente.
