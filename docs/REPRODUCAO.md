# Ler, conferir e reproduzir

[Início](../README.md) · [Guia do projeto](GUIA_COMPLETO.md) ·
[Notebook único](../notebooks/seminario_completo.ipynb) · [Resultados](RESULTADOS.md)

Escolha o percurso que corresponde ao que você quer fazer. Ler resultados,
executar demonstrações, conferir previsões e repetir os treinos têm requisitos
diferentes. Os modelos grandes já foram executados; não é preciso carregá-los
para estudar o trabalho.

## 1. Apenas ler ou apresentar — sem instalar nada

Abra o [notebook único](../notebooks/seminario_completo.ipynb) no GitHub. Ele já
mostra código, saídas e figuras. Use o sumário do Jupyter/VS Code ou a busca do
navegador por número/título: o roteiro da capa é textual para evitar atalhos
internos incompatíveis com o visualizador do GitHub.

Para apresentar, baixe os [slides](../output/slides/seminario_mae5920.pptx) e
consulte as [notas de fala](NOTAS_APRESENTACAO.md). Relatórios, HTMLs e notebooks
estão no [pacote atual](https://github.com/costaleirbag/projeto-mae5920/releases/latest).
Baixe os HTMLs e abra no navegador; o GitHub mostra seu código-fonte.

## 2. Executar somente o notebook único — CPU, sem modelos grandes

Baixe `seminario_completo.ipynb`. Crie um ambiente com Python 3.11 ou posterior:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy scikit-learn matplotlib torch 'tokenizers>=0.22,<0.23' jupyterlab
python -m jupyterlab
```

Abra o notebook, selecione o kernel desse ambiente e execute as células de cima
para baixo. O arquivo deve existir na pasta de execução ou na subpasta
`notebooks/`: o código lê seu próprio anexo. Em um serviço que abre o notebook
sem copiá-lo para os arquivos do runtime, copie também o `.ipynb` para essa pasta.
Não é necessário clonar o repositório neste percurso.

O caminho padrão executa 45 células: exercícios pequenos em CPU e conferência dos
resultados salvos. Não há download de encoders ou do Qwen. O treino completo das
seis CNNs fica desligado; ativá-lo requer o corpus restaurado e uma saída nova.
Veja [o conteúdo de cada capítulo e o teste de portabilidade](NOTEBOOK_UNICO.md).

## 3. Instalar o repositório para verificações e desenvolvimento

```bash
git clone https://github.com/costaleirbag/projeto-mae5920.git
cd projeto-mae5920
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

`pyproject.toml` declara intervalos de compatibilidade. `requirements-lock.txt`
registra o ambiente histórico usado no Mac para baseline, CNN e relatórios;
os metadados das etapas NVIDIA registram suas versões. Para repetir o treino
sequencial, siga o ambiente específico do [guia de encoders](REPRODUCAO_SEQUENCIAS.md).
A disponibilidade de pacotes de PyTorch varia conforme a plataforma.

## 4. Conferir os arquivos e as evidências salvas — sem GPU

Execute antes de modificar ou reexecutar notebooks:

```bash
python tools/verify_publication.py
python tools/build_results_overview.py --check
python verify_seminar_notebook.py
python verify_sequences.py
python run_bpe.py verify --device cpu
python verify_didactic.py
python verify_qwen_improved.py
python -m pytest -q
```

| Verificação | Evidência que confere |
|---|---|
| Publicação | Inventário, hashes, caminhos locais da documentação e notebooks executados; usa apenas a biblioteca padrão |
| Resumo de métricas | As 23 linhas de acurácia/F1 macro/CE e sua ligação às fontes congeladas |
| Notebook único | Código incorporado, anexos, arquitetura real e execução das células |
| Sequências | Seleção pela validação, 87 ajustes e reconstrução das métricas salvas |
| BPE | Tokenizer aprendido só no treino, seis checkpoints pareados e métricas preservadas |
| MLP e casos sintéticos | Seleções, trajetórias, auditoria interna e condições sintéticas |
| Qwen atual | Direção dos estados de treino, fontes, prompts, tokens, controles e entregáveis |
| Testes | Matemática, padding, pooling, gradientes, hooks e invariantes do protocolo, com modelos pequenos |

Esses comandos não repetem o treino ou a geração grande. Os testes não substituem
a avaliação dos modelos reais. Os certificados e suas limitações estão em cada
pasta de resultados.

Para reexecutar o notebook único no ambiente do repositório:

```bash
python execute_notebook.py --notebook notebooks/seminario_completo.ipynb
```

A execução pode mudar tempos, metadados e hashes. Preserve a versão publicada e
trabalhe numa cópia; os manifestos descrevem a construção distribuída, não qualquer
execução futura. O notebook específico do Qwen também roda sem GPU com as evidências
incluídas, usando `--notebook notebooks/qwen3_8b_steering.ipynb`.

## 5. Restaurar o IMDb — somente quando a operação precisar do corpus

Em uma cópia de trabalho:

```bash
python run.py prepare
python execute_notebook.py --notebook notebooks/experimentos_imdb.ipynb
```

A preparação baixa a revisão pública fixada, aplica o protocolo de duplicações
e recria os papéis originais. O corpus não faz parte do Git. Uma nova preparação
pode registrar duração e serialização diferentes; verificadores históricos
estritos exigem os arquivos correspondentes aos hashes preservados. O
[manifesto original](../data/manifest.json) documenta a execução distribuída.

## 6. Restaurar pesos históricos ou auditar checkpoints completos

O complemento `artefatos_reproducao.zip` da [release histórica v1.0.0](https://github.com/costaleirbag/projeto-mae5920/releases/tag/v1.0.0)
contém checkpoints originais de TF-IDF/CNN/BERT e caches numéricos descritos em seu
manifesto interno. Ele não inclui os novos checkpoints CNN/encoders/BPE nem os
pesos-base do Qwen. O pacote atual de leitura também não inclui esses pesos.

Extraia o complemento na raiz de uma cópia compatível. Com os dados históricos
restaurados, os verificadores originais são:

```bash
python verify_results.py
python verify_qwen.py
```

Para reconstruir os novos checkpoints, siga os protocolos abaixo. As verificações
salvas documentam a reavaliação feita nos pesos reais, mas não substituem a
presença dos pesos quando você quer repetir a inferência.

## 7. Repetir treinos ou geração — nova execução separada

| Estudo | Configuração e instruções |
|---|---|
| Original: TF-IDF, CNN, BERT, análise e RFM | [Protocolo técnico](EXPERIMENTO_ORIGINAL.md) · `config.json` · `python run.py --help` |
| CNNs, BERT, DistilBERT e ModernBERT | [Reprodução sequencial](REPRODUCAO_SEQUENCIAS.md) · `sequence_config.json` |
| BPE | [Protocolo e reprodução BPE](PROTOCOLO_BPE.md) · `bpe_config.json` |
| MLPs e mecanismos sintéticos | [Guia dos mecanismos](CONEXOES_DISCIPLINA.md) · `didactic_config.json` |
| Qwen: classificação e primeira rodada | [Protocolo Qwen](../QWEN_EXPERIMENTO.md) · `qwen_config.json` |
| Qwen: demonstração atual de geração | [Protocolo Qwen](../QWEN_EXPERIMENTO.md) · `qwen_improve_config.json` · `python run_qwen_improve.py --help` |

Use pastas novas, sem substituir seleções congeladas ou resultados publicados.
O treino escolhe pela validação; não volte a selecionar configuração pelo teste
já examinado. Os guias de sequências/BPE explicam a separação física da entrada de
auditoria e o congelamento antes de sua liberação.

Baseline e CNN originais foram executados em Apple M1; BERT e etapas posteriores
usaram NVIDIA GB10. Tempos entre esses dispositivos não comparam eficiência em
hardware idêntico. A inferência Qwen implementada usa CUDA e baixa a revisão fixada
do modelo; o percurso de leitura não precisa disso.

## 8. Reconstruir os materiais de leitura

O resumo de resultados tem um gerador que usa somente a biblioteca padrão:

```bash
python tools/build_results_overview.py
python tools/build_results_overview.py --check
```

Para reconstruir o notebook único a partir das evidências do repositório, siga o
[guia do notebook](NOTEBOOK_UNICO.md). O builder substitui o arquivo por uma cópia
sem saídas; é necessário executá-la, conferir e revisar as figuras antes de publicar.
O teste de portabilidade copia somente esse arquivo para fora do repositório.

O relatório Qwen pode ser reconstruído com `python build_qwen_improved_report.py`,
seguido da execução do notebook específico. Qualquer nova construção de relatório
ou apresentação requer atualização dos manifestos e inspeção visual dos artefatos.

## 9. GitHub Actions

O [modelo de verificação automática](ci/verificar-projeto.yml) reúne verificações
em CPU. Permanece como template; não há workflow ativo nesta publicação.
A credencial utilizada não tinha permissão para criar workflows. Os testes e
verificadores da entrega foram executados localmente. Ativar o template futuramente
requer uma credencial com a permissão correspondente.
