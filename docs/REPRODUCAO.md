# Reprodução dos experimentos

[Voltar ao início](../README.md) · [Guia completo](GUIA_COMPLETO.md)

O repositório separa leitura dos resultados, verificação dos artefatos salvos
e novas execuções dos modelos. As previsões, os textos gerados e as direções
selecionadas são resultados de execuções já concluídas.

## 1. Leitura, sem instalar dependências

Abra os [dois relatórios](../README.md#comece-por-aqui), os notebooks já
executados ou baixe os HTMLs e abra no navegador. A galeria Qwen é
autocontida e apresenta as 64 fontes com intervenções negativa, zero e positiva.

Os mesmos materiais estão reunidos em `imdb_qwen3_entrega.zip`, nos
[downloads da release](https://github.com/costaleirbag/projeto-mae5920/releases/latest).

## 2. Instalação

Na raiz do repositório, com Python 3.11 ou posterior:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

`pyproject.toml` declara os intervalos de compatibilidade.
`requirements-lock.txt` fixa o ambiente usado no Mac para baseline, CNN e
relatórios. As versões usadas nas etapas NVIDIA estão registradas nos
respectivos arquivos de metadados. A disponibilidade de wheels pode exigir
uma versão equivalente de PyTorch em outra plataforma.

## 3. Verificações sem GPU e sem corpus completo

Faça as verificações dos arquivos publicados antes de modificar ou
reexecutar notebooks:

```bash
python tools/verify_publication.py
python verify_qwen_improved.py
python -m pytest -q
```

O primeiro comando usa apenas a biblioteca padrão: confere os arquivos do
manifesto de publicação, caminhos e links locais da documentação.
O segundo reconstrói a direção escolhida dos estados de treino, confere
seleção congelada, fontes, prompts, tokens, controles e os entregáveis.
Não carrega os pesos do Qwen. Os testes usam modelos minúsculos aleatórios
para verificar a matemática e o comportamento dos hooks; não repetem os treinos.

O notebook didático pode ser reexecutado com os arquivos incluídos:

```bash
python execute_notebook.py --notebook notebooks/qwen3_8b_steering.ipynb
```

Essa execução pode alterar metadados e representação numérica de saídas,
inclusive o hash do notebook. O manifesto e `report_build.json` descrevem
a versão publicada, não uma futura execução em outro ambiente.

## 4. Restaurar o IMDb

```bash
python run.py prepare
python execute_notebook.py --notebook notebooks/experimentos_imdb.ipynb
```

A preparação baixa a revisão fixada de `stanfordnlp/imdb`, remove
duplicações conforme o protocolo e recria treino, validação, calibração e
teste. O corpus completo não faz parte da distribuição Git.
`data/manifest.json` documenta as divisões originais.

Uma nova preparação pode registrar duração e serialização diferentes.
Para auditoria dos hashes históricos, compare as chaves, os textos e as
divisões com os registros originais; os verificadores estritos exigem os
artefatos históricos correspondentes.

## 5. Artefatos maiores e auditoria completa

O arquivo opcional `artefatos_reproducao.zip`, na release, complementa o
repositório com os checkpoints treinados de TF-IDF/CNN/BERT e os caches
numéricos históricos necessários aos verificadores completos. Seu manifesto
interno descreve exatamente os arquivos incluídos e seus hashes.

Extraia o conteúdo na raiz de uma cópia de trabalho. Os diretórios
`artifacts/` e `results/` serão complementados. Nenhuma credencial ou
configuração de acesso à infraestrutura é distribuída.

Após restaurar também os dados compatíveis:

```bash
python verify_results.py
python verify_qwen.py
```

Esses verificadores são estritos quanto aos hashes da execução histórica.
O pacote não inclui os pesos-base do Qwen; a leitura e a auditoria das
saídas salvas não precisam deles.

## 6. Novas execuções

Use uma cópia separada para não substituir a evidência publicada.
O pipeline principal pode ser executado por etapas:

```bash
python run.py prepare
python run.py baseline
python run.py cnn
python run.py bert
python run.py analyze
python run.py steer
python run.py rewrite
python run.py report
```

Baseline e CNN foram executados em Apple M1; BERT e as etapas Qwen usaram
NVIDIA GB10. Os tempos observados não comparam hardware idêntico.
O pipeline Qwen usa CUDA e baixa a revisão fixada de `Qwen/Qwen3-8B`.

```bash
python run_qwen.py --help
python run_qwen_improve.py --help
```

Siga o [protocolo Qwen](../QWEN_EXPERIMENTO.md) e os parâmetros dos arquivos
`config.json`, `qwen_config.json` e `qwen_improve_config.json`.
Não repita seleção de direção, camada ou intensidade usando os resultados
do teste já aberto. Para um novo estudo, defina nova validação e novo teste.

## 7. Reconstruir os materiais de leitura

```bash
python build_qwen_improved_report.py
python execute_notebook.py --notebook notebooks/qwen3_8b_steering.ipynb
```

O gerador usa os resultados salvos. PDFs, HTMLs e notebook reexecutado
passam a representar uma nova construção; atualize os manifestos e faça
nova inspeção visual antes de distribuir essa versão.

## 8. GitHub Actions

O [modelo de verificação automática](ci/verificar-projeto.yml) executa os
mesmos testes e a auditoria Qwen em CPU. Para ativá-lo, publique-o em
`.github/workflows/verify.yml` usando uma credencial com permissão para
criar workflows. A credencial disponível durante esta publicação permitia
enviar código e releases, mas não workflows; por isso o modelo permanece
em `docs/ci/`. As verificações locais da entrega foram executadas e passaram.
