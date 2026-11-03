# Escopo da publicação

Esta branch reúne o experimento original de sentimento no IMDb, o steering
no Qwen3-8B e duas extensões ligadas à disciplina. Os relatórios originais
permanecem preservados. A release v1.0.0 representa a entrega original;
os novos materiais estão na branch `feat/experimentos-didaticos`.

[Voltar ao início](../README.md)

## Materiais da branch

- Quatro relatórios em PDF e suas versões HTML: IMDb, Qwen3, mecanismos com
  MLPs e demonstrações sintéticas, CNN e encoders modernos.
- Quatro notebooks executados, com leitura dos resultados sem carregar os
  grandes modelos nem usar GPU.
- Uma apresentação editável com diagramas, gráficos nativos e notas de fala.
- Código, configurações, revisões de modelos, predições e trajetórias.
- Evidências de seleção por validação, controles, falhas e verificações.

O [protocolo CNN e encoders](PROTOCOLO_SEQUENCIAS.md) identifica os 87 novos
treinos neurais e a nova auditoria interna. O [guia MLP](CONEXOES_DISCIPLINA.md)
identifica os 51 treinos e 880 condições sintéticas anteriores. O teste
oficial não foi reaberto nessas duas extensões.

## Resultados e preservação

Os PDFs, HTMLs e notebooks originais do IMDb e do Qwen são preservados
byte a byte. O suplemento MLP também mantém seus resultados congelados.
A extensão sequencial usa uma nova pasta de resultados e um novo
congelamento de seleção. Todos os checkpoints planejados são mantidos;
nenhuma semente é escolhida pela auditoria.

Na extensão MLP, uma divergência de precisão foi corrigida e a repetição
preservou os parâmetros bit a bit. A evidência histórica continua no
respectivo diretório. A nova extensão desativa TF32 em treino e avaliação
e registra a reavaliação independente de todos os checkpoints auditados.

As figuras e diagramas novos são próprios. O livro e os slides do acervo
são referências, sem redistribuição de suas páginas ou figuras. A presença
de um slide no acervo não comprova que ele foi ministrado.

## Arquivos maiores e reprodução

O Git contém as predições, métricas, tokenizadores, figuras, relatórios,
notebooks, código e manifestos. Os checkpoints grandes treinados ficam
fora do Git, com hashes no congelamento científico. O corpus preparado,
as dependências e os caches locais também ficam fora da distribuição.

A release original oferece um pacote opcional dos checkpoints originais.
Os novos treinos e a separação dos papéis dos dados são descritos no
[guia de reprodução sequencial](REPRODUCAO_SEQUENCIAS.md). A reprodução
cria uma pasta nova, preservando os resultados publicados.

## Verificar a distribuição

```bash
python tools/verify_publication.py
python verify_sequences.py
python verify_didactic.py
python verify_qwen_improved.py
python -m pytest -q
```

`publication_manifest.json` registra cada arquivo distribuído, tamanho e
hash SHA-256. Ele não substitui os congelamentos científicos de cada estudo.
Reexecutar notebooks pode mudar os seus hashes; use uma cópia de trabalho.

## Dados e autoria

Projeto de Nestor, Letícia e Gabriel, para MAE5920, Introdução a Deep Learning.
Dados e modelos de terceiros mantêm seus termos de uso e atribuição.
As fontes estão no README, nos relatórios e nos catálogos de referências.
O repositório não atribui nova licença aos dados ou modelos de terceiros.
As análises qualitativas assistidas por IA estão identificadas;
não houve anotação humana independente.

## Automação de testes

A credencial de publicação disponível não tinha o escopo `workflow`.
A configuração de CI está preservada em `docs/ci/verificar-projeto.yml`
como modelo, sem workflow ativo. Os testes e verificadores foram
executados localmente; a avaliação dos checkpoints usa a GPU.
