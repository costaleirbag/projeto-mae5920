# v1.1.0 — Seminário completo, CNNs, BPE e Transformers

Projeto de Nestor, Letícia e Gabriel para MAE5920 — Introdução a Deep Learning.

A entrega passa a ter um ponto de entrada único: um notebook autocontido, executado,
com código real da arquitetura, tokenização, treinamento, hiperparâmetros, resultados
e intervenções nas ativações. A página principal e o guia foram organizados para
quem não acompanhou o desenvolvimento.

## Comece por aqui

1. [Notebook único do seminário](https://github.com/costaleirbag/projeto-mae5920/blob/main/notebooks/seminario_completo.ipynb).
2. [Guia do projeto](https://github.com/costaleirbag/projeto-mae5920/blob/main/docs/GUIA_COMPLETO.md).
3. [Acurácia, F1 macro e loss por estudo](https://github.com/costaleirbag/projeto-mae5920/blob/main/docs/RESULTADOS.md).
4. [Apresentação editável](https://github.com/costaleirbag/projeto-mae5920/blob/main/output/slides/seminario_mae5920.pptx).

## Conteúdo

- Notebook com 15 capítulos, 45 células executadas, 16 figuras, 206 arquivos de
  evidências incorporados e 476 verificações. O caminho padrão executa exemplos
  pequenos em CPU e lê resultados reais preservados; treinos completos são opcionais.
- 90 treinos neurais na extensão sequencial: 42 CNNs, 27 cabeças sobre encoders
  congelados, 18 ajustes completos e três CNNs com BPE. Três sementes por condição.
- BERT, DistilBERT e ModernBERT, com tokenização, pooling, arquiteturas e grades.
- Suplemento de mecanismos com 51 treinos de MLP e 880 condições sintéticas.
- Qwen3-8B congelado: classificação e demonstração de steering, com a direção
  reconstruída de 256 estados reais e saídas nos três sentidos, incluindo falhas.
- Tabelas e CSV com acurácia, F1 macro e CE para as comparações principais,
  identificando as três amostras de avaliação; fontes e hashes incluídos.
- Relatórios, cinco notebooks, código, protocolos, referências e slides editáveis.
  Os resultados originais permanecem preservados.

## Downloads

- **`mae5920_seminario_completo.zip`**: distribuição de leitura completa com
  código, documentação, relatórios, slides, notebooks e resultados salvos.
- **`seminario_completo.ipynb`**: somente o roteiro único, para compartilhar com o grupo.
- **`seminario_mae5920.pptx`** e **quatro PDFs**: entregáveis individuais.
- **`resumo_metricas.csv`**: 23 linhas com as métricas e identificação do estudo.
- **`publication_manifest.json`** e **`SHA256SUMS.txt`**: integridade dos arquivos.

O pacote não inclui corpus preparado nem checkpoints treinados. Para pesos e
caches originais, a v1.0.0 mantém seu complemento histórico; os novos checkpoints
sequenciais/BPE devem ser restaurados de uma execução compatível ou reproduzidos.
Os pesos-base do Qwen não são distribuídos.

## Verificação e limites

O notebook foi executado sozinho fora do repositório, com GPU oculta e conexões de
sockets Python bloqueadas. Seu fluxo opcional também passou por uma execução de
uma época por representação em CPU, com dimensões completas; isso testa o caminho,
sem acrescentar um resultado ao benchmark.

Os testes de invariantes e verificadores de publicação, notebook, sequências, BPE,
MLP, Qwen e tabelas passaram. Eles conferem matemática, código incorporado,
seleções, métricas, fontes, prompts, tokens e hashes, sem repetir os treinos grandes.
O template de CI está incluído; não há workflow ativo.

As auditorias internas são distintas e não devem ser comparadas como um ranking
único. Double descent é demonstrado na regressão sintética. Diagnósticos da geração
usam o próprio Qwen; não houve anotação humana independente nem demonstração de
superioridade do steering sobre prompting. Os protocolos explicam os limites.

[Reprodução](https://github.com/costaleirbag/projeto-mae5920/blob/main/docs/REPRODUCAO.md) ·
[Escopo da publicação](https://github.com/costaleirbag/projeto-mae5920/blob/main/docs/PUBLICACAO.md) ·
[Entrega histórica v1.0.0](https://github.com/costaleirbag/projeto-mae5920/releases/tag/v1.0.0)
