# Entrega didática — IMDb e Qwen3-8B

Projeto de Nestor, Letícia e Gabriel para MAE5920 — Introdução a Deep Learning.

## Leitura recomendada

1. [Relatório principal: modelos no IMDb](https://github.com/costaleirbag/projeto-mae5920/blob/main/relatorio_imdb.pdf).
2. [Relatório Qwen3-8B e steering](https://github.com/costaleirbag/projeto-mae5920/blob/main/relatorio_qwen3.pdf).
3. [Notebooks executados](https://github.com/costaleirbag/projeto-mae5920/tree/main/notebooks).

O experimento principal compara TF-IDF, CNN e BERT. A extensão usa um
Qwen3-8B congelado para ilustrar intervenções nas ativações: mesma entrada,
mesmo prompt e saídas sem steering e com deslocamentos negativos e positivos.
As tentativas anteriores, os controles e as falhas estão preservados.

## Downloads

- **`imdb_qwen3_entrega.zip`**: relatórios PDF/HTML, notebooks, código,
  documentação, figuras, previsões e todas as gerações consolidadas.
- **`artefatos_reproducao.zip`**: complemento opcional com os checkpoints
  treinados de TF-IDF/CNN/BERT e os caches numéricos históricos maiores.
  Extraia na raiz de uma cópia do projeto.
- **PDFs e HTMLs individuais**: disponíveis também como downloads diretos.
- **`SHA256SUMS.txt`**: hashes dos arquivos desta release.

O corpus completo é restaurado da revisão pública fixada do IMDb.
Os pesos-base do Qwen não são incluídos; o notebook didático usa os
artefatos salvos e pode ser reexecutado sem GPU.

## Verificações

Os 13 testes de invariantes científicos passaram na cópia preparada para
publicação. A verificação da rodada atual confirmou a direção reconstruída,
as 704 saídas de teste, as 160 saídas da reavaliação, os prompts e as fontes,
os controles aleatórios e os hashes dos entregáveis. Os relatórios atuais e
os dois notebooks foram preservados sem alteração em relação à entrega local.

Resultados de sentimento na geração são diagnósticos automáticos;
não houve anotação humana independente nem demonstração de superioridade
do steering sobre prompting.

Consulte o [guia de reprodução](https://github.com/costaleirbag/projeto-mae5920/blob/main/docs/REPRODUCAO.md)
e o [escopo da publicação](https://github.com/costaleirbag/projeto-mae5920/blob/main/docs/PUBLICACAO.md).
