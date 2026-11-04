# Escopo da publicação

[Início](../README.md) · [Guia do projeto](GUIA_COMPLETO.md) · [Resultados](RESULTADOS.md)

A versão atual reúne o experimento original de sentimento no IMDb, as duas rodadas
Qwen, os mecanismos da disciplina, CNNs/encoders, BPE e o notebook único. O conteúdo
está disponível na página principal do repositório e na release v1.1.0. A
[release v1.0.0](https://github.com/costaleirbag/projeto-mae5920/releases/tag/v1.0.0)
preserva a entrega histórica.

## O que está incluído

- Cinco notebooks executados: um roteiro único autocontido e quatro específicos.
- Quatro relatórios PDF e suas versões HTML: IMDb, Qwen, mecanismos, CNN/encoders.
- Uma apresentação editável com diagramas, gráficos e notas de fala.
- Guia para novos leitores, tabelas de métricas, protocolos e reprodução.
- Código, configurações, revisões de modelos, curvas, previsões e gerações.
- Evidências de seleção por validação, controles, falhas e verificação independente.

O [estudo sequencial](PROTOCOLO_SEQUENCIAS.md) contém 87 ajustes; o
[BPE](PROTOCOLO_BPE.md) acrescenta três ajustes e reutiliza três checkpoints por
palavras. O [suplemento MLP](CONEXOES_DISCIPLINA.md) contém 51 treinos e 880 condições
sintéticas. Cada estudo identifica seus papéis de dados e suas limitações.

## Preservação e interpretação

Esta revisão da documentação não altera pesos, previsões, seleções, relatórios,
slides ou notebooks. Os resultados científicos permanecem congelados; tabelas de
visão geral são derivadas das fontes preservadas e trazem proveniência.

As auditorias internas das extensões são diferentes do teste oficial e entre si.
O teste oficial não foi reaberto nessas extensões. A primeira rodada Qwen e a direção
RFM/AGOP continuam preservadas; a demonstração atual de geração usa diferença de
médias. O material identifica o que é exercício sintético, resultado real e leitura
qualitativa assistida por IA. Não houve anotação humana independente.

Na extensão MLP, uma divergência de precisão foi corrigida e a repetição preservou
os parâmetros bit a bit; a evidência histórica continua no diretório de proveniência.
As novas extensões registram sua política numérica e reavaliações independentes.

Figuras e diagramas do projeto são próprios. Livro e slides do acervo são
referências, sem redistribuição de páginas ou figuras. A presença no acervo não
comprova que o conteúdo foi ministrado.

## Arquivos maiores e downloads

O Git inclui os materiais de leitura e as evidências numéricas selecionadas.
O corpus preparado, checkpoints treinados, ambientes e caches maiores ficam fora
do Git. O notebook único incorpora o necessário ao caminho didático padrão, mas
não os pesos dos modelos grandes.

A release v1.1.0 oferece o pacote completo de leitura e os entregáveis principais
individuais. O complemento histórico de pesos originais está na v1.0.0; não inclui
os novos checkpoints sequenciais/BPE. O [guia de reprodução](REPRODUCAO.md) explica
como conferir resultados salvos, restaurar complementos ou repetir os treinos.

## Integridade da distribuição

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

`publication_manifest.json` registra tamanho e SHA-256 dos arquivos distribuídos;
não substitui os congelamentos científicos. O pacote e os arquivos individuais da
release têm hashes em `SHA256SUMS.txt`. Reexecutar notebooks pode mudar seus hashes;
trabalhe em uma cópia e confira a publicação antes de modificá-la.

## Dados, autoria e automação

Projeto de Nestor, Letícia e Gabriel para MAE5920 — Introdução a Deep Learning.
Dados e modelos mantêm seus termos de uso e atribuição. O repositório não atribui
nova licença aos materiais de terceiros.

A configuração de CI permanece como template em `docs/ci/verificar-projeto.yml`,
sem workflow ativo: a credencial usada na publicação não tinha essa permissão.
Os testes e verificadores foram executados localmente; a reavaliação dos checkpoints
grandes foi feita no ambiente científico descrito nos resultados.
