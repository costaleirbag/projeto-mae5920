# Escopo da publicação

[Voltar ao início](../README.md)

Esta distribuição reúne o experimento principal de sentimento no IMDb e
a extensão didática com Qwen3-8B. As versões atuais dos relatórios ficam
na raiz; as versões anteriores estão identificadas em
`results/qwen_improved/previous_didactic/`.

## Conteúdo

- Dois relatórios em PDF e suas versões HTML.
- Dois notebooks executados.
- Código de treinamento, inferência, intervenção, análise e geração de relatórios.
- Configurações científicas, revisões de origem e auditoria das divisões.
- Previsões, métricas, figuras, fontes selecionadas e todas as gerações consolidadas.
- Evidências da seleção em validação, controles aleatórios, falhas e leituras qualitativas.
- Testes e verificadores de integridade e do protocolo científico.

O ZIP da entrega reproduz esse material, incluindo a documentação.
O pacote opcional de reprodução contém os checkpoints e os caches numéricos
maiores. O corpus completo pode ser obtido da fonte pelo comando de preparação.

## Preservação dos resultados

Os dois PDFs atuais, os dois HTMLs atuais e os dois notebooks atuais foram
copiados da entrega verificada, sem modificar seu conteúdo. Os hashes
constam de `publication_manifest.json`.

A página inicial foi reescrita para orientar a leitura no GitHub.
O README detalhado anterior foi preservado em `docs/GUIA_COMPLETO.md`.
Foram acrescentados guias de reprodução, uma capa vetorial, verificação da
distribuição e um modelo de automação de testes. O empacotador foi adaptado para incluir
os novos arquivos. Nenhum treinamento ou nova inferência foi realizado
durante a publicação.

No notebook **histórico**, em `previous_didactic/`, foi removida apenas uma
saída de aviso que exibia um caminho absoluto da máquina de execução.
O notebook atual não recebeu essa alteração. Textos gerados, rótulos,
previsões, parâmetros congelados e evidências científicas foram preservados.

Arquivos temporários de retomada foram omitidos quando as mesmas saídas
já estavam preservadas nos JSONs consolidados. Caches de ambiente,
arquivos temporários e configuração de acesso à infraestrutura são locais.

## Verificar a cópia publicada

```bash
python tools/verify_publication.py
```

`publication_manifest.json` identifica os arquivos distribuídos, seus
tamanhos e hashes SHA-256. Ele descreve a distribuição, sem substituir
`selection_frozen.json`, `report_build.json` ou os metadados científicos
históricos. As verificações científicas mantêm seus próprios requisitos,
descritos no [guia de reprodução](REPRODUCAO.md).

## Dados e autoria

Projeto de **Nestor, Letícia e Gabriel**, para MAE5920 — Introdução a Deep Learning.
Dados e modelos de terceiros mantêm seus termos de uso e atribuição;
as fontes estão listadas no README e nos relatórios.
O repositório não atribui uma nova licença aos dados ou aos modelos de terceiros.

## Automação de testes

O GitHub bloqueou o envio de um workflow com a credencial disponível, que
não tinha o escopo `workflow`. A configuração foi preservada como
`docs/ci/verificar-projeto.yml`, pronta para ativação futura com uma
credencial adequada. Os testes e verificadores já passaram localmente;
não há workflow ativo nesta versão.
