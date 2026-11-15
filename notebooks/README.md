# Qual notebook abrir?

[Voltar ao projeto](../README.md) · [Resultados](../docs/RESULTADOS.md)

**[seminario_completo.ipynb](seminario_completo.ipynb) é o notebook de entrega.**
O sumário associa cada seção a uma página do relatório e aos slides. O arquivo
incorpora as previsões necessárias à conferência em CPU e o código e as
configurações usados na reprodução.

| Notebook | Quando usar | Para reexecutar |
|---|---|---|
| **[Seminário completo](seminario_completo.ipynb)** | Conferir e entregar | Conferência em CPU ou nova execução no modo de reprodução |
| [CNN e encoders](cnn_encoders.ipynb) | Consultar grades, curvas, BERT/DistilBERT/ModernBERT e BPE em detalhe | Repositório e resultados incluídos; sem treinar novamente ou carregar os encoders |
| [Conexões com a disciplina](conexoes_disciplina.ipynb) | Aprofundar otimização, MLPs, resíduos, regularização e double descent | Repositório e resultados incluídos; exercícios/reconstruções em CPU |
| [Qwen3-8B e steering](qwen3_8b_steering.ipynb) | Reconstruir direções e percorrer exemplos e controles das rodadas Qwen | Repositório e estados/saídas incluídos; sem carregar os pesos do Qwen |
| [Experimento original IMDb](experimentos_imdb.ipynb) | Examinar a análise original dos modelos e dos erros | Algumas análises exigem restaurar as tabelas do IMDb; veja a reprodução |

Todos estão publicados com saídas. **Para ler no GitHub, nenhum exige instalar
dependências.** Para rodar, siga o [guia de reprodução](../docs/REPRODUCAO.md) e
use uma cópia: a execução pode mudar tempos, metadados e hashes.

O [guia do notebook único](../docs/NOTEBOOK_UNICO.md) descreve os dois modos de
execução e o teste de portabilidade.
