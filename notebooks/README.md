# Qual notebook abrir?

[Voltar ao projeto](../README.md) · [Resultados](../docs/RESULTADOS.md)

**Comece por [seminario_completo.ipynb](seminario_completo.ipynb).** É o roteiro
único, de ponta a ponta, para estudar e compartilhar com o grupo. Tem 15 capítulos,
código real das arquiteturas, parâmetros, tokenização, treinamento, gráficos e
resultados. O arquivo incorpora as evidências necessárias ao caminho padrão;
você pode compartilhar somente esse `.ipynb`.

| Notebook | Quando usar | Para reexecutar |
|---|---|---|
| **[Seminário completo](seminario_completo.ipynb)** | Estudar e apresentar o percurso inteiro | Dependências comuns de Python/PyTorch; exercícios em CPU e evidências incorporadas |
| [CNN e encoders](cnn_encoders.ipynb) | Consultar grades, curvas, BERT/DistilBERT/ModernBERT e BPE em detalhe | Repositório e resultados incluídos; sem treinar novamente ou carregar os encoders |
| [Conexões com a disciplina](conexoes_disciplina.ipynb) | Aprofundar otimização, MLPs, resíduos, regularização e double descent | Repositório e resultados incluídos; exercícios/reconstruções em CPU |
| [Qwen3-8B e steering](qwen3_8b_steering.ipynb) | Reconstruir direções e percorrer exemplos e controles das rodadas Qwen | Repositório e estados/saídas incluídos; sem carregar os pesos do Qwen |
| [Experimento original IMDb](experimentos_imdb.ipynb) | Examinar a análise original dos modelos e dos erros | Algumas análises exigem restaurar as tabelas do IMDb; veja a reprodução |

Todos estão publicados com saídas. **Para ler no GitHub, nenhum exige instalar
dependências.** Para rodar, siga o [guia de reprodução](../docs/REPRODUCAO.md) e
use uma cópia: a execução pode mudar tempos, metadados e hashes.

O notebook único identifica o que é exercício pequeno, resultado real já salvo
e treino completo opcional. Executar suas células padrão não repete os 90 ajustes
sequenciais, os 51 treinos MLP nem a inferência do Qwen.

No Jupyter/VS Code, use o sumário lateral. No GitHub, procure o número/título da
seção na busca do navegador. O [guia do notebook único](../docs/NOTEBOOK_UNICO.md)
descreve todos os capítulos e o teste de portabilidade.
