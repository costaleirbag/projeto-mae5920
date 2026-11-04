# Notas da apresentação

Os slides 1–14 formam o roteiro principal. Os slides 15–24 são apoio. Os slides 22–24 acrescentam BPE e podem entrar após o slide 4. Para uma fala curta, priorize 1–3, 5–6 e 9–14.

## 1. Sentimento, redes e treinamento

Objetivo didático: relacionar os classificadores com convoluções, atenção, otimização e generalização. A parte principal acompanha o estudo de texto sequencial. Há material de apoio para MLPs, double descent e steering, cada um com seu próprio protocolo.

## 2. Amostras e escolhas do experimento

Dados: results/sequences/data_manifest.json e selection_frozen.json. O prefixo de até 128 palavras é comum a todos. CNN e encoders têm limite de 256 tokens e taxas de truncagem próprias; o TF-IDF lê o prefixo completo. A nova auditoria exclui todos os papéis da extensão MLP. O teste oficial não foi reaberto. Selecionamos época e taxa somente pela CE de validação.

## 3. Arquitetura da TextCNN

Fonte: src/imdb_lab/models.py e sequence_config.json. Embedding treinável 128, Conv1D paralelas de janelas 3, 4 e 5, cem filtros por janela, ReLU, máximo global com máscara de padding, concatenação de 300, dropout e Linear(300,2). Inspiração: Kim (2014), https://aclanthology.org/D14-1181/. Prince capítulo 10 e slides Redes Convolucionais, pp. 11–22 e 36.

## 4. Convolução: padrões locais com pesos compartilhados

A frase é um exemplo ilustrativo, não uma medição de ativação. O kernel combina embeddings, não regras fixas para palavras. Cada filtro aprende seus pesos e desliza por toda a entrada. ReLU e máximo global resumem a resposta. Uma janela de 3 tokens não equivale a atenção global. Prince capítulo 10 e slides Redes Convolucionais, pp. 12–20.

## 5. CNN: otimizador e taxa de aprendizado

Dados: results/sequences/runs e presentation_data.json. Linhas são médias de três sementes. Pesos iniciais e minibatches pareados. Doze épocas, batch 64, clipping 1, mesmo dropout. Esquerda: taxa comum 0,001. Direita: cada método usa a melhor média de CE de validação entre três taxas candidatas. Não selecionamos a melhor semente. Faixas completas estão no relatório. Prince 6.1–6.5.

## 6. CNN: ajuste do treino e overfitting

Dados: results/sequences/presentation_data.json. Curvas médias das três sementes. Dropout desligado para calcular CE de treino e validação. Mesmo Adam e taxa selecionada para a CNN principal. O treino segue até 12 épocas, mas o checkpoint usado na auditoria vem da menor CE de validação. Prince capítulos 8–9; slides Regularização.

## 7. CNN: janelas e quantidade de filtros

Dados: results/sequences/summary.json. Três sementes em cada variante; auditoria interna de 2.048 entradas. A grade de arquiteturas reutiliza a taxa de Adam selecionada para a CNN principal, sem retuning por variante. O embedding contribui com muitos parâmetros, portanto reduzir filtros não reduz toda a rede na mesma proporção.

## 8. Representações contextuais no Transformer

Fonte: Prince capítulos 11–12; slides Transformers pp. 7–18 e 21–31; Devlin et al., https://aclanthology.org/N19-1423/. Diagrama funcional: operações e normalização têm ordens distintas entre famílias. Q, K e V têm projeções aprendidas; softmax de compatibilidades controla a combinação de valores. A média usada neste projeto exclui padding e tokens especiais.

## 9. Os três encoders treinados

Fontes: models_pinned.json e arquivos *_architecture.json; Devlin et al. (2019), Sanh et al. (2019), https://arxiv.org/abs/1910.01108; Warner et al. (2024), https://arxiv.org/abs/2412.13663; https://huggingface.co/answerdotai/ModernBERT-base. Contagens incluem nossa cabeça de média mais Linear. BERT não inclui pooler. DistilBERT incorpora destilação no pré-treinamento público; não executamos a destilação neste trabalho. ModernBERT usa posições rotativas e atenção local intercalada com global.

## 10. Encoder congelado e ajuste completo

Dados e política: sequence_config.json, docs/PROTOCOLO_SEQUENCIAS.md. Congelado: encoder em avaliação, vetores pré-computados, cabeça com dropout 0,1, Adam por 40 épocas. Completo: dropout do encoder ativo durante treino, AdamW por 3 épocas, warmup 10%, redução linear e decaimento 0,01. Cabeças compartilham partida por semente. Comparação entre receitas práticas; múltiplos fatores mudam junto da atualização do encoder.

## 11. Encoders na nova auditoria interna

Dados: results/sequences/summary.json. Mesmos 2.048 exemplos e prefixos comuns. Cada barra é a média de três sementes após escolher taxa e época na validação. Taxas diferem por encoder e regime. A amostra é interna. Os valores do teste oficial dos relatórios anteriores não entram nesta figura. Probabilidades não calibradas.

## 12. Qualidade e custo do ajuste

Dados: summary.json, training_seconds_mean, feature_extraction_seconds e audit_end_to_end_seconds. A coluna de ajuste exclui avaliações entre épocas. Congelado adiciona extração dos vetores de treino e validação. A inferência inclui o encoder também no modo congelado, batch 64. Download, tokenização e carregamento excluídos. NVIDIA GB10, BF16 para encoders, FP32 para CNN. Medição descritiva desta implementação, sem otimizar kernels por família.

## 13. Um mesmo texto, decisões diferentes

Texto próprio: synthetic_examples.json, synthetic_reversal. Os modelos usam a semente 17, sem escolher pela auditoria. Texto definido antes de observar as predições e avaliado após o congelamento dos modelos. Intenção positiva do autor, sem constituir benchmark. As probabilidades não foram calibradas.

## 14. O que relacionamos com as aulas

Fontes: docs/REFERENCIAS_SEQUENCIAS.json, docs/REFERENCIAS_DISCIPLINA.json, resultados originais e extensões. O double descent é uma regressão sintética com camada aleatória congelada. O steering altera ativações de Qwen3-8B com pesos congelados. Esses mecanismos não foram testados como melhorias da CNN.

## 15. Grades de hiperparâmetros

Fonte: sequence_config.json. Três sementes 17, 29, 43 em toda condição neural. Seleção da taxa pela média da menor CE de validação por execução; checkpoint escolhe a primeira época com mínimo. Sementes publicadas integralmente.

## 16. Curvas do ajuste completo

Fonte: results/sequences/presentation_data.json. Curvas médias de três sementes com a taxa escolhida separadamente para cada encoder. Época zero usa cabeça aleatória e não participa da seleção. CE de validação, não acurácia de auditoria, escolhe taxa e época.

## 17. Variação entre as três sementes

Fonte: summary.json. A faixa é mínimo a máximo entre três inicializações na mesma partição. Não é intervalo de confiança amostral. Todas as execuções e configurações descartadas estão publicadas. Auditoria interna de 2.048 exemplos.

## 18. Como a última camada lê a CNN

Fonte: synthetic_examples.json. Margem logit positivo menos negativo = viés + soma dos máximos dos filtros vezes a diferença dos pesos da saída. A decomposição é exata com dropout desligado. Agrupamos filtros que ganharam na mesma janela. Isso não identifica efeito causal de uma palavra: alterar a entrada pode mudar outros máximos.

## 19. Double descent no exemplo sintético

Estudo separado, anterior aos treinos de CNN/encoders: results/didactic/double_descent.csv. Vinte sementes prescritas, n=64, camada ReLU aleatória congelada. Ajustamos p coeficientes de saída por norma mínima ou ridge lambda 0,001. Cada largura p ocupa uma categoria: o espaçamento horizontal não representa diferenças numéricas entre larguras. O eixo vertical mostra log10 da mediana do MSE, calculada sobre todas as vinte sementes. O pico ocorre em p=64. O relatório próprio mostra o eixo quantitativo p/n, faixas interquartis e todas as 880 condições. Não demonstramos double descent na CNN. Fonte: Prince 8.4 e Belkin et al. (2019), https://arxiv.org/abs/1812.11118.

## 20. Steering no Qwen3-8B congelado

Estudo anterior, sem nova inferência do Qwen: results/qwen_improved/presentation_examples.json, primeira fonte sintética do conjunto reservado, sem seleção por resultado. Direção unitária pela diferença de médias dos estados de 128 fontes com instruções positivas e negativas no bloco 28. h novo = h + alpha vezes escala vezes direção. Pesos e prompt permanecem iguais. A intervenção começa depois do marcador REVIEW. O prefixo FACT tem tokens idênticos nas três condições. O objetivo é demonstrar uma intervenção nas ativações, sem provar superioridade sobre prompting.

## 21. Fontes e limites

Referências completas: docs/REFERENCIAS_SEQUENCIAS.json e docs/REFERENCIAS_DISCIPLINA.json. Fontes externas primárias: Kim https://aclanthology.org/D14-1181/; Devlin https://aclanthology.org/N19-1423/; Sanh https://arxiv.org/abs/1910.01108; Warner https://arxiv.org/abs/2412.13663; modelo https://huggingface.co/answerdotai/ModernBERT-base. Diagramas e gráficos próprios. Nenhuma página dos PDFs de referência foi redistribuída.

## 22. BPE aprendido no texto de treino

Fontes: results/bpe/tokenizer_training.json, tokenizer.json e bpe_config.json. BPE de caracteres, WhitespaceSplit e marcador </w>, após a mesma limpeza por regex da CNN por palavras. Não é BPE de bytes. Ajustamos regras e vocabulário em 6.144 textos de treino sem rótulos, depois congelamos o tokenizer e treinamos a CNN por gradiente. Sennrich et al. (2016), https://aclanthology.org/P16-1162/; documentação oficial https://huggingface.co/docs/tokenizers/quicktour. O algoritmo aprende pares frequentes, sem garantir que sejam morfemas. Os exemplos são saídas reais do tokenizer aprendido.

## 23. Controle da comparação de tokenização

Fonte: results/bpe/selection_frozen.json, data_manifest.json e verification.json. Três novos treinos BPE e três checkpoints por palavras reutilizados. Mesmas matrizes numéricas iniciais, ordens de minibatches, 12 épocas, batch 64, clipping 1, dropout 0,5, Adam 0,001, arquitetura 3/4/5 com cem filtros e embedding 128. Taxa herdada da CNN por palavras, sem retuning. Seleção independente da época pela CE de validação. Nova auditoria de 2.048 resenhas exclui todos os papéis dos estudos MLP e CNN/encoders. As seis escolhas foram congeladas antes de liberar a entrada. O teste oficial não foi aberto. Não misturar esta auditoria com a tabela dos encoders.

## 24. Cobertura e acurácia medem efeitos diferentes

Resultados: results/bpe/summary.json e audit_predictions.npz. Mesma nova auditoria interna para palavras e BPE, média das três sementes. Palavras: 81,9987%, CE 0,3964. BPE: 81,4128%, CE 0,4052. Diferença média BPE menos palavras: -0,58594 ponto percentual; diferenças por semente -0,43945, -0,83008, -0,48828 p.p. UNK: 3,26984% para 0%. Média de tokens 121,30 para 129,09, sem truncagem. Este ensaio não retesta a taxa para BPE e não demonstra inferioridade universal do algoritmo. Aumentar cobertura não garante melhorar esta CNN. Seis checkpoints reproduziram exatamente os logits em outro processo. Três sementes e um holdout não capturam toda a incerteza.
