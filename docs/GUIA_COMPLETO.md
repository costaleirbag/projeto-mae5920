# IMDb: modelos, análise de erros e RFM steering

Projeto de Nestor, Letícia e Gabriel para Introdução a Deep Learning.

**Foco didático do Qwen3-8B:** o relatório e o notebook explicam como
intervir nas ativações de um modelo congelado, mantendo texto e prompt fixos.
O material mostra a mesma entrada sem steering e com intervenções nos dois
sentidos. A pergunta é o que muda na decisão e na geração; demonstrar
superioridade sobre prompting não é requisito do trabalho. O sinal da
intervenção não garante o sentimento da saída. Os exemplos são escolhidos
por ordem e rótulo, sem selecionar apenas respostas bem-sucedidas.

**Extensão Qwen3-8B:** consulte `QWEN_EXPERIMENTO.md` para o protocolo de
classificação sem ajuste dos pesos, RFM/AGOP, geração de texto e controles por
instrução. A primeira rodada fica em `results/qwen/`; a melhoria da geração,
em `results/qwen_improved/`. O relatório atual e o notebook de leitura são
`relatorio_qwen3.pdf`, `relatorio_qwen3.html` e
`notebooks/qwen3_8b_steering.ipynb`. O material didático anterior é preservado
em `results/qwen_improved/previous_didactic/`.

Na rodada atual, a direção de geração é uma **diferença de médias no bloco 28**:
128 enredos de treino aparecem com pedidos positivo/negativo, totalizando
256 estados internos. Não se ajustam pesos do Qwen nem se treina um scorer
para essa alternativa. Isso difere da direção original, extraída dos gradientes
de uma RFM sobre 512 resenhas rotuladas. As variantes RFM da busca nova usam
gradientes centrados, uma covariância, distinta da AGOP bruta original.

O prompt da demonstração tem dois campos comuns, FACT e REVIEW, sem indicar
o sentimento-alvo. O deslocamento começa somente após o marcador REVIEW:
gerado pelo modelo. Configuração e intensidade são escolhidas na validação e
congeladas antes de 64 novas fontes de geração (32 reais e 32 sintéticas).
São 704 saídas, incluindo três direções aleatórias, e mais 160 saídas na
reavaliação das 32 fontes antigas. A avaliação considera opinião sobre o filme,
fatos e limites de geração separadamente. Nenhum texto é editado ou descartado.

No teste reservado, o juiz semântico reconheceu os dois sentidos-alvo em
**55/64 fontes**: **24/32 reais** e **31/32 sintéticas**. TF-IDF, CNN e BERT
também registram maior separação nos textos completos. São diagnósticos
automáticos, não acurácia contra anotação humana. O prefixo factual ficou
idêntico nos três sentidos, mas a frase de opinião ainda pode inventar um
cenário; há um caso claro preservado no relatório. Alertas do juiz que parecem
confundir opinião diferente com erro factual recebem notas separadas.

Para ler e conferir a rodada atual, sem carregar o Qwen:

```bash
python execute_notebook.py --notebook notebooks/qwen3_8b_steering.ipynb
python verify_qwen_improved.py
```

`build_qwen_improved_report.py` reconstrói o relatório e o notebook a partir
dos resultados salvos; execute o notebook depois de reconstruí-lo. A inferência
está em `run_qwen_improve.py`, e a seleção congelada em
`results/qwen_improved/selection_frozen.json`. Não reexecute `prepare`, `fit`
ou `freeze` sobre um teste já aberto. Use uma cópia separada para novos estudos.

Na classificação completa, o Qwen congelado obteve **95,260% de acurácia**
e **95,260% de F1 macro**, com **569 FP** e **616 FN** em 25.000 avaliações.
O ganho observado sobre o BERT foi de **4,316 pontos percentuais**.
O Qwen recebe até 768 tokens da fonte; o BERT, 192 WordPieces. O audit com
256 exemplos de validação reduziu a acurácia do Qwen de 93,75% para 89,84%
ao limitar seu prefixo a 192 tokens. Essa comparação não isola a arquitetura.
A inferência completa do Qwen, incluindo carregamento, validação e calibração,
levou cerca de 63,3 minutos, antes do audit final de prompt.

Os pesos do Qwen não são ajustados; a temperatura usa dados rotulados de
calibração e a RFM usa 512 exemplos rotulados de treino. O teste já havia sido
consultado na etapa original: a extensão é exploratória. Possível exposição
ao IMDb no pré-treinamento do Qwen não foi auditada.

No teste de intervenção com texto idêntico, RFM mudou 127/129 das 256 decisões
nos extremos -4/+4; diferença de médias produziu o mesmo efeito. Esse controle
da decisão não se transferiu de modo confiável ao tom da geração. Nas 544
reescritas, a concordância dos três juízes com o alvo foi de 34,4%/50,0% para
RFM negativa/positiva e 78,1%/75,0% para instruções diretas negativa/positiva.
Esses são resultados da primeira rodada: diagnósticos automáticos em 32
fontes, não taxas de sucesso humano.
Instruções diretas também falharam em alguns alvos e omitiram fatos centrais.

O relatório próprio está em `relatorio_qwen3.pdf` e `relatorio_qwen3.html`.
O notebook `notebooks/qwen3_8b_steering.ipynb` lê os resultados salvos sem GPU
ou pesos Qwen e reconstrói a nova direção a partir do pequeno arquivo de
ativações selecionadas. A leitura atual inclui os 12 trios definidos antes
do novo teste; a leitura histórica inclui 60 saídas e 12 erros de
classificação, com origem no assistente declarada. Não houve anotação humana
independente. A planilha e a rubrica em `AVALIACAO_HUMANA.md` estão preparadas
para uma revisão posterior pelo grupo.

O experimento compara **TF-IDF + regressão logística**, **CNN em PyTorch** e
**BERT pré-treinado ajustado**. Inclui probabilidades calibradas, exemplos por
tipo de erro, segmentação semântica e um experimento de activation steering
baseado em **Recursive Feature Machines / AGOP**.

## Execução verificada

No teste oficial com 25.000 avaliações:

| Modelo | Acurácia | F1 macro | FP | FN |
|---|---:|---:|---:|---:|
| TF-IDF + regressão logística | 90,144% | 90,144% | 1.234 | 1.230 |
| CNN | 86,784% | 86,782% | 1.818 | 1.486 |
| BERT-base com ajuste integral | 90,944% | 90,941% | 1.342 | 922 |

O ganho de acurácia do BERT sobre o baseline foi de 0,8 ponto percentual.
Ele reduziu falsos negativos, mas aumentou falsos positivos. Sua acurácia cai
de 93,10% em textos de até 100 palavras para 82,60% acima de 500 palavras.
Há exemplos com evidência favorável fora do prefixo recebido pela rede e
textos cujo rótulo oficial é discutível. Essas leituras não provam causas.

O steering RFM alterou 6 de 256 decisões em cada extremo (-4 e +4). A diferença
de médias teve efeito igual ou maior; não foi demonstrada vantagem da RFM.
A melhor iteração kernel foi a primeira: as atualizações recursivas adicionais
pioraram a AUC de validação. A edição mascarada gerou alguns elogios claros,
mas também mudanças de fatos e frases pouco naturais. Não é reescrita
bidirecional confiável; o controle sem steering já muda palavras e polaridade.

O notebook foi executado, oito testes científicos passaram e as métricas de
todas as divisões foram recalculadas a partir das previsões. Consulte
`results/verification.json`. As interpretações dos exemplos estão em
`results/analysis/qualitative_review.json`, com indicação de sua origem no
assistente; não são avaliações independentes por revisores humanos.

## Resultados e leitura

- `relatorio.html`: relatório autocontido com métricas, figuras e exemplos.
- `relatorio_imdb.pdf`: versão de oito páginas para leitura e apresentação.
- `notebooks/experimentos_imdb.ipynb`: notebook com resultados e verificações.
- `results/comparison.csv`: métricas de todos os modelos e divisões.
- `results/{tfidf,cnn,bert}/`: previsões por exemplo, métricas e custo.
- `results/analysis/`: grupos semânticos, intervalos, FP/FN e exemplos.
- `results/steering/`: direções, RFM, controles e todas as intervenções.
- `results/rewrite/`: textos realmente editados e avaliação sem hooks.
- `data/manifest.json`: origem, revisão, papéis das divisões e auditoria.
- `logs/*_status.json`: estado registrado de cada etapa. Um estado `running`
  sozinho não prova que o processo ainda está vivo; verifique seu PID.

As pastas de resultados só aparecem conforme as etapas são concluídas. Não
interprete a existência de código ou de um checkpoint como treino concluído.

O pacote ZIP para compartilhar omite dados preparados, checkpoints e o cache
de embeddings. Inclui o subconjunto textual e os estados selecionados da nova
rodada. O notebook do Qwen usa apenas esses artefatos, sem restaurar o corpus.
Para executar as análises originais que precisam das tabelas preparadas,
rode `python run.py prepare`. A auditoria completa dos pesos exige os artefatos locais
ou uma nova execução dos treinos.

## Instalar e executar

Python 3.11 ou posterior. Baseline e CNN foram executados em Apple M1;
BERT e análises posteriores usam NVIDIA GB10. Os metadados de cada etapa
registram o ambiente real. Tempos entre esses dispositivos não medem uma
comparação de eficiência em hardware idêntico.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python run.py all
```

Também é possível instalar o pacote com `python -m pip install -e .`. O arquivo
`requirements-lock.txt` fixa o ambiente local usado no baseline, CNN e relatórios;
os metadados do BERT registram sua versão distinta de PyTorch/CUDA.
`pyproject.toml` declara intervalos de compatibilidade. Em outra plataforma,
a disponibilidade de wheels pode exigir uma versão equivalente do PyTorch.

Execução por etapas:

```bash
python run.py prepare
python run.py baseline
python run.py cnn
python run.py bert
python run.py analyze
python run.py steer
python run.py rewrite
python run.py report
python -m pytest -q
```

A configuração fica em `config.json`. A execução não contrata GPU,
não envia avaliações para serviços de LLM e não precisa de credenciais para
esses datasets/modelos públicos. CUDA é usada quando disponível; caso contrário,
MPS ou CPU. O BERT usa atenção `eager` para preservar dropout em MPS.

## Protocolo de dados

Fonte: [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb).
Revisões de origem são fixadas em `config.json` e registradas nos metadados,
assim como os hashes dos arquivos preparados.

1. Preserve o teste oficial com 25.000 avaliações.
2. Remova do treino textos repetidos no teste, usando somente hashes dos textos.
3. Remova repetições e conflitos de rótulo internos do treino.
4. Separe o restante em treino, validação e calibração, com estratificação e
   sementes fixas. Nenhum texto idêntico pode atravessar as divisões.
5. Ajuste vocabulários/TF-IDF/pesos no treino; selecione C e checkpoint na
   validação; ajuste temperatura somente na calibração.
6. Avalie uma vez os modelos escolhidos no teste e realize a análise descritiva
   dos erros. Não volte a escolher modelos com base nessa análise.

Na preparação registrada: treino **19.823**, validação **2.478**, calibração
**2.478**, teste **25.000**. Excluíram-se 123 linhas do treino repetidas no teste e
98 repetições internas. O teste oficial retém 201 repetições internas; intervalos
baseados em independência estrita devem ser interpretados com essa limitação.

O arquivo de notas/metadados nunca é uma variável preditora. Menções de notas
dentro do próprio texto são tratadas como característica a investigar. As
50.000 avaliações adicionais não rotuladas da distribuição original não são
usadas no ajuste.

**Não existe classe neutra no IMDb usado aqui.** Escore próximo de 0,5 indica
incerteza. Exemplos neutros de demonstração são sintéticos e identificados.

## Modelos e orçamento

- TF-IDF: 80 mil características no máximo, unigramas/bigramas, negações
  preservadas, busca de C = 0,5 / 2 / 8 na validação.
- CNN: vocabulário do treino, embeddings aprendidos de 128 dimensões, filtros
  3/4/5, pooling que exclui padding, dropout, até 400 palavras, até cinco épocas.
- BERT-base-uncased: até 192 WordPieces, ajuste integral por duas épocas,
  batch de 32, taxa inicial de 2e-5 e warmup de 10% dos passos.
  `train_last_n_layers=-1` habilita o ajuste integral; valores positivos permitem
  uma nova execução com ajuste parcial quando a memória for limitada.

Os textos completos chegam ao baseline; as redes recebem prefixos limitados.
Reporte essa diferença, as sementes e o custo antes de atribuir um resultado
somente à arquitetura. Checkpoints são escolhidos por F1 macro de validação.

## Erros e análise semântica

Classe positiva = avaliação positiva. **FP / tipo I:** negativo previsto como
positivo. **FN / tipo II:** positivo previsto como negativo.

Use acurácia, F1 macro, log loss, Brier, ECE e curvas de confiabilidade. As taxas
de FP/FN por grupo têm denominadores condicionais à classe real e intervalos de
Wilson. Para caracterizar erros, compare FP a TN e FN a TP.

Um encoder independente, `all-MiniLM-L6-v2`, representa as avaliações. O K-means
aprende oito clusters em amostra do treino sem rótulos e atribui todos os textos
do teste. Há também marcadores heurísticos de negação, contraste, opiniões
mistas, nota explícita e possível ironia. Esses marcadores são sobrepostos e
ruidosos. Padrões encontrados são **associações exploratórias**, não causas
comprovadas nem rótulos semânticos humanos. O encoder também tem limite de tokens.

## RFM / AGOP: o que está implementado

Referência: [Beaglehole et al., *Toward universal steering and monitoring of AI
models*](https://arxiv.org/html/2502.03708v2), equações 1 e 2.

Para cada bloco candidato, usa-se a representação CLS. O RFM alterna:

\[
K_M(x,z)=\exp[-\sqrt{(x-z)^T M(x-z)}/L],\qquad
\alpha=(K_M+\lambda I)^{-1}(y-\bar y),
\]
\[
M_{t+1}=\frac{1}{n}\sum_i \nabla f_t(h_i)\nabla f_t(h_i)^T.
\]

O autovetor principal da AGOP é orientado pela correlação positiva com o rótulo.
Gradientes são analíticos e testados contra autograd. O código usa a atualização
pela AGOP bruta, sem substituí-la por PCA, pesos logísticos ou diferenças de
médias. Centramento e escala escalar única são aprendidos no treino.

**AGOP não é a Hessiana exata.** Para a perda quadrática de um preditor escalar,
a Hessiana com respeito à entrada contém um termo `2 grad(f) grad(f)^T` e um
termo residual `2(f-y) Hessian(f)`. A AGOP corresponde à média do primeiro termo
(à constante perto), não automaticamente à curvatura completa do BERT.

Camada, largura e iteração são escolhidas em validação. A intervenção é
`h_CLS <- h_CLS + alpha * scale * v`. Todas as direções têm norma unitária e
recebem uma escala comum, fixada no treino. Controles: diferença de médias,
sonda logística e cinco vetores aleatórios. São verificados intensidade zero,
equivalência do prefixo em cache e remoção dos hooks após a intervenção.

Esse é um teste de controle da **decisão do classificador com texto idêntico**.
Ele não reescreve a avaliação e não prova mudança de sentimento humano.
Probabilidades sob intervenção ficam fora do regime da calibração usual.

## Extensão que modifica o texto

Para experimentar a transformação de avaliações neutras/negativas em textos
com outra polaridade, há uma etapa separada de **edição mascarada**. A cabeça
masked-LM recebe exatamente o encoder ajustado. A transformação e o bias vêm
do pré-treinamento; a projeção de saída continua ligada aos embeddings
ajustados. Duas palavras de opinião são mascaradas e preenchidas sob steering
RFM ou controle aleatório.

São mantidos todos os resultados, incluindo falhas. Não há reranqueamento
guiado por sentimento. É preciso comparar com intensidade zero: retirar as
palavras e preenchê-las já pode alterar o tom. Classificadores sem hooks e
similaridade semântica fornecem diagnósticos, mas fluência, sentimento e fatos
exigem leitura humana. Isso não é geração autoregressiva nem reprodução
integral dos experimentos com Llama do artigo.

## Testes científicos

Os testes verificam vazamento por duplicações, invariância da CNN a padding,
definições de FP/FN, temperatura, gradientes analíticos da RFM, AGOP positiva
semidefinida, orientação da direção e intervenção/restauração de hooks. O
caminho de continuação do BERT a partir de uma camada em cache é comparado a um
forward completo. Testes não substituem as métricas do experimento real.

## Fontes

- [Maas et al. / Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
- [PyTorch](https://docs.pytorch.org/docs/stable/)
- [Beaglehole et al.](https://arxiv.org/abs/2502.03708)
- [Implementação dos autores](https://github.com/dmbeaglehole/neural_controllers)

O projeto é uma implementação educacional independente e explicita as adaptações.
