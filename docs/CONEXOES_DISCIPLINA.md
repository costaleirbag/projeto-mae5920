# Da classificação aos mecanismos

[Início](../README.md) · [Suplemento em PDF](../output/pdf/relatorio_disciplina.pdf) ·
[Notebook executado](../notebooks/conexoes_disciplina.ipynb) · [Galeria HTML](../relatorio_disciplina.html)

Esta extensão acrescenta experimentos ligados ao Prince e aos slides
disponíveis no acervo do grupo. Os gráficos são originais, gerados pelas
execuções salvas. O livro e os slides são referências; sua disponibilidade
não demonstra quais páginas foram efetivamente ministradas em aula.

## Quatro blocos para a apresentação

| Pergunta | Alteração controlada | Material de referência |
|---|---|---|
| Adam é sempre melhor que SGD? | MLP, dados, pesos iniciais e minibatches iguais; taxa comum e três taxas candidatas por método | Prince 6.1-6.5; slides Ajuste de Modelos, pp. 26-39 |
| Mais profundidade ajuda? | 1/4/12 camadas; ablações de atalho e escala com parâmetros e pesos iniciais iguais nas redes de 12 camadas | Prince 7.5 e 11.1-11.3; slides Inicialização, pp. 31-42, e Residuais, pp. 5-18 |
| Quando parar de ajustar o treino? | MLP em 512 resenhas; dropout, L2 explícita e regra de parada por validação | Prince 9.1 e 9.3; slides Regularização, pp. 7-17 e 29-36 |
| Mais capacidade pode voltar a melhorar o teste? | Rede ReLU aleatória congelada, ajustando apenas a saída, em regressão sintética | Prince 8.4-8.5; slides Generalização, pp. 30-37; Belkin et al. |

Os números de páginas dos slides são posições no arquivo PDF, que podem
diferir da numeração impressa. Hashes, títulos e localizadores constam de
[`source_references.json`](../results/didactic/source_references.json).
Os PDFs das referências não são redistribuídos.

## Protocolo e papéis dos dados

O antigo treino limpo do IMDb fornece 6.144 resenhas de ajuste, 1.536 de
validação e 2.048 de auditoria interna, em partes balanceadas e sem
sobreposição textual. O teste oficial de 25.000 avaliações não é aberto
nesta extensão. A auditoria interna não é um novo benchmark externo.

Vocabulário TF-IDF, IDF, SVD de 128 componentes e padronização usam somente
as 6.144 resenhas de ajuste. Todas as redes densas recebem as mesmas
representações. Na regularização, só 512 rótulos são usados no treinamento
da MLP, mantendo essa representação prévia sem rótulos. Isso deve ser
mencionado ao apresentar o experimento com menos supervisão.

O arquivo [`didactic_config.json`](../didactic_config.json) fixa:

- Três sementes de inicialização: 17, 29 e 43.
- 27 treinos de otimizadores: três métodos, três taxas e três sementes.
- 15 treinos de arquitetura: cinco variantes e três sementes.
- Nove treinos de regularização: três variantes e três sementes.
- Orçamento de 32 épocas nos dois primeiros blocos e 120 no terceiro.
- Vinte sementes e 880 condições no experimento sintético de capacidade.

Taxas são escolhidas pela CE final média na validação. Arquiteturas e
regularizadores reutilizam a taxa Adam escolhida, sem retuning próprio.
A seleção e os checkpoints ficam congelados em
[`selection_frozen.json`](../results/didactic/selection_frozen.json).
Os dados da auditoria só foram transferidos ao ambiente de execução depois
desse congelamento. Todos os checkpoints planejados foram avaliados; a
auditoria não escolheu modelos para refit.

## O que os gráficos permitem dizer

**Otimizadores.** A taxa comum 0,003 produz trajetórias muito diferentes.
Após dar a cada método três candidatas e selecionar pela mesma regra, as
acurácias ficam próximas. A taxa faz parte da comparação. O momentum
PyTorch usa `v = 0.9*v + g`, com uma convenção de escala diferente da
média móvel normalizada de algumas derivações.

**Arquiteturas.** Mais camadas ajustam melhor o treino e podem generalizar
pior. As três redes profundas têm 54.146 parâmetros e partida idêntica.
O atalho com escala unitária isola a soma identidade; a variante com
`1/sqrt(11)` testa o controle de escala. Não há normalização adicional.
Neste ensaio o treino profundo chega a erro baixo, portanto a pior
validação não deve ser descrita como pura falha de otimização.

**Inicialização.** Um diagnóstico sintético sem treinamento mostra o
sinal e os gradientes em 32 camadas ReLU, com a escala He multiplicada por
0,1, 1 ou 3. A verificação confere a homogeneidade positiva da ReLU.
He fornece uma aproximação de estabilidade, não variância constante em
toda realização finita.

**Regularização.** A parada é escolhida por CE de validação, com paciência
15 e melhora mínima 0,0001. O treino continua para mostrar o que teria
acontecido depois; a linha de parada é contrafactual e não mede economia
real de tempo. L2 entra explicitamente no objetivo, excluindo vieses;
`weight_decay=0` no Adam. As curvas de CE usam modo avaliação e não
incluem a penalização. Menor CE e maior acurácia não são a mesma coisa,
e dropout pode subajustar neste orçamento.

**Double descent.** Aparece claramente no experimento sintético de
regressão: 64 amostras, saída com `p` coeficientes ajustados, ReLU oculta
congelada e ruído de desvio 0,3. As larguras compartilham as primeiras
colunas de características aleatórias. Os gráficos exibem mediana e IQR
das vinte sementes; valores extremos permanecem no CSV. A norma dos
coeficientes aumenta perto da interpolação e ridge atenua o pico.
Isso não estabelece double descent na CNN do IMDb, nem reproduz
integralmente o treinamento end-to-end mostrado no livro.

Faixas mín.-máx. de três sementes e IQR de vinte sementes são resumos
descritivos. Nenhuma delas é apresentada como intervalo de confiança
populacional.

## Uma correção detectada pela verificação

A primeira reconstrução dos checkpoints em CPU identificou uma diferença
de precisão: o treino desligava TF32, mas a avaliação inicial herdava a
configuração NVIDIA que o habilitava. A política foi igualada; os 51
treinos foram repetidos, preservando exatamente os arrays dos 60 conjuntos
de pesos e as mesmas escolhas de taxas. A auditoria foi reavaliada com a
política corrigida e novamente conferida por reconstrução em CPU.

Não houve alteração de arquiteturas, taxas candidatas ou sementes por causa
da auditoria. A evidência da correção está em
[`numerical_audit_correction.json`](../results/didactic/numerical_audit_correction.json).

## Como executar

Para ler o notebook com os artefatos incluídos, sem GPU e sem o corpus:

```bash
python -m pip install -e .
python verify_didactic.py
python execute_notebook.py --notebook notebooks/conexoes_disciplina.ipynb
```

Para refazer os experimentos, use uma cópia de trabalho sem a seleção
congelada desta execução. Prepare os dados pelo pipeline original e rode:

```bash
python run_didactic.py prepare
python run_didactic.py train --device cpu
python run_didactic.py toys
python run_didactic.py audit --device cpu
python verify_didactic.py
python run_didactic.py report
python tools/finalize_didactic_report.py
```

`--device cuda` usa uma GPU compatível, quando disponível. O treino
publicado foi feito em NVIDIA GB10 com precisão float32 e TF32 desligado;
os casos sintéticos usam float64 em CPU. A leitura e a reconstrução dos
artefatos salvos não exigem GPU. Metadados e verificadores descrevem o
ambiente real.

O catálogo de referências portável fica em `docs/REFERENCIAS_DISCIPLINA.json`.
O gerador o restaura nos resultados caso seja uma execução em diretório limpo.
O finalizador executa o notebook e atualiza os hashes dos entregáveis; a
inspeção visual do PDF deve ser refeita antes de distribuir uma nova construção.

As redes anteriores e os resultados Qwen permanecem preservados.
`python -m pytest -q` verifica também os novos invariantes.
[`verification.json`](../results/didactic/verification.json) registra a
conferência dos 51 treinos, 42 checkpoints de auditoria e 880 condições
sintéticas. O [mapa das figuras](../results/didactic/figures/) contém PNGs
e SVGs para usar na apresentação.
