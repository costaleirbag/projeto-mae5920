# Palavras ou BPE na mesma TextCNN

Este é um experimento didático sobre a representação da entrada. Acrescenta
**três treinos BPE** aos 87 treinos do estudo CNN/encoders e reutiliza os três
checkpoints da CNN por palavras com Adam 0,001. Não refaz as grades anteriores.
O código e os resultados antigos continuam congelados.

## Duas aprendizagens distintas

O tokenizer aprende **regras de união e vocabulário** usando apenas o texto dos
6.144 exemplos de treino, sem consultar seus rótulos. Depois fica congelado.
A CNN aprende **embeddings, filtros e saída** por gradiente usando os rótulos.
As duas representações exigem suas próprias CNNs treinadas do zero, porque
seus índices representam unidades diferentes.

Mantemos a normalização anterior: minúsculas, regex
`[a-z]+(?:'[a-z]+)?|\d+` e espaços entre as peças. No BPE, usamos
`WhitespaceSplit`, caracteres iniciais, uniões aprendidas, frequência mínima
2 e `</w>` no fim de cada palavra. O vocabulário final tem exatamente 20.000
tokens, incluindo `<pad>` 0 e `<unk>` 1. Não usamos BPE de bytes ou byte
fallback. Caracteres não vistos no treino ainda podem produzir UNK.

O exemplo `pallbearer` recebe UNK no vocabulário por palavras e
`p + all + bearer</w>` no BPE aprendido. O relatório e o notebook mostram a
aplicação passo a passo das prioridades reais de união. `</w>` é parte do
token final, não um token adicional. As peças não precisam ser morfemas.

## Controles e diferenças

| Propriedade | Nos dois modelos |
|---|---|
| Dados | Mesmos 6.144 de treino e 1.536 de validação |
| Texto | Mesmo prefixo limpo de até 128 palavras por espaço |
| Limite | 256 tokens, sem truncagem observada nos três papéis |
| Embedding | 20.000 × 128, aprendido do zero, PAD fixo em zero |
| Convoluções | Janelas 3/4/5, cem filtros cada, ReLU e máximo global |
| Saída | 300 máximos, dropout 0,5, Linear(300,2) |
| Parâmetros | 2.714.502 |
| Otimizador | Adam 0,001, batch 64, clipping de norma 1 |
| Orçamento | 12 épocas, três sementes: 17, 29 e 43 |
| Seleção | Menor CE de validação de cada semente, épocas 1 a 12 |
| Pareamento | Mesmas matrizes numéricas iniciais e ordem de minibatches |

A taxa veio da seleção anterior da CNN por palavras. Não a reajustamos para
BPE. Esta comparação responde o efeito da troca **nesta receita fixa**.
Cada índice de embedding passa a representar outra unidade. Uma janela de
três subpalavras pode cobrir menos palavras do que a janela de três palavras.
Essas mudanças fazem parte do tratamento, mesmo com a arquitetura igual.

## Uma nova amostra interna

Selecionamos 2.048 resenhas balanceadas do treino oficial limpo, excluindo
todos os papéis dos estudos didáticos MLP e CNN/encoders. Os IDs e hashes
ficaram fixados antes do treino. O arquivo de entrada ficou fora da execução
de treino até o congelamento dos seis checkpoints. A liberação valida o hash
da seleção e do arquivo e registra essa ordem.

É uma amostra nova **para os seis checkpoints desta comparação**. Outros
experimentos originais do projeto já usaram o treino oficial mais amplo.
Não reabrimos o teste oficial. Esta auditoria é diferente da auditoria dos
encoders: compare palavras e BPE aqui, sem misturar os números entre estudos.

## Resultados

| Representação | Acurácia média | Mínimo–máximo de três sementes | CE média | F1 macro médio |
|---|---:|---:|---:|---:|
| Palavras | 82,00% | 81,64–82,32% | 0,396 | 0,820 |
| BPE 20 mil | 81,41% | 81,20–81,54% | 0,405 | 0,814 |

Diferença média BPE menos palavras: **−0,59 ponto percentual**. As diferenças
pareadas são −0,44, −0,83 e −0,49 p.p. Nenhuma semente foi descartada.
Na nova auditoria, UNK cai de **3,27% para 0%**, e o comprimento médio sobe
de **121,3 para 129,1 tokens**. Logo, a cobertura melhorou sem melhorar a
acurácia nesta receita. Não concluímos inferioridade universal do BPE.

Faixas entre três sementes são descritivas, não intervalos de confiança.
Usamos uma divisão, pouco treino e taxa herdada. Não fizemos busca de tamanho
de vocabulário ou de janelas em número equivalente de palavras. Os tempos
brutos preservados incluem aquecimento e não fundamentam um ranking de
velocidade. Seis checkpoints reproduziram exatamente os logits em outro
processo; o verificador reconstrói 18 conjuntos de métricas.

## Conferir e reproduzir

Sem dados privados ou GPU, em uma cópia de trabalho:

```bash
python run_bpe.py verify --device cpu
python execute_notebook.py --notebook notebooks/cnn_encoders.ipynb
```

O tokenizer treinado, curvas, logits, métricas e escolhas estão em
[`results/bpe/`](../results/bpe/). A versão de Tokenizers usada para aprender
as regras foi 0.22.2; o treino neural usou 0.22.1 apenas para aplicar o JSON
congelado. O runtime está registrado. Na reprodução da seleção já publicada,
use exatamente o JSON de tokenizer distribuído. Para **refazer a aprendizagem
do tokenizer**, crie uma execução separada; não substitua os resultados
publicados. No teste de reprodução, outro ajuste com a mesma versão 0.22.2
alterou 104 IDs, com a primeira regra diferente na prioridade 3.419, apesar
de receber exatamente o mesmo texto. Frequências empatadas são uma possível
explicação, sem auditoria do desempate interno da biblioteca. Por isso o
experimento congela e distribui o JSON efetivamente utilizado.

Para repetir o treinamento, restaure os dados com o hash original e os três
checkpoints por palavras em `artifacts/sequences/`, a partir de sua execução
anterior. O [guia CNN/encoders](REPRODUCAO_SEQUENCIAS.md) descreve esse estudo.
Pesos treinados e corpus ficam fora do Git. O utilitário de workspace verifica
os três pesos antes de copiar, sem alterar os resultados publicados:

```bash
python tools/create_bpe_workspace.py --destination ../mae5920-bpe-repeticao
cd ../mae5920-bpe-repeticao
python run_bpe.py prepare --device cpu
```

Para reproduzir **as mesmas regras publicadas**, aplique a etapa abaixo antes
de qualquer treino, apontando para `results/bpe` no clone original. O utilitário
verifica código, todos os arquivos de entrada e a impressão digital do texto
de treino antes de reutilizar os quatro arquivos de preparação. Ele recusa
alterar uma execução já iniciada ou congelada. O tempo de ajuste preservado
é o do tokenizer original, e a reutilização ganha um registro próprio.

```bash
python tools/reuse_bpe_tokenizer.py --source ../projeto-mae5920/results/bpe
```

Se preferir estudar outra aprendizagem do BPE, pule essa etapa e preserve seus
novos resultados como outra execução, sem exigir os mesmos IDs ou métricas.

Mova `.cache/bpe_inputs/audit.parquet` para fora desse workspace e do ambiente
de treino. Transfira apenas treino/validação e os arquivos de protocolo. Depois:

```bash
python run_bpe.py train --device cuda
python tools/bpe_audit_input.py --source ../mae5920-bpe-audit.parquet
python run_bpe.py audit --device cuda
python run_bpe.py replay --device cuda
python run_bpe.py verify --device cpu
```

O treino recusa uma entrada de auditoria presente. A liberação recusa ocorrer
sem o congelamento. A seleção anterior por palavras é uma referência fixa,
com seus pesos e metadados validados por hash. Outro runtime pode alterar
resultados; os testes de reprodução não prometem igualdade entre plataformas.

## Fontes

- [Sennrich, Haddow e Birch (2016), Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/).
- [Hugging Face Tokenizers, Quicktour](https://huggingface.co/docs/tokenizers/quicktour).
- [Kim (2014), Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/).

O artigo de subpalavras trata tradução, e não é evidência de ganho garantido
na nossa classificação. O tutorial é referência de API; nossa divisão para
aprender o tokenizer usa exclusivamente treino.
