# Reproduzir a extensão CNN e encoders

## Ler e conferir sem GPU

O [notebook executado](../notebooks/cnn_encoders.ipynb) usa logits e resultados
salvos. Ele não baixa modelos, não treina e não exige os checkpoints grandes.

```bash
python verify_sequences.py
python execute_notebook.py --notebook notebooks/cnn_encoders.ipynb
```

O primeiro comando reconstrói métricas, escolha das taxas, épocas e controles
de inicialização. A reavaliação dos pesos na GPU está registrada separadamente
em `results/sequences/replay_verification.json`. Reexecutar notebooks pode
alterar seus hashes; faça isso numa cópia de trabalho.

## Treinar outra execução

Os resultados publicados são congelados. O programa recusa sobrescrevê-los.
Para repetir os treinos, crie uma pasta de execução separada. É necessário
restaurar `data/train.parquet` da preparação original do projeto, com o hash
registrado no manifesto de dados. O corpus não é redistribuído no Git.

```bash
python tools/create_sequence_workspace.py --destination ../mae5920-nova-execucao
cd ../mae5920-nova-execucao
python run_sequences.py prepare --device cpu
python tools/sequence_audit_input.py withhold --file ../mae5920-audit-input.parquet
python tools/fetch_sequence_models.py
python run_sequences.py pilot --pretrained .cache/sequence_pretrained --device cuda
python run_sequences.py train --pretrained .cache/sequence_pretrained --device cuda
```

O piloto verifica compatibilidade e tempo em 32 exemplos de treino, sem
escolhas de validação. O treinamento carrega somente treino e validação.
Se houver interrupção, o estágio pode ser retomado com código e entradas
idênticos; execuções concluídas são verificadas antes de serem reutilizadas.

Na execução original, o arquivo de auditoria ficou fora do ambiente de treino
até a criação de `selection_frozen.json`. Para reproduzir essa separação física
em outro ambiente, transfira primeiro apenas os dois arquivos de treino e
validação de `.cache/sequence_inputs/`; transfira `audit.parquet` depois do
congelamento. O manifesto inicial contém IDs e hashes da auditoria, mas não
seus rótulos individuais.

Depois de congelar os 87 treinos:

```bash
python tools/sequence_audit_input.py restore --file ../mae5920-audit-input.parquet
python run_sequences.py audit --pretrained .cache/sequence_pretrained --device cuda
python replay_sequences.py --pretrained .cache/sequence_pretrained --device cuda
python analyze_sequences.py --pretrained .cache/sequence_pretrained --device cuda
python verify_sequences.py --checkpoints
python run_sequences.py report --device cpu
python execute_notebook.py --notebook notebooks/cnn_encoders.ipynb
```

O utilitário `restore` verifica o congelamento e os hashes e gera o registro
de transferência usado pelo verificador antes da primeira auditoria.

O replay mantém a primeira auditoria e executa uma segunda avaliação de todos
os 43 checkpoints em pasta separada. A análise produz exemplos sintéticos e
uma decomposição exata da leitura linear da CNN. Nenhuma dessas etapas muda
taxas ou checkpoints.

## Ambiente e pesos

As versões efetivamente usadas estão em `results/sequences/runtime.json` e
`protocol_started.json`. O estudo usa PyTorch com CUDA, Transformers 4.57.1,
BF16 para os encoders e FP32 para CNN/cabeças. A implementação desativa TF32
em treino e auditoria e solicita algoritmos determinísticos. Resultados em
outro hardware ou biblioteca podem diferir.

Os pesos originais vêm de revisões fixas de BERT-base, DistilBERT e ModernBERT.
O utilitário de download usa repositórios públicos e verifica cada hash.
Não usamos pesos já ajustados no IMDb. Os checkpoints treinados ficam em
`artifacts/sequences/`, fora do Git, com hashes no congelamento da seleção.
As predições, métricas, curvas, tokenizadores e decisões ficam no repositório.

O [protocolo científico](PROTOCOLO_SEQUENCIAS.md) descreve as grades e os
limites das comparações. O suplemento MLP, o teste oficial dos modelos
originais e o steering do Qwen são estudos distintos.

## Extensão BPE

O relatório e o notebook atuais acrescentam três treinos BPE e outra amostra
interna para comparar as duas tokenizações. O [protocolo BPE](PROTOCOLO_BPE.md)
descreve os controles e sua reprodução separada. `verify_sequences.py` continua
conferindo os 87 treinos congelados; `run_bpe.py verify --device cpu` confere
o novo protocolo. O total da extensão sequencial passa a 90 treinos.
