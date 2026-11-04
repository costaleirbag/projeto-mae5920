# Um arquivo para percorrer o seminário

Abra o **[notebook completo](../notebooks/seminario_completo.ipynb)**. Ele é o ponto
de entrada para Nestor, Letícia e Gabriel estudarem juntos. O GitHub mostra a versão
já executada. Para executar localmente, baixe esse `.ipynb` e abra no Jupyter ou
no VS Code com as dependências descritas no início.

O arquivo contém as evidências de leitura em um anexo interno. Não exige clonar o
repositório, carregar checkpoints ou baixar os modelos grandes. Se o ambiente
abre o notebook sem salvá-lo no sistema de arquivos do runtime, copie também o
próprio `.ipynb` para a pasta de execução, pois o código lê seu anexo.

Para navegar, use o sumário lateral do Jupyter/VS Code ou a busca do navegador pelo
número e título da seção. O roteiro da capa é textual: o visualizador do GitHub
redireciona atalhos internos de notebooks para a pasta e não preserva a navegação.

## Como está organizado

1. Problema, rótulos e papéis dos dados, incluindo as auditorias diferentes.
2. Limpeza, regex, vocabulário, IDs, padding e baseline TF-IDF.
3. Classe TextCNN real, diagrama, formas dos tensores e conta de parâmetros.
4. Entropia cruzada, backward, clipping, Adam e o laço real de treinamento.
5. Grades, sementes, curvas, seleção por validação, arquitetura e dropout.
6. BPE aprendido só no treino, regras, aplicação passo a passo e comparação real.
7. Tokenização dos encoders, pooling, wrapper e congelamento versus ajuste completo.
8. Métricas, exemplos comuns e decomposição da leitura final dos filtros.
9. MLPs, resíduos, inicialização, geometria e regularização.
10. Double descent na regressão sintética, com reconstrução de um interpolador.
11. Qwen congelado, classificação por rótulos, direção por médias e steering.
12. Estudo original, calibração, análises descritivas e limites.
13. Testes e reconstrução das 49 avaliações de auditoria incorporadas.
14. Função opcional para repetir seis CNNs com o corpus original restaurado.
15. Referências ao livro e aos slides, roteiro curto e perguntas para o grupo.

As classes e funções centrais são extraídas da implementação original pelo builder.
O verificador compara sua árvore sintática com a fonte. Assim, o código explicado
corresponde à arquitetura que produziu os resultados.

## O que roda ao usar Run All

O caminho padrão executa 45 células em CPU. Há exemplos pequenos com frases próprias,
um passo de otimização, um treino de CNN de demonstração, testes de gradientes,
pooling e hooks, além de reconstruções de resultados numéricos reais. Esses exercícios
são identificados e não são adicionados ao benchmark de 90 treinos sequenciais.

O notebook refaz a direção escolhida do Qwen a partir dos 256 estados reais de treino,
sem carregar o Qwen. Também contém os 64 trios de geração já salvos. A primeira fonte
sintética e um caso com falha factual aparecem no roteiro; uma função permite explorar
os demais. O juiz automático e as limitações ficam explícitos.

O treino completo está desligado por padrão. Para repeti-lo, restaure o corpus com
o hash original, escolha uma saída nova e altere a flag da seção 14. O runner mostra
a sequência completa de preparo, seis ajustes, seleção, salvamento e auditoria.
A reprodução operacional com separação física da auditoria está nos
[guias de CNN/encoders](REPRODUCAO_SEQUENCIAS.md) e [BPE](PROTOCOLO_BPE.md).

O caminho padrão foi executado com apenas o notebook numa pasta temporária fora do
repositório, GPU oculta e conexões de sockets Python bloqueadas. O runner opcional
também foi exercitado com os dados completos, uma semente, uma época por representação
e a arquitetura original em CPU. Esse teste verifica o fluxo e não é um novo resultado
do seminário. Os certificados ficam em `results/seminar/`.

## Conferir e gerar outra cópia

```bash
python verify_seminar_notebook.py
python execute_notebook.py --notebook notebooks/seminario_completo.ipynb
```

Reexecutar muda tempos e metadados, alterando o hash do arquivo. Faça isso numa cópia
de trabalho. Para reconstruir o notebook a partir das evidências congeladas:

```bash
python tools/build_seminar_notebook.py
python execute_notebook.py --notebook notebooks/seminario_completo.ipynb
python verify_seminar_notebook.py
python tools/check_seminar_portability.py
```

O builder precisa do repositório, mas o notebook produzido é autocontido. Ele incorpora
206 arquivos de evidências e derivações compactas, sem o corpus preparado nem pesos
treinados. Os manifestos compactos mantêm os IDs dos papéis e a proveniência dos arquivos
originais; omitem as listas volumosas de hashes de cada texto. Nenhum resultado original
é reestimado ou escolhido pelo builder. Os quatro notebooks específicos continuam
disponíveis para consultas complementares.
