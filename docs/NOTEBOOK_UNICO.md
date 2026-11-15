# Notebook de entrega do seminário

[Baixar o notebook](../notebooks/seminario_completo.ipynb) · [Relatório de oito páginas](../output/pdf/relatorio_seminario.pdf) · [Slides](https://costaleirbag.github.io/projeto-mae5920/)

O sumário associa as oito páginas do relatório às seções do notebook e aos
slides. O corpo cobre divisões, tokenização, arquitetura, treino, teste oficial,
diagnósticos, contraste de frases e avaliação fora do IMDb. O trio de gerações
do Qwen aparece no apoio, como na apresentação.

## Conferir os resultados

Baixe o arquivo e abra no Jupyter ou VS Code. Deixe o próprio arquivo na pasta
de execução do kernel ou em sua subpasta notebooks/. Use Python 3.11 ou mais
recente e instale as dependências do caminho em CPU:

~~~bash
python -m pip install jupyterlab numpy pandas scipy scikit-learn matplotlib torch 'tokenizers>=0.22,<0.23'
~~~

Rode todas as células com MODO = "conferir". O código recalcula métricas a partir
dos escores anexados, reproduz os exemplos e mostra o laço de treino real. Uma
passagem não treinada da CNN confere tensores, convolução e máscara de padding.
O arquivo inclui o código de produção, as configurações e os dados necessários
para esse percurso. Sua cópia pode ser aberta longe do repositório e sem rede.

## Reexecutar os experimentos

No bloco final do notebook, selecione MODO = "reproduzir" e escolha uma pasta de
trabalho. Siga os requisitos de bibliotecas e de CUDA/BF16 descritos no próprio
bloco. Execute desde o início. O código baixa a revisão fixada do IMDb, confere
as divisões e baixa os pesos-base públicos quando cada etapa precisa deles.

- O estudo original ajusta TF-IDF, CNN e BERT. O estudo de encoders refaz a grade
  sequencial com a auditoria separada. A comparação por palavras e BPE treina
  os seis checkpoints e só então avalia a auditoria.
- A extensão de Qwen repete a classificação. As etapas de diagnóstico reavaliam
  recorte, controles TF-IDF, exemplos sintéticos e fora do IMDb. O apoio refaz a
  direção e gera o trio de Signal 2040 com os parâmetros já escolhidos.
- As etapas escrevem novos resultados, comparações e logs na pasta escolhida.
  As referências publicadas ficam em sua subpasta referencia/. Para recomeçar
  do zero, use outra pasta de trabalho.

O BPE usa o JSON de tokenizer congelado da comparação principal. Aprender
novamente as fusões pode desempatar pares de outra forma. Métricas do retreino,
tokens gerados e tempos também dependem das versões e do hardware. O notebook
compara a nova saída com a registrada. Em uma pasta vazia, o download e o
preparo recriaram as quatro divisões. O TF-IDF reproduziu 90,144% de acurácia
e coincidiu nas métricas publicadas a seis casas decimais. A CNN reajustada
chegou a 86,988%, ante os 86,784% registrados. A conferência em CPU foi
verificada. A sequência completa em CUDA é uma execução separada.

## Reconstruir o arquivo publicado

~~~bash
python tools/build_seminar_notebook.py
python execute_notebook.py --notebook notebooks/seminario_completo.ipynb
python verify_seminar_notebook.py
python tools/check_seminar_portability.py
~~~

O builder usa pandas e PyArrow para projetar as previsões originais por ID
antes de anexá-las. O teste de portabilidade copia só o notebook para uma
pasta vazia, desativa CUDA, bloqueia as conexões Python no kernel e confere a
extração do código. Com --corpus apontando para os Parquets do IMDb preparado,
ele também repete o ajuste completo do TF-IDF.
