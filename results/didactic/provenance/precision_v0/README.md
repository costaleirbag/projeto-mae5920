# Auditoria inicial substituída por correção numérica

Estes arquivos preservam a primeira avaliação, que usava a configuração
NVIDIA padrão com TF32 habilitado. O treino havia desligado TF32. A
reconstrução dos checkpoints em CPU detectou a divergência.

Os resultados atuais ficam dois níveis acima, em `results/didactic/`.
Não use os números desta pasta como os resultados finais. Ela é evidência
histórica, não uma variante adicional selecionada por desempenho.

`didactic_precision_v0.py` é a cópia exata do código inicial; seu hash
corresponde ao registrado no congelamento inicial. As configurações e
sementes não foram alteradas na correção. Os 60 conjuntos de parâmetros
da repetição são bit a bit iguais aos primeiros; seus fingerprints por
tensor estão em `../repeated_parameter_fingerprints.json` e são conferidos
pelo verificador atual. Os arquivos NPZ históricos dos pesos foram
substituídos na distribuição pelos equivalentes atuais com os mesmos
arrays; os hashes ZIP podem diferir pela data de serialização.
