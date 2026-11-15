"""Células do notebook que reproduzem as tabelas e exemplos do seminário."""

REPORT = ("https://github.com/costaleirbag/projeto-mae5920/blob/main/"
          "output/pdf/relatorio_seminario.pdf")
EVIDENCE = [
    "data/manifest.json",
    "results/comparison.csv",
    "results/tfidf/metrics.json",
    "results/cnn/metrics.json",
    "results/bert/metrics.json",
    "results/qwen/classification/metrics.json",
    "results/tfidf/metadata.json",
    "results/cnn/metadata.json",
    "results/bert/metadata.json",
    "results/sequences/summary.json",
    "results/tfidf_ablation/summary.json",
    "results/tfidf_regularization/summary.json",
    "results/tfidf_regularization/selection.json",
    "seminario/dados/tfidf_slide.json",
    "seminario/dados/comprimentos.json",
    "seminario/dados/discordancias.json",
    "seminario/dados/recorte_teste/resumo.json",
    "seminario/dados/recorte_teste/predicoes.npz",
    "seminario/dados/recorte_bert/resumo.json",
    "seminario/dados/recorte_bert/predicoes.npz",
    "seminario/dados/pares_resultados.json",
    "seminario/dados/animacoes.json",
    "seminario/dados/passo_resultados.json",
    "seminario/dados/passo_predicoes.npz",
    "seminario/dados/fora_dominio_resultados.json",
    "seminario/dados/fora_dominio_predicoes.npz",
    "seminario/dados/tfidf_youtube_resultados.json",
    "seminario/dados/qwen_youtube_resultados.json",
]


def datasets(md, code):
    md(f"""
    <a id="dados"></a>
    ## Dados e protocolo de avaliação

    [Relatório, p. 2]({REPORT}#page=2) · Slide 3. Os ensaios internos usam
    amostras do treino oficial já limpo. O teste oficial continua separado.
    """, "dados")
    code("""
    imdb = read_json("data/manifest.json")
    papeis_oficiais = {"train": "Treino", "validation": "Validação",
                       "calibration": "Calibração", "test": "Teste oficial"}
    linhas_oficiais = []
    for papel, nome in papeis_oficiais.items():
        dados = imdb["splits"][papel]
        linhas_oficiais.append({"papel": nome, "total": dados["n"],
                                "positivas": dados["label_counts"]["1"],
                                "negativas": dados["label_counts"]["0"]})
    classes = pd.DataFrame(linhas_oficiais).set_index("papel")
    classes["positivas (%)"] = 100 * classes["positivas"] / classes["total"]
    conferir("Contagens oficiais somam o total", classes["total"].equals(
        classes["positivas"] + classes["negativas"]))
    display(classes.round(2))

    audit = imdb["audit"]
    removidas = {descricao: audit[chave] for descricao, chave in [
        ("Textos de treino iguais a textos de teste", "train_rows_matching_test_text"),
        ("Linhas de treino com rótulos conflitantes", "conflicting_train_rows_excluded"),
        ("Duplicatas no treino", "duplicate_train_rows_excluded"),
    ]}
    print("Exclusões do treino oficial", removidas)
    conferir("Treino limpo dividido em treino, validação e calibração",
             classes.iloc[:3]["total"].sum() + sum(removidas.values()) == audit["official_train_n"])

    internos = {nome: read_json(f"results/{nome}/data_manifest.json")
                for nome in ("sequences", "bpe")}
    for nome, manifesto in internos.items():
        papeis = manifesto["roles"]
        ids_por_papel = [set(papel["ids"]) for papel in papeis.values()]
        conferir(f"Sem sobreposição interna de IDs · {nome}",
                 len(set.union(*ids_por_papel)) == sum(map(len, ids_por_papel)))
        conferir(f"Classes internas balanceadas · {nome}", all(
            papel["label_counts"]["0"] == papel["label_counts"]["1"] for papel in papeis.values()))
    conferir("BPE e encoders têm as mesmas amostras de ajuste",
             all(internos["sequences"]["roles"][papel]["ids"] == internos["bpe"]["roles"][papel]["ids"]
                 for papel in ("train", "validation")))
    exibir_papeis = {"train": "Treino", "validation": "Validação", "audit": "Auditoria"}
    display(pd.DataFrame([
        {"ensaio": nome, "papel": exibir_papeis[papel], "total": dados["n"],
         "positivas": dados["label_counts"]["1"], "negativas": dados["label_counts"]["0"]}
        for nome, manifesto in internos.items() for papel, dados in manifesto["roles"].items()
    ]))
    """, "divisoes-balanceamento")
    code("""
    comprimentos = read_json("seminario/dados/comprimentos.json")
    estatisticas = comprimentos["lengths"]
    display(pd.DataFrame({
        "Resenha no treino": estatisticas["train"]["review_words"],
        "Resenha no teste": estatisticas["test"]["review_words"],
        "Frase no teste": estatisticas["test"]["sentence_words"],
    }).T[["n", "min", "p10", "mean", "median", "p90", "max"]].round(1).rename(
        columns={"mean": "média", "median": "mediana"}))
    print(comprimentos["units"]["sentences"])
    """, "comprimentos-resenhas")
    md("""
    O corpus preparado guarda **id**, **label** (0 negativa, 1 positiva),
    **clean_text**, **word_count** e **text_sha256**. O texto limpo é a entrada
    dos modelos. O hash serve à checagem de duplicatas.
    """)


def official(md, code):
    md(f"""
    <a id="oficial"></a>
    ## TF-IDF e teste oficial

    [Relatório, p. 5]({REPORT}#page=5) · Slides 7–9. As parcelas do exemplo
    usam o vocabulário, o IDF e os coeficientes ajustados no estudo original.
    """, "oficial")
    code(r"""
    from collections import Counter
    import re
    from scipy.special import expit

    exemplo_tfidf = read_json("seminario/dados/tfidf_slide.json")
    palavras_exemplo = re.findall(r"(?u)\b\w\w+\b", exemplo_tfidf["text"].lower())
    contagens = Counter(palavras_exemplo + [
        f"{a} {b}" for a, b in zip(palavras_exemplo, palavras_exemplo[1:])
    ])
    termos = pd.DataFrame(exemplo_tfidf["terms"]).set_index("term")
    conferir("Unigramas e bigramas do exemplo", contagens == termos["count"].to_dict())
    tf = 1 + np.log(termos["count"].to_numpy())
    idf = 1 + np.log((1 + exemplo_tfidf["train_documents"]) /
                     (1 + termos["document_frequency"].to_numpy()))
    sem_normalizar = tf * idf
    vetor = sem_normalizar / np.linalg.norm(sem_normalizar)
    parcelas = vetor * termos["coefficient"].to_numpy()
    margem = exemplo_tfidf["bias"] + parcelas.sum()
    conferir("TF-IDF e escore do slide", np.allclose(vetor, termos["tfidf"])
             and np.isclose(margem, exemplo_tfidf["logit"])
             and np.isclose(expit(margem), exemplo_tfidf["positive_raw"]))
    print(exemplo_tfidf["text"])
    display(termos.assign(TF=tf, IDF=idf, parcela=parcelas)[
        ["count", "document_frequency", "TF", "IDF", "tfidf", "coefficient", "parcela"]].round(4))
    print(f"Margem com viés = {margem:.4f} · P(positiva) = {expit(margem):.2%}")
    """, "tfidf-calculo-real")
    code("""
    from sklearn.metrics import accuracy_score, f1_score

    receitas = read_json("results/tfidf_ablation/summary.json")["recipes"]
    recortes = {r["scope"]: r for r in receitas if r["ngrams"] == "unigramas_bigramas"}
    integral = recortes["integral"]
    candidato = next(r for r in integral["candidates"] if r["C"] == integral["selected_C"])
    controle = integral["shuffled_labels"]
    display(pd.DataFrame({
        "Rótulos reais": [candidato["train"]["accuracy"], candidato["validation"]["accuracy"]],
        "Rótulos de treino embaralhados": [controle["train_against_shuffled_labels"]["accuracy"],
                                           controle["validation_against_real_labels"]["accuracy"]],
    }, index=["Treino, rótulos usados no ajuste", "Validação, rótulos reais"]).mul(100).round(2))

    with read_npz("derived/tfidf_ablation.npz") as ablation:
        linhas_recorte = []
        for entrada in ("integral", "400"):
            receita = recortes[entrada]
            margem = ablation[f"test_{entrada}_unigramas_bigramas"]
            acuracia = accuracy_score(ablation["test_labels"], margem >= 0)
            conferir(f"Acurácia da nova receita TF-IDF · {entrada}",
                     np.isclose(acuracia, receita["test"]["accuracy"]))
            linhas_recorte.append({"entrada": entrada, "C": receita["selected_C"],
                                    "acurácia (%)": 100 * acuracia})
    display(pd.DataFrame(linhas_recorte).round(2))

    selecao_tfidf = read_json("results/tfidf_regularization/selection.json")
    vencedor = f"{selecao_tfidf['winner']['variant']}_c{selecao_tfidf['winner']['C']:g}"
    with read_npz("derived/tfidf_regularization.npz") as arrays:
        acc_vencedor = accuracy_score(arrays["test_labels"], arrays[f"test_{vencedor}"] >= 0)
    conferencia = read_json("results/tfidf_regularization/summary.json")
    conferir("Melhor C selecionado na validação", vencedor == conferencia["selected_by_validation"]
             and np.isclose(acc_vencedor, conferencia["test"][vencedor]["accuracy"]))
    print(f"Grade ampliada · seleção pela validação (C={selecao_tfidf['winner']['C']:g})"
          f" · teste {acc_vencedor:.2%}")
    """, "tfidf-controles")
    md("""
    O corte em 400 ocorre **antes** de ajustar o vetor TF-IDF e a regressão.
    Esse controle refaz dois ajustes. O teste oficial abaixo usa as receitas originais.
    """)
    code("""
    modelos_oficiais = {"tfidf": "TF-IDF + logística", "cnn": "TextCNN", "bert": "BERT"}
    comparacao = read_csv("results/comparison.csv")
    brutas = comparacao.loc[
        comparacao["probabilities"].eq("raw")
        & comparacao["split"].isin(["train", "validation", "test"]),
        ["split", "model", "accuracy", "f1_macro", "log_loss"]
    ].copy()
    brutas["split"] = pd.Categorical(brutas["split"], ["train", "validation", "test"], ordered=True)
    brutas["model"] = pd.Categorical(brutas["model"], list(modelos_oficiais), ordered=True)
    brutas = brutas.sort_values(["split", "model"])
    brutas["model"] = brutas["model"].map(modelos_oficiais)
    brutas["split"] = brutas["split"].map({"train": "Treino", "validation": "Validação", "test": "Teste"})
    display(brutas.rename(columns={
        "split": "divisão", "model": "modelo", "accuracy": "acurácia", "f1_macro": "F1 macro",
        "log_loss": "CE bruta",
    }).set_index(["divisão", "modelo"]).round(4))
    """, "metricas-originais")
    code("""
    discordancias = read_json("seminario/dados/discordancias.json")
    registros = []
    for caso in discordancias["casos"]:
        for modelo, chave in [("TextCNN", "cnn"), ("TF-IDF", "tfidf")]:
            for entrada in ("prefixo", "inteiro"):
                medida = caso.get(f"{chave}_{entrada}")
                if medida is None:
                    continue
                if entrada == "prefixo" and caso["tokens_cnn"] <= discordancias["limite_cnn_tokens"]:
                    entrada = "integral (cabe no limite)"
                registros.append({"resenha": caso["id"], "tokens": caso["tokens_cnn"],
                                  "rótulo": caso["rotulo"], "modelo": modelo, "entrada": entrada,
                                  "P(positiva) calibrada": medida["p_positiva"]})
    display(pd.DataFrame(registros).round(4))
    longo = next(caso for caso in discordancias["casos"] if caso["id"] == "imdb_test_03531")
    curto = next(caso for caso in discordancias["casos"] if caso["id"] == "imdb_test_22831")
    for nome, trecho in longo["trechos"].items():
        print(f"03531 · {nome} · token {trecho['primeiro_token']}: {trecho['texto']}")
    destaques = [
        ("03531 · CNN integral", longo["cnn_inteiro"]["janelas"], ("hate to waste",)),
        ("22831 · CNN", curto["cnn_prefixo"]["janelas"], ("part is brilliant", "the ending doesn't make much")),
        ("22831 · TF-IDF", curto["tfidf_inteiro"]["termos"], ("brilliant", "not one")),
    ]
    termos = []
    for caso, sinais, pedidos in destaques:
        for grupo in sinais.values():
            termos.extend({"caso": caso, **item} for item in grupo if item["termo"] in pedidos)
    display(pd.DataFrame(termos).round(4))
    """, "resenhas-com-recorte")


def diagnostics(md, code):
    md(f"""
    <a id="diagnosticos"></a>
    ## Erros, recorte, calibração e custo

    [Relatório, pp. 5–6]({REPORT}#page=6) · Slides 9–10, 14 e apoios 16–17.
    As métricas abaixo são recalculadas sobre previsões e rótulos alinhados por ID.
    """, "diagnosticos")
    code("""
    from scipy.special import expit
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

    def metricas_prob(rotulos, probabilidades):
        p = np.clip(np.asarray(probabilidades, dtype=float), 1e-7, 1 - 1e-7)
        classes = p >= .5
        _, fp, fn, _ = confusion_matrix(rotulos, classes, labels=[0, 1]).ravel()
        return {"accuracy": accuracy_score(rotulos, classes),
                "f1_macro": f1_score(rotulos, classes, average="macro", labels=[0, 1], zero_division=0),
                "log_loss": log_loss(rotulos, p, labels=[0, 1]), "fp": fp, "fn": fn}

    nomes = {"tfidf": "TF-IDF", "cnn": "TextCNN", "bert": "BERT", "qwen": "Qwen (extensão)"}
    testes = []
    with read_npz("derived/oficial.npz") as pacote_teste:
        ids_teste = pd.Index(pacote_teste["ids"])
        rotulos_teste = pacote_teste["labels"]
        palavras_teste = pacote_teste["word_counts"]
        escores_teste = {modelo: pacote_teste[f"{modelo}_calibrated"] for modelo in nomes}
        for modelo, nome in nomes.items():
            fonte = "qwen/classification" if modelo == "qwen" else modelo
            esperado = read_json(f"results/{fonte}/metrics.json")["test"]
            bruto = metricas_prob(rotulos_teste, expit(pacote_teste[f"{modelo}_logits"]))
            calibrado = metricas_prob(rotulos_teste, escores_teste[modelo])
            for modo, medido in (("raw", bruto), ("calibrated", calibrado)):
                conferir(f"Métricas oficiais · {nome} · {modo}", all(
                    np.isclose(valor, esperado[modo][metrica], atol=1e-7)
                    for metrica, valor in medido.items()))
            conferir(f"Classes preservadas na calibração · {nome}",
                     np.array_equal(expit(pacote_teste[f"{modelo}_logits"]) >= .5,
                                    escores_teste[modelo] >= .5))
            testes.append({"modelo": nome, "acurácia": bruto["accuracy"], "F1 macro": bruto["f1_macro"],
                           "FP": bruto["fp"], "FN": bruto["fn"], "CE bruta": bruto["log_loss"],
                           "CE calibrada": calibrado["log_loss"]})
    display(pd.DataFrame(testes).set_index("modelo").round(4))
    acertos = {modelo: (escore >= .5) == rotulos_teste for modelo, escore in escores_teste.items()}
    diferentes = {"Só TF-IDF acertou": int((acertos["tfidf"] & ~acertos["cnn"]).sum()),
                  "Só TextCNN acertou": int((acertos["cnn"] & ~acertos["tfidf"]).sum()),
                  "TF-IDF acertou e Qwen errou": int((acertos["tfidf"] & ~acertos["qwen"]).sum())}
    print(diferentes)
    """, "teste-oficial-verificado")
    code("""
    recorte_cnn = read_json("seminario/dados/recorte_teste/resumo.json")
    with read_npz("seminario/dados/recorte_teste/predicoes.npz") as par_cnn:
        posicoes = ids_teste.get_indexer(par_cnn["ids"])
        conferir("IDs do recorte alinhados ao teste", all(posicoes >= 0)
                 and np.array_equal(rotulos_teste[posicoes], par_cnn["labels"]))
        tau = recorte_cnn["temperatura_original"]
        prefixo = expit(par_cnn["prefixo"].astype(float) / tau)
        conferir("Reexecução dos prefixos da TextCNN", np.allclose(
            prefixo, escores_teste["cnn"][posicoes], atol=1e-6))
        inteira = escores_teste["cnn"].copy()
        inteira[posicoes] = expit(par_cnn["inteiro"].astype(float) / tau)
    escores_teste["cnn_integral"] = inteira

    tabela_faixas = []
    for faixa in read_json("seminario/dados/comprimentos.json")["bands"]:
        inicio, fim = faixa["min_words"], faixa["max_words"]
        titulo = "Até 100" if inicio == 0 else f"{inicio}–{fim}" if fim is not None else "Acima de 500"
        selecao = palavras_teste >= inicio
        if fim is not None:
            selecao &= palavras_teste <= fim
        linha = {"faixa (palavras)": titulo, "n": int(selecao.sum())}
        conferir(f"Quantidade de resenhas · {titulo}", linha["n"] == faixa["n"])
        for modelo, p in escores_teste.items():
            m = metricas_prob(rotulos_teste[selecao], p[selecao])
            conferir(f"Métricas por comprimento · {modelo} · {titulo}", all(
                np.isclose(m[k], v, atol=1e-7) for k, v in faixa["metrics"][modelo].items()))
            nome = nomes.get(modelo, "TextCNN integral")
            linha[nome] = f'{100 * m["accuracy"]:.2f} / {100 * m["f1_macro"]:.2f}'
        tabela_faixas.append(linha)
    print("Acurácia / F1 macro em %")
    display(pd.DataFrame(tabela_faixas).set_index("faixa (palavras)"))

    controle_bert = read_json("seminario/dados/recorte_bert/resumo.json")
    linhas_bert = []
    with read_npz("seminario/dados/recorte_bert/predicoes.npz") as par_bert:
        for coluna, teto in zip(("replay", "extended"), controle_bert["caps_including_specials"]):
            p = expit(par_bert[coluna].astype(float) / controle_bert["temperature"])
            medido = metricas_prob(par_bert["labels"], p)
            conferir(f"Mesmo BERT no grupo longo · teto {teto}", all(
                np.isclose(medido[k], controle_bert[coluna][k], atol=1e-7)
                for k in ("accuracy", "f1_macro", "log_loss")))
            linhas_bert.append({"limite de tokens": teto, "n": len(p), "acurácia": medido["accuracy"],
                                "F1 macro": medido["f1_macro"], "CE": medido["log_loss"]})
    display(pd.DataFrame(linhas_bert).round(4))
    """, "metricas-comprimento-recorte")
    md("""
    Os tempos são registros das execuções originais. A tabela de custo usa entradas
    já representadas na inferência. O contraste dentro do BERT abaixo usa a
    auditoria interna de 2.048 prefixos, com três sementes por receita.
    """)
    code("""
    custo = []
    for modelo, dispositivo in [("tfidf", "M1 CPU"), ("cnn", "M1 GPU"), ("bert", "GB10 GPU")]:
        registro = read_json(f"results/{modelo}/metadata.json")
        custo.append({"modelo": nomes[modelo], "dispositivo": dispositivo,
                      "épocas": len(registro.get("history", [])) or "—",
                      "ajuste (s)": registro["training_seconds"],
                      "inferência (s)": registro["all_splits_inference_seconds"]})
    display(pd.DataFrame(custo).set_index("modelo").round(2))
    encoders = read_json("results/sequences/summary.json")["groups"]
    display(pd.DataFrame([
        {"receita": grupo["label"], "acurácia (%)": 100 * grupo["accuracy_mean"],
         "ajuste com extração, se aplicável (s)": grupo["fit_plus_features_seconds"]}
        for grupo in encoders if grupo["key"] in {"bert_probe", "bert_full"}
    ]).set_index("receita").round(2))
    """, "custo-computacional")


def failures(md, code):
    md(f"""
    <a id="falhas"></a>
    ## Contraste e janelas da TextCNN

    [Relatório, p. 7]({REPORT}#page=7) · Slides 2, 12 e apoio 19.
    Recalculamos a sigmoide dos escores e acompanhamos o filtro 89 da CNN por palavras.
    """, "falhas")
    code(r"""
    from collections import Counter
    import re
    from scipy.special import expit

    pares = read_json("seminario/dados/pares_resultados.json")["pares"]
    par = next(par for par in pares if par["id"] == "contraste")
    contraste_tfidf = read_json("seminario/dados/discordancias.json")["contraste_tfidf"]
    print("A ·", par["a"])
    print("B ·", par["b"])
    print("Leitura proposta (A/B)", par["leitura"])
    fichas = [{"modelo": "TF-IDF", "semente": "—",
               "P(+) em A": expit(contraste_tfidf["a"]["logit"]),
               "P(+) em B": expit(contraste_tfidf["b"]["logit"])}]
    for tokenizador, resultado in par["resultados"].items():
        for rodada in resultado["predicoes"]:
            a, b = rodada["a"], rodada["b"]
            margem_a, margem_b = (r["logits"][1] - r["logits"][0] for r in (a, b))
            conferir(f"Probabilidades do par · {tokenizador} · {rodada['semente']}",
                     np.allclose(expit([margem_a, margem_b]), [a["p_positiva"], b["p_positiva"]]))
            fichas.append({"modelo": f"TextCNN · {tokenizador}", "semente": rodada["semente"],
                           "P(+) em A": expit(margem_a), "P(+) em B": expit(margem_b)})
    padrao = r"(?u)\b\w\w+\b"
    a, b = (re.findall(padrao, par[chave].lower()) for chave in ("a", "b"))
    conferir("Mesmos unigramas, bigramas diferentes", Counter(a) == Counter(b)
             and Counter(zip(a, a[1:])) != Counter(zip(b, b[1:])))
    display(pd.DataFrame(fichas).assign(
        classe_A=lambda df: np.where(df["P(+) em A"] >= .5, "positiva", "negativa"),
        classe_B=lambda df: np.where(df["P(+) em B"] >= .5, "positiva", "negativa"),
    ).round(4))
    """, "predicoes-pareadas-frases")
    code("""
    rastreio = read_json("seminario/dados/animacoes.json")["cnn"]
    figura_filtro, eixos = plt.subplots(1, 2, figsize=(9, 3), sharey=True, layout="constrained")
    for letra, eixo in zip(("A", "B"), eixos):
        caso = rastreio["casos"][letra]
        resposta = caso["janelas"]["3"]
        ativacoes = np.maximum(resposta["pre_relu"], 0)
        pico = int(ativacoes.argmax())
        conferir(f"ReLU e janela máxima · frase {letra}",
                 np.allclose(ativacoes, resposta["ativacoes"]) and pico == resposta["max_indice"])
        janela = " ".join(caso["tokens"][pico:pico + 3])
        print(f"{letra} · {janela} · máximo {ativacoes[pico]:.2f} · passo 2 {ativacoes[::2].max():.2f}")
        eixo.bar(np.arange(1, len(ativacoes) + 1), ativacoes,
                 color=["#167D92" if i == pico else "#b6c9d4" for i in range(len(ativacoes))])
        eixo.set(title=f"Frase {letra}", xlabel="Janela", xticks=np.arange(1, len(ativacoes) + 1))
    eixos[0].set_ylabel("Ativação após ReLU")
    figura_filtro.suptitle("Filtro 89 · palavras · semente 17")
    plt.show()

    passos = read_json("seminario/dados/passo_resultados.json")
    with read_npz("seminario/dados/passo_predicoes.npz") as predicoes_passo:
        for passo, registro in passos["medias"].items():
            acc = [np.mean(predicoes_passo[f"passo{passo}_s{s}"].argmax(1) == predicoes_passo["labels"])
                   for s in (17, 29, 43)]
            conferir(f"Acurácia com pesos congelados · passo {passo}", np.isclose(np.mean(acc), registro["accuracy"]))
    display(pd.DataFrame(passos["medias"]).T[["accuracy", "f1_macro", "ce", "fracao_de_janelas"]].round(4))
    """, "filtro-e-passo")


def external(md, code):
    md(f"""
    <a id="fora-do-imdb"></a>
    ## Avaliação fora do IMDb

    [Relatório, p. 8]({REPORT}#page=8) · Slides 13 e 15. Amazon e Yelp têm
    rótulos de referência. Os dois comentários do YouTube são exemplos qualitativos.
    """, "fora-do-imdb")
    code("""
    from scipy.special import softmax
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    fora_dominio = read_json("seminario/dados/fora_dominio_resultados.json")
    linhas_dominio = []
    with read_npz("seminario/dados/fora_dominio_predicoes.npz") as predicoes_dominio:
        for dominio, dados in fora_dominio["dominios_rotulados"].items():
            chave = dominio.lower()
            y = predicoes_dominio[f"{chave}_rotulos"]
            conferir(f"Frases rotuladas e balanceadas · {dominio}", len(y) == dados["n"]
                     and int(y.sum()) == dados["positivos"] == len(y) // 2)
            for modo, esperado in dados["medias"].items():
                metricas = []
                for semente in (17, 29, 43):
                    z = predicoes_dominio[f"{chave}_{modo.lower()}_s{semente}"].astype(float)
                    pred = z.argmax(1)
                    metricas.append({"accuracy": accuracy_score(y, pred),
                                     "f1_macro": f1_score(y, pred, average="macro"),
                                     "ce": log_loss(y, softmax(z, axis=1), labels=[0, 1])})
                media = pd.DataFrame(metricas).mean().to_dict()
                conferir(f"Média em três sementes · {dominio} · {modo}",
                         all(np.isclose(media[k], esperado[k]) for k in esperado))
                linhas_dominio.append({"fonte": dominio, "CNN": modo, "n": len(y),
                                       "acurácia (%)": 100 * media["accuracy"],
                                       "F1 macro": media["f1_macro"], "CE": media["ce"]})
    display(pd.DataFrame(linhas_dominio).set_index(["fonte", "CNN"]).round(4))
    """, "amazon-yelp-logits")
    code("""
    tfidf_youtube = read_json("seminario/dados/tfidf_youtube_resultados.json")["comentarios"]
    qwen_youtube = read_json("seminario/dados/qwen_youtube_resultados.json")["comentarios"]
    cnn_youtube = {r["id"]: r for r in fora_dominio["comentarios_youtube"]}
    baseline_youtube = {r["id"]: r for r in tfidf_youtube}
    classificacoes = []
    for qwen in qwen_youtube:
        caso = qwen["id"]
        cnn, tfidf = cnn_youtube[caso], baseline_youtube[caso]
        conferir(f"Mesmo comentário sem rótulo · {qwen['origem']}",
                 qwen["texto"] == cnn["texto"] == tfidf["texto"]
                 and all(r["rotulo_referencia"] is None for r in (qwen, cnn, tfidf)))
        print(qwen["origem"], "·", qwen["texto"])
        linhas = {"TF-IDF": [tfidf["classe"]], "CNN": [r["classe"] for r in cnn["predicoes"]],
                  "Qwen sem raciocínio": [r["classe"] for r in qwen["geracao"]["sem_raciocinio"]],
                  "Qwen com raciocínio": [r["classe"] for r in qwen["geracao"]["com_raciocinio"]]}
        for modelo, classes in linhas.items():
            classificacoes.append({"filme": qwen["origem"], "modelo": modelo,
                                    "positivas": classes.count("positiva"),
                                    "negativas": classes.count("negativa")})
    display(pd.DataFrame(classificacoes).set_index(["filme", "modelo"]))
    """, "youtube-cnn-tfidf-qwen")
    md(f"""
    ### Conclusão

    O TF-IDF ficou à frente da TextCNN nas quatro faixas de comprimento.
    Na CNN, ler a resenha inteira recuperou 0,50 ponto com os mesmos pesos.
    No contraste com *but*, as CNNs ainda seguiram a primeira opinião.
    A conclusão e os limites estão no [relatório, p. 8]({REPORT}#page=8).
    """)
