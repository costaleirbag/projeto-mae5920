from __future__ import annotations

import base64
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, config, read_json, save_json
from .metrics import comparison

COLORS = {"tfidf": "#0072B2", "cnn": "#D55E00", "bert": "#009E73"}
NAMES = {"tfidf": "TF-IDF + regressão logística", "cnn": "CNN", "bert": "BERT-base ajustado"}


def summarize_results():
    table = comparison()
    test = table.loc[(table.split == "test") & (table.probabilities == "calibrated")].set_index("model")
    winner = test.f1_macro.idxmax()
    delta = 100 * (test.loc["bert", "accuracy"] - test.loc["tfidf", "accuracy"])
    groups = pd.read_csv(ROOT / "results" / "analysis" / "subgroup_metrics.csv")
    length = groups.loc[(groups.model == "bert") & (groups.group == "length_group")].set_index("value")
    steering = pd.read_csv(ROOT / "results" / "steering" / "intervention_summary.csv")
    heldout = steering.loc[steering.split == "test"]
    rfm = heldout.loc[heldout.method == "rfm_agop"].set_index("alpha")
    random = heldout.loc[heldout.method.str.startswith("random_")].groupby("alpha").mean(numeric_only=True)
    edits = pd.read_csv(ROOT / "results" / "rewrite" / "examples.csv")
    edited = edits.loc[edits.method == "rfm_agop"].groupby("alpha").mean(numeric_only=True)
    findings = {
        "comparison": f"Maior F1 macro observado no teste: {NAMES[winner]} ({test.loc[winner, 'f1_macro']:.2%}). Diferença de acurácia BERT menos TF-IDF: {delta:+.2f} pontos percentuais. Uma execução por modelo, com limites de entrada diferentes.",
        "length": f"BERT: acurácia de {length.loc['<=100','accuracy']:.2%} até 100 palavras e {length.loc['>500','accuracy']:.2%} acima de 500. Associação descritiva; não identifica isoladamente o efeito do truncamento.",
        "steering": f"RFM, 256 exemplos de teste: P(positivo) média bruta em alfa -4 / 0 / +4 = {rfm.loc[-4,'mean_p_positive_raw']:.3f} / {rfm.loc[0,'mean_p_positive_raw']:.3f} / {rfm.loc[4,'mean_p_positive_raw']:.3f}. Nos cinco controles aleatórios, médias em -4 / +4 = {random.loc[-4,'mean_p_positive_raw']:.3f} / {random.loc[4,'mean_p_positive_raw']:.3f}. Texto idêntico; probabilidades sob intervenção não têm calibração validada.",
        "editing": f"Edição RFM em cinco sondas sintéticas: média de P(positivo) do juiz TF-IDF sem hooks em alfa -4 / 0 / +4 = {edited.loc[-4,'tfidf_edited_p_positive']:.3f} / {edited.loc[0,'tfidf_edited_p_positive']:.3f} / {edited.loc[4,'tfidf_edited_p_positive']:.3f}. Diagnóstico de escores, não avaliação humana nem benchmark de reescrita.",
    }
    save_json(ROOT / "results" / "findings.json", findings)
    return findings


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    folder = ROOT / "results" / "figures"
    folder.mkdir(exist_ok=True)
    table = comparison()
    test = table.loc[(table.split == "test") & (table.probabilities == "calibrated")].set_index("model")
    models = ["tfidf", "cnn", "bert"]

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(folder / name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(3)
    for axis, metric, label in zip(axes, ["accuracy", "f1_macro"], ["Acurácia", "F1 macro"]):
        values = [test.loc[m, metric] for m in models]
        axis.bar(x, values, color=[COLORS[m] for m in models], width=.6)
        axis.set_xticks(x, ["TF-IDF", "CNN", "BERT"])
        axis.set_ylim(.5, 1)
        axis.set_ylabel(label)
        for i, value in enumerate(values):
            axis.text(i, value + .009, f"{value:.2%}", ha="center")
    fig.suptitle("Teste oficial: 25.000 avaliações; limiar de decisão = 0,5", y=1.03)
    finish(fig, "model_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model in models:
        metrics = read_json(ROOT / "results" / model / "metrics.json")["test"]
        for axis, view, label in zip(axes, ["raw", "calibrated"], ["Antes da calibração", "Após temperatura"]):
            bins = metrics["reliability_" + view]
            axis.plot([b["mean_p_positive"] for b in bins], [b["fraction_positive"] for b in bins],
                      "o-", color=COLORS[model], label=model.upper(), markersize=4)
            axis.plot([0, 1], [0, 1], "k--", alpha=.3)
            axis.set(xlim=(0, 1), ylim=(0, 1), title=label, xlabel="Probabilidade prevista de positivo",
                     ylabel="Fração realmente positiva")
    axes[0].legend()
    finish(fig, "calibration.png")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for axis, model in zip(axes, models):
        row = test.loc[model]
        counts = np.array([[row.tn, row.fp], [row.fn, row.tp]]).astype(int)
        axis.imshow(counts, cmap="Blues", vmin=0, vmax=12500)
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(counts[i, j]), ha="center", va="center", color="white" if counts[i, j] > 7000 else "black", fontsize=12)
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negativo", "Positivo"], yticklabels=["Negativo", "Positivo"],
                 xlabel="Predição", ylabel="Rótulo real", title=model.upper())
    finish(fig, "confusion_matrices.png")

    groups = pd.read_csv(ROOT / "results" / "analysis" / "subgroup_metrics.csv")
    fig, axis = plt.subplots(figsize=(9, 3.8))
    order = ["<=100", "101-250", "251-500", ">500"]
    for model in models:
        rows = groups.loc[(groups.model == model) & (groups.group == "length_group")].set_index("value").reindex(order)
        axis.plot(order, rows.accuracy, "o-", color=COLORS[model], label=model.upper())
    axis.set(xlabel="Comprimento da avaliação (palavras)", ylabel="Acurácia", ylim=(.5, 1), title="Desempenho por comprimento do texto")
    axis.legend(ncol=3)
    finish(fig, "length_performance.png")

    terms = read_json(ROOT / "results" / "analysis" / "cluster_terms.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, metric, title in zip(axes, ["false_positive_rate", "false_negative_rate"], ["Falsos positivos / negativos reais", "Falsos negativos / positivos reais"]):
        rows = groups.loc[groups.group == "semantic_cluster"].copy()
        rows["cluster"] = rows.value.astype(int)
        matrix = rows.pivot(index="cluster", columns="model", values=metric).reindex(columns=models)
        image = axis.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(.3, float(matrix.max().max())))
        axis.set_xticks(range(3), ["TF-IDF", "CNN", "BERT"])
        axis.set_yticks(range(len(matrix)), [f"{i}: " + ", ".join(terms[str(i)][:3]) for i in matrix.index], fontsize=10.5)
        axis.set_title(title, fontsize=11)
        for i in range(len(matrix)):
            for j in range(3):
                axis.text(j, i, f"{matrix.iloc[i, j]:.1%}", ha="center", va="center", fontsize=11)
        fig.colorbar(image, ax=axis, shrink=.6)
    finish(fig, "semantic_error_rates.png")

    contrasts = pd.read_csv(ROOT / "results" / "analysis" / "marker_contrasts.csv")
    contrasts["comparison"] = contrasts.model.str.upper() + " / " + contrasts.error
    matrix = contrasts.pivot(index="marker", columns="comparison", values="difference")
    fig, axis = plt.subplots(figsize=(10, 4))
    limit = max(.05, float(np.abs(matrix).max().max()))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    marker_names = {"contrast": "Contraste", "explicit_rating": "Nota no texto", "irony_marker_heuristic": "Marcador de ironia",
                    "mixed_polarity_lexicon": "Léxico misto", "negation": "Negação", "question": "Interrogação"}
    axis.set_yticks(range(len(matrix)), [marker_names.get(m, m) for m in matrix.index])
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            axis.text(j, i, f"{matrix.iloc[i, j]:+.1%}", ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=axis, label="Diferença de frequência (mesma classe)")
    axis.set_title("Associações exploratórias com erros; marcadores heurísticos, não causas")
    finish(fig, "error_marker_contrasts.png")

    steering = pd.read_csv(ROOT / "results" / "steering" / "intervention_summary.csv")
    heldout = steering.loc[steering.split == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, color, label in [("rfm_agop", "#7B3294", "RFM / AGOP"), ("difference_in_means", "#008837", "Diferença de médias"), ("logistic_probe", "#0072B2", "Sonda logística")]:
        rows = heldout.loc[heldout.method == method].sort_values("alpha")
        axes[0].plot(rows.alpha, rows.mean_p_positive_raw, "o-", color=color, label=label, markersize=4)
        axes[1].plot(rows.alpha, rows.prediction_flip_fraction, "o-", color=color, label=label, markersize=4)
    random = heldout.loc[heldout.method.str.startswith("random_")].groupby("alpha")
    for axis, metric in zip(axes, ["mean_p_positive_raw", "prediction_flip_fraction"]):
        mean, lo, hi = random[metric].mean(), random[metric].min(), random[metric].max()
        axis.plot(mean.index, mean.values, "--", color="gray", label="5 direções aleatórias")
        axis.fill_between(mean.index, lo.values, hi.values, color="gray", alpha=.15)
        axis.axvline(0, color="black", linewidth=.6)
        axis.set(xlabel="Intensidade da intervenção (mesma norma entre métodos)", ylim=(0, 1))
    axes[0].set_ylabel("Média da probabilidade bruta de positivo")
    axes[1].set_ylabel("Fração de decisões alteradas")
    axes[0].legend(fontsize=8)
    fig.suptitle("Intervenção nas ativações: o texto de entrada permanece idêntico", y=1.03)
    finish(fig, "steering.png")

    edits = pd.read_csv(ROOT / "results" / "rewrite" / "examples.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for method, style, label in [("rfm_agop", "-", "RFM"), ("random_1", "--", "Controle aleatório")]:
        part = edits.loc[edits.method == method].groupby("alpha")
        for model in ("tfidf", "bert"):
            mean = part[model + "_edited_p_positive"].mean()
            axes[0].plot(mean.index, mean, style, color=COLORS[model], label=f"{label} / juiz {model.upper()}")
        mean = part.semantic_cosine_to_original.mean()
        axes[1].plot(mean.index, mean, style, label=label)
    axes[0].set(xlabel="Intensidade na edição mascarada", ylabel="Probabilidade de positivo após editar", ylim=(0, 1))
    axes[1].set(xlabel="Intensidade na edição mascarada", ylabel="Similaridade semântica ao original", ylim=(0, 1))
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=8)
    fig.suptitle("Edição de palavras com masked LM: demonstração sintética, não benchmark", y=1.03)
    finish(fig, "text_editing.png")
    return folder


def image_html(path):
    return '<img alt="' + escape(path.stem) + '" src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '">'


def html_table(frame, percent_columns=()):
    frame = frame.copy()
    for col in percent_columns:
        frame[col] = frame[col].map(lambda value: f"{value:.2%}")
    return frame.to_html(index=False, escape=True, border=0, classes="data-table", float_format=lambda x: f"{x:.4f}")


def build_html(figures):
    findings = read_json(ROOT / "results" / "findings.json")
    table = comparison()
    manifest = read_json(ROOT / "data" / "manifest.json")
    test = table.loc[(table.split == "test") & (table.probabilities == "calibrated")]
    results = test[["model", "accuracy", "f1_macro", "log_loss", "brier", "ece_10", "fp", "fn"]]
    all_metrics = table[["model", "split", "probabilities", "n", "accuracy", "f1_macro", "log_loss", "ece_10"]]
    cost_rows = []
    for model in ("tfidf", "cnn", "bert"):
        metadata = read_json(ROOT / "results" / model / "metadata.json")
        cost_rows.append({"modelo": model, "treino_e_selecao_min": metadata["training_seconds"] / 60,
                          "vetorizacao_dentro_treino_min": metadata.get("featurization_seconds", 0) / 60,
                          "inferencia_todos_splits_min": metadata["all_splits_inference_seconds"] / 60,
                          "parametros_ajustados": metadata["trainable_parameters"], "dispositivo": metadata["environment"].get("accelerator", metadata["environment"]["device"])})
    steering_meta = read_json(ROOT / "results" / "steering" / "metadata.json")
    probe = read_json(ROOT / "results" / "steering" / "probe_selection.json")
    error_examples = read_json(ROOT / "results" / "analysis" / "error_examples.json")
    review_path = ROOT / "results" / "analysis" / "qualitative_review.json"
    reviews = read_json(review_path) if review_path.exists() else {}
    example_html = []
    for model in ("tfidf", "cnn", "bert"):
        for outcome in ("FP", "FN"):
            row = next(e for e in error_examples if e["model"] == model and e["outcome"] == outcome)
            words = row["text"].split()
            snippet = " ".join(words[:75]) + (" [...]" if len(words) > 75 else "")
            note = reviews.get(f"{model}:{row['id']}", "")
            example_html.append(f'<article><h3>{model.upper()} / {outcome} / P(positivo)={row["p_positive"]:.3f}</h3><p class="small">{escape(row["id"])} | {row["word_count"]} palavras</p><blockquote>{escape(snippet)}</blockquote><p>{escape(note)}</p></article>')
    edits = pd.read_csv(ROOT / "results" / "rewrite" / "examples.csv")
    edit_table = edits.loc[(edits.method == "rfm_agop") & edits.alpha.isin([-4, 0, 4]),
                          ["id", "alpha", "edited", "tfidf_edited_p_positive", "bert_edited_p_positive", "semantic_cosine_to_original"]]
    split_table = pd.DataFrame([{"divisao": name, "n": info["n"], "negativos": info["label_counts"].get("0", 0),
                                 "positivos": info["label_counts"].get("1", 0)} for name, info in manifest["splits"].items()])
    best_validation = table.loc[(table.split == "validation") & (table.probabilities == "raw")].sort_values("f1_macro", ascending=False).iloc[0]
    return f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>IMDb: modelos, erros e steering</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 22px;color:#17212b;line-height:1.55}}h1{{font-size:32px;line-height:1.2}}h2{{margin-top:40px;color:#164e63}}h3{{font-size:16px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #d4dce2}}th{{background:#eaf2f6}}blockquote{{border-left:3px solid #7aa9b6;padding:8px 16px;margin-left:0;background:#f5f8fa}}.small{{font-size:12px;color:#526575}}.note{{background:#fff6df;padding:16px;border-left:4px solid #b47a14}}article{{break-inside:avoid}}@media print{{body{{max-width:none;margin:0}}h2{{break-before:page}}img{{max-height:370px;object-fit:contain}}}}</style>
<h1>IMDb: classificação de sentimentos, análise dos erros e intervenção em ativações</h1>
<p>Nestor, Letícia e Gabriel | Introdução a Deep Learning | Experimento reproduzível</p>
<p class="note">Resultados medidos nesta execução. O experimento principal classifica textos. O steering altera a resposta do classificador com texto idêntico; a edição textual com masked LM é uma demonstração separada. Probabilidades sob intervenção não possuem calibração validada.</p>
<p><b>Resultado principal.</b> {escape(findings['comparison'])}</p>
<h2>1. Pergunta e protocolo</h2>
<p>Comparar uma referência esparsa e duas redes neurais na classificação binária de avaliações de filmes. Positivo é a classe 1. FP (tipo I) significa avaliação negativa classificada como positiva; FN (tipo II), avaliação positiva classificada como negativa.</p>
{html_table(split_table)}
<p>A fonte é <a href="https://huggingface.co/datasets/stanfordnlp/imdb">stanfordnlp/imdb</a>, revisão <code>{escape(manifest['revision'])}</code>. O teste oficial é preservado. Foram excluídas do treino {manifest['audit']['train_rows_matching_test_text']} linhas com texto também presente no teste e {manifest['audit']['duplicate_train_rows_excluded']} repetições internas. O teste retém {manifest['audit']['duplicate_test_rows_retained']} repetições internas. A auditoria usa somente hashes dos textos do teste, nunca seus rótulos para seleção ou ajuste.</p>
<p>Treino ajusta vocabulários e pesos; validação escolhe hiperparâmetros e checkpoint; calibração ajusta uma única temperatura por modelo; teste recebe avaliação final. A fonte fornece avaliações polarizadas, sem uma classe neutra. Um escore próximo de 0,5 expressa incerteza, não neutralidade rotulada.</p>
<h2>2. Modelos e comparação</h2>
<p>TF-IDF usa unigramas e bigramas e regressão logística, com C escolhido na validação. CNN usa embeddings aprendidos, convoluções de larguras 3, 4 e 5, pooling com máscara de padding e dropout. BERT-base-uncased recebe ajuste integral dos pesos. A CNN usa até 400 palavras e o BERT até 192 WordPieces; TF-IDF recebe a avaliação completa. Essa diferença de orçamento de entrada limita a atribuição de ganhos apenas à arquitetura.</p>
{html_table(results, ['accuracy','f1_macro'])}
{image_html(figures / 'model_comparison.png')}
<p>O maior F1 de validação pertence a {escape(NAMES[best_validation.model])}. O teste não foi usado para escolher os hiperparâmetros. A comparação inclui uma execução por modelo; não representa média entre sementes nem busca exaustiva. Uma rede perder para TF-IDF é um resultado legítimo sob este protocolo.</p>
<h2>3. Probabilidades e custo</h2>
{image_html(figures / 'calibration.png')}
<p>Temperatura é ajustada em uma divisão própria, após a escolha do modelo, minimizando log loss. Ela preserva as decisões no limiar 0,5. Log loss, Brier e ECE avaliam probabilidades; acurácia e F1 avaliam decisões. ECE usa dez intervalos fixos e depende dessa discretização.</p>
{html_table(pd.DataFrame(cost_rows))}
<p>O baseline e a CNN foram executados em Apple M1; BERT e análises posteriores usaram NVIDIA GB10. Tempos de parede descrevem essas execuções e não permitem comparar eficiência das arquiteturas em hardware idêntico. O treino do baseline inclui vetorização e busca de C; a coluna de vetorização detalha uma parcela desse tempo. As redes incluem épocas e validação. RSS registrado nos metadados mede memória do processo, não toda a memória do acelerador.</p>
<h2>4. Onde os modelos erram?</h2>
{image_html(figures / 'confusion_matrices.png')}
{image_html(figures / 'length_performance.png')}
<p>As análises por comprimento usam todos os 25 mil exemplos. Avaliações compridas podem perder trechos importantes ao serem truncadas nas redes. A associação entre comprimento e erro não identifica isoladamente o mecanismo causal.</p>
<p>{escape(findings['length'])}</p>
<p>A leitura dos casos de alta confiança revela também rótulos discutíveis: alguns textos oficialmente negativos contêm elogios diretos. Eles permanecem erros segundo o benchmark, mas não são evidência clara de interpretação incorreta do sentimento textual. Não corrigimos rótulos com base nas previsões.</p>
{''.join(example_html)}
<h2>5. Análise semântica segmentada</h2>
<p>Um encoder independente (all-MiniLM-L6-v2) representa os textos. K-means aprende oito grupos em 2.500 exemplos do treino sem rótulos e atribui todos os textos do teste. Termos característicos são extraídos do treino. Os nomes temáticos são aproximações e requerem interpretação humana.</p>
{image_html(figures / 'semantic_error_rates.png')}
{image_html(figures / 'error_marker_contrasts.png')}
<p>As comparações condicionam no rótulo real: FP versus TN entre negativos e FN versus TP entre positivos. Negação, contraste, mistura lexical e ironia são marcadores heurísticos sobrepostos. Diferenças de frequência são hipóteses exploratórias, não explicações causais. Intervalos de Wilson e tamanhos dos grupos constam nas tabelas exportadas; o prefixo limitado do encoder também pode perder contexto.</p>
<h2>6. RFM / AGOP e steering no BERT</h2>
<p>Adaptação de <a href="https://arxiv.org/html/2502.03708v2">Beaglehole et al., Toward universal steering and monitoring of AI models</a>: regressão kernel Laplaciana alterna ajuste do preditor e atualização da métrica pela média dos produtos externos dos gradientes. O autovetor principal dessa AGOP, orientado pela correlação com o rótulo, define a direção de sentimento. A AGOP não é a Hessiana exata.</p>
<p>Camada escolhida na validação: bloco {steering_meta['selected_layer']}; AUC da sonda RFM no subconjunto de teste: {probe['heldout_test_rfm_auc']:.3f}. Usam-se 512 exemplos de treino, 256 de validação e 256 de teste, com IDs registrados. As direções de comparação são diferença de médias, sonda logística e cinco controles aleatórios. Todas têm norma unitária e recebem a mesma escala aprendida no treino.</p>
<p>A validação escolheu a iteração {probe['selected']['selected_iteration']}: a AGOP foi calculada a partir do primeiro ajuste kernel; as atualizações recursivas seguintes pioraram a AUC de validação. No teste de intervenção, diferença de médias provocou deslocamentos iguais ou maiores que a RFM nas intensidades extremas. Esta execução não demonstra vantagem da RFM sobre esse controle simples.</p>
{image_html(figures / 'steering.png')}
<p>{escape(findings['steering'])}</p>
<p>O texto não muda. O resultado mede sensibilidade e controle da decisão do classificador, sem demonstrar reescrita nem mudança do sentimento humano. O código testa a equivalência do caminho com prefixo em cache, intensidade zero e remoção dos hooks. Os controles aleatórios são resumidos pela média e faixa entre cinco direções, não por intervalo de confiança.</p>
<h2>7. Exercício de edição textual</h2>
<p>A cabeça masked-LM recebe exatamente o encoder do classificador ajustado. Sua transformação e seu bias vêm do pré-treinamento; a projeção de saída continua ligada aos embeddings de entrada ajustados. Duas palavras de opinião são mascaradas, e o modelo preenche as posições da esquerda para a direita sob intervenção. Mantêm-se todos os resultados de cinco sondas sintéticas e sete intensidades, além do controle aleatório. Não há reranqueamento guiado por sentimento. Compare o texto editado à condição alfa=0, pois o mascaramento já modifica a tarefa.</p>
{image_html(figures / 'text_editing.png')}
<p>{escape(findings['editing'])}</p>
{html_table(edit_table)}
<p>Probabilidades de três classificadores sem hooks e similaridade semântica oferecem diagnósticos. Fluência, preservação dos fatos e polaridade ainda exigem leitura humana. Transferir uma direção de CLS para todos os tokens e para uma cabeça MLM pode falhar; resultados desfavoráveis também são parte do experimento.</p>
<p>Na leitura qualitativa desta execução, alfa=+4 produziu “the acting was excellent and the story was wonderful”. Contudo, alfa=0 já preenchia essas máscaras com “good”, portanto a transformação desde o original negativo não pode ser atribuída integralmente ao steering. Em alfa=-4, saídas como “the acting was suspended” alteraram fatos ou perderam naturalidade. A edição mostrou controle parcial das escolhas lexicais, sem obter uma reescrita bidirecional confiável.</p>
<h2>8. Quando o modelo de aprendizagem profunda falha?</h2>
<p>As situações investigadas incluem avaliações com elogios e críticas simultâneos, negações, ironia, enredos descritos sem julgamento direto, textos longos e mudança de contexto entre treino e aplicação. Marcadores e exemplos apoiam investigação, sem provar que cada fenômeno causou um erro. A anotação positiva/negativa por nota também simplifica textos com opiniões ambíguas.</p>
<p>Há limitações adicionais: uma semente, orçamentos de entrada distintos, ausência de classe neutra, amostra pequena no steering, duplicações internas mantidas no teste oficial e seleção de arquitetura restrita. Controlar o escore de uma rede pode expor sua fragilidade, e não melhorar sua classificação. Os resultados de edição mascarada não devem ser generalizados a reescrita livre.</p>
<h2>Reprodução e arquivos</h2>
<p>O notebook executado acompanha os dados preparados, modelos salvos, previsões por exemplo, tabelas segmentadas, revisões dos modelos de origem, configurações e testes. Para reconstruir o experimento, siga o README e execute <code>python run.py all</code>. Os artefatos maiores são locais e excluídos do controle de versão.</p>
{html_table(all_metrics, ['accuracy','f1_macro'])}
<p>Fontes: <a href="https://ai.stanford.edu/~amaas/data/sentiment/">Maas et al., IMDb</a>; <a href="https://huggingface.co/google-bert/bert-base-uncased">BERT-base</a>; <a href="https://arxiv.org/abs/2502.03708">Beaglehole et al.</a>. Figuras e métricas foram calculadas pelo código deste projeto.</p>
</html>'''


def build_notebook():
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    def md(text):
        return nbf.v4.new_markdown_cell(text)
    def code(text):
        return nbf.v4.new_code_cell(text)
    nb.cells = [
        md("# IMDb: modelos, erros e steering\n\nNestor, Letícia e Gabriel. Notebook de leitura e verificação dos resultados salvos. As células de treino são opcionais para evitar repetir uma execução longa. O código PyTorch está em `src/imdb_lab/`."),
        code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'config.json').exists())\nsys.path.insert(0, str(ROOT / 'src'))\ndef read(p): return json.loads((ROOT / p).read_text())\ndef plot(name): display(Image(filename=str(ROOT / 'results' / 'figures' / name)))\nmanifest = read('data/manifest.json')\npd.DataFrame({k: {'n': v['n'], **v['label_counts']} for k,v in manifest['splits'].items()}).T"),
        md("## Integridade dos dados\n\nTeste oficial preservado; duplicações textuais removidas do treino. A validação escolhe modelos; uma divisão adicional calibra probabilidades. Não existe classe neutra nesta base."),
        code("display(manifest['audit'])\nframes = {s: pd.read_parquet(ROOT / 'data' / f'{s}.parquet') for s in manifest['splits']}\nfor a in frames:\n    for b in frames:\n        if a != b: assert not set(frames[a].text_sha256) & set(frames[b].text_sha256)\nassert len(frames['test']) == 25000\nprint('Divisões sem sobreposição textual; teste com 25.000 avaliações.')"),
        md("## Modelos\n\nReferência: TF-IDF + regressão logística. CNN PyTorch: embeddings aprendidos + filtros 3/4/5 + pooling mascarado. BERT-base: ajuste integral dos pesos. Vocabulários e pesos usam apenas treino. Compare os orçamentos de entrada, o hardware e a seleção de hiperparâmetros nos metadados."),
        code("from imdb_lab.models import TextCNN\nimport inspect\nprint(inspect.getsource(TextCNN))\nfor m in ['tfidf','cnn','bert']:\n    meta = read(f'results/{m}/metadata.json')\n    print(m, 'parâmetros ajustados:', meta['trainable_parameters'], 'tempo de treino (min):', round(meta['training_seconds']/60,2))"),
        code("comparison = pd.read_csv(ROOT / 'results/comparison.csv')\ndisplay(comparison.loc[(comparison.split=='test') & (comparison.probabilities=='calibrated')])\nplot('model_comparison.png')"),
        code("display(read('results/findings.json'))"),
        md("## Calibração e custo\n\nUma temperatura por modelo é ajustada no conjunto de calibração. Seu ajuste não usa teste e não altera as decisões no limiar 0,5. ECE depende da discretização escolhida."),
        code("display(comparison[['model','split','probabilities','accuracy','f1_macro','log_loss','ece_10']])\nplot('calibration.png')"),
        md("## Falsos positivos e falsos negativos\n\nClasse positiva = avaliação positiva. FP: negativo → positivo. FN: positivo → negativo. As análises comparam FP a TN e FN a TP, mantendo o rótulo real fixo."),
        code("plot('confusion_matrices.png')\nplot('length_performance.png')\nexamples = read('results/analysis/error_examples.json')\ndisplay(pd.DataFrame(examples)[['model','id','outcome','p_positive','word_count','text']])\nreview_path = ROOT/'results/analysis/qualitative_review.json'\nif review_path.exists(): display(json.loads(review_path.read_text()))"),
        md("## Segmentação semântica\n\nClusters aprendidos em amostra do treino sem rótulos, com encoder independente. Frequências e marcadores são exploratórios, sobrepostos e não causais. A tabela inclui tamanhos e intervalos de Wilson."),
        code("display(read('results/analysis/cluster_terms.json'))\nsegments = pd.read_csv(ROOT / 'results/analysis/subgroup_metrics.csv')\ndisplay(segments.loc[segments.group=='semantic_cluster'])\nplot('semantic_error_rates.png')\nplot('error_marker_contrasts.png')"),
        md("## RFM e AGOP\n\nCom $K_M(x,z)=\\exp[-\\sqrt{(x-z)^TM(x-z)}/L]$, ajusta-se regressão kernel e atualiza-se $M$ pela AGOP: $M_{t+1}=n^{-1}\\sum_i\\nabla f_t(h_i)\\nabla f_t(h_i)^T$. Usa-se o autovetor principal, orientado positivamente pela correlação com o rótulo. **AGOP não é a Hessiana exata.** O experimento adapta [Beaglehole et al.](https://arxiv.org/abs/2502.03708) ao CLS de um encoder classificador."),
        code("selection = read('results/steering/probe_selection.json')\ndisplay(selection['selected'])\nprint('AUC da sonda no subconjunto de teste:', selection['heldout_test_rfm_auc'])\nsteering_meta = read('results/steering/metadata.json')\nassert steering_meta['hook_restoration_exact']\nassert max(steering_meta['zero_alpha_max_logit_difference'].values()) < 2e-5\nplot('steering.png')"),
        md("## Demonstrações neutras e mudança do escore\n\nSondas sintéticas neutras não constituem uma classe neutra no IMDb. O texto abaixo permanece idêntico; somente a saída do classificador é alterada. Sob intervenção, não se assume que a calibração usual continue válida."),
        code("probes = pd.DataFrame(read('results/steering/synthetic_probes.json'))\ninterventions = pd.read_parquet(ROOT / 'results/steering/interventions.parquet')\nview = interventions.loc[(interventions.split=='synthetic') & (interventions.method=='rfm_agop') & interventions.alpha.isin([-4,0,4])]\ndisplay(view.merge(probes,on='id')[['id','alpha','clean_text','p_positive_raw','input_text_unchanged']])"),
        md("## Edição real de palavras com masked LM\n\nEtapa separada para alterar o texto. O encoder coincide com o BERT ajustado. A transformação e o bias da cabeça MLM vêm do pré-treinamento; sua projeção de saída permanece ligada aos embeddings ajustados. Preenchimento sequencial de duas palavras mascaradas, sem reranqueamento por sentimento; todos os resultados e o controle aleatório são preservados. Fluência e sentimento ainda exigem revisão humana."),
        code("edits = pd.DataFrame(read('results/rewrite/examples.json'))\ndisplay(edits.loc[edits.alpha.isin([-4,0,4]), ['id','method','alpha','original','edited','tfidf_edited_p_positive','bert_edited_p_positive','semantic_cosine_to_original']])\nplot('text_editing.png')"),
        md("## Quando o modelo de aprendizagem profunda falha?\n\nInvestigue negação, mistura de opiniões, ironia, descrição do enredo e truncamento. Não atribua causalidade somente a marcadores. Limitações: uma semente, orçamentos de entrada distintos, notas polarizadas, teste com repetições internas e amostra pequena no steering. Alterar o escore pode revelar fragilidade; não demonstra melhora na compreensão."),
        md("## Reproduzir treinos\n\nO comando abaixo baixa os dados públicos, refaz as divisões, treina, avalia, analisa e gera os arquivos. Exige tempo e memória. Também é possível executar cada etapa: `prepare`, `baseline`, `cnn`, `bert`, `analyze`, `steer`, `rewrite`, `report`. Veja README e configuração."),
        code("RUN_TRAINING = False\nif RUN_TRAINING:\n    import subprocess\n    subprocess.run([sys.executable, str(ROOT/'run.py'), 'all'], cwd=ROOT, check=True)\nelse:\n    print('Leitura dos resultados concluída; treino não repetido.')"),
    ]
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}}
    notebook_folder = ROOT / "notebooks"
    notebook_folder.mkdir(exist_ok=True)
    nbf.write(nb, notebook_folder / "experimentos_imdb.ipynb")


def run():
    import subprocess
    import sys
    required = [ROOT / "results" / m / "metadata.json" for m in ("tfidf", "cnn", "bert", "analysis", "steering", "rewrite")]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required completed stage missing: {path}")
    summarize_results()
    figures = make_figures()
    (ROOT / "relatorio.html").write_text(build_html(figures))
    build_notebook()
    from .pdf_report import build_pdf
    pdf = build_pdf()
    save_json(ROOT / "results" / "report_manifest.json", {"report": "relatorio.html", "pdf": pdf.name, "notebook": "notebooks/experimentos_imdb.ipynb",
              "figures": [p.name for p in figures.glob("*.png")], "generated_from_saved_results": True})
    subprocess.run([sys.executable, str(ROOT / "execute_notebook.py")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "verify_results.py")], cwd=ROOT, check=True)
    print("Reports, executed notebook and scientific verification completed", flush=True)
