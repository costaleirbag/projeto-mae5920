"""Eight-page report assembled only from completed experiment artifacts."""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

from .common import ROOT, read_json
from .metrics import comparison


def build_pdf():
    from matplotlib import font_manager
    from pypdf import PdfReader
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                   SimpleDocTemplate, Spacer, Table, TableStyle)

    for name, weight in (("Body", "normal"), ("BodyBold", "bold")):
        path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=weight))
        pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")
    ink, teal, muted = colors.HexColor("#19313C"), colors.HexColor("#007D83"), colors.HexColor("#51656F")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Main", fontName="BodyBold", fontSize=20, leading=25, textColor=ink, spaceAfter=13))
    styles.add(ParagraphStyle("Sub", fontName="BodyBold", fontSize=11, leading=15, textColor=teal, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle("Text", fontName="Body", fontSize=9, leading=13, textColor=ink, spaceAfter=8))
    styles.add(ParagraphStyle("SmallText", fontName="Body", fontSize=7.6, leading=10.5, textColor=muted, spaceAfter=6))
    styles.add(ParagraphStyle("Cell", fontName="Body", fontSize=7.7, leading=10.6, textColor=ink, alignment=TA_LEFT))
    styles.add(ParagraphStyle("HeadCell", parent=styles["Cell"], fontName="BodyBold", textColor=colors.white))
    width = A4[0] - 84
    story = []

    def p(text, style="Text"):
        return Paragraph(text, styles[style])

    def add(text, style="Text"):
        story.append(p(text, style))

    def title(number, text):
        if number > 1:
            story.append(PageBreak())
        add(f"IMDB / EXPERIMENTO {number:02d}", "SmallText")
        add(text, "Main")

    def table(headers, rows, widths=None):
        contents = [[p(escape(str(v)), "HeadCell") for v in headers]]
        contents.extend([[p(escape(str(v)).replace("\n", "<br/>"), "Cell") for v in row] for row in rows])
        result = Table(contents, colWidths=widths or [width / len(headers)] * len(headers), repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), teal), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EFF5F6"), colors.white]),
            ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#D5E1E4")),
        ]))
        story.extend([result, Spacer(1, 9)])

    def figure(name, caption, max_height=220):
        path = ROOT / "results" / "figures" / name
        im = Image(str(path))
        ratio = min(width / im.imageWidth, max_height / im.imageHeight)
        im.drawWidth, im.drawHeight = im.imageWidth * ratio, im.imageHeight * ratio
        story.append(KeepTogether([im, Spacer(1, 4), p(caption, "SmallText")]))

    def pct(v):
        return f"{v:.2%}"

    names = {"tfidf": "TF-IDF + logística", "cnn": "CNN", "bert": "BERT-base"}
    models = list(names)
    metrics = comparison()
    test = metrics.loc[(metrics.split == "test") & (metrics.probabilities == "calibrated")].set_index("model")
    manifest = read_json(ROOT / "data" / "manifest.json")
    metadata = {m: read_json(ROOT / "results" / m / "metadata.json") for m in models}
    subgroups = pd.read_csv(ROOT / "results" / "analysis" / "subgroup_metrics.csv")
    steering = pd.read_csv(ROOT / "results" / "steering" / "intervention_summary.csv")
    smeta = read_json(ROOT / "results" / "steering" / "metadata.json")
    selection = read_json(ROOT / "results" / "steering" / "probe_selection.json")
    edits = pd.read_csv(ROOT / "results" / "rewrite" / "examples.csv")
    findings = read_json(ROOT / "results" / "findings.json")

    title(1, "Sentimento em avaliações de filmes")
    add("Nestor, Letícia e Gabriel | Introdução a Deep Learning", "Sub")
    add("<b>Pergunta.</b> Uma CNN e um BERT ajustado superam uma referência simples de TF-IDF? Em quais avaliações cometem falsos positivos e falsos negativos? Como exercício adicional, uma direção RFM/AGOP consegue controlar a saída do BERT?")
    table(["Modelo", "Acurácia", "F1 macro", "FP", "FN"],
          [[names[m], pct(test.loc[m, "accuracy"]), pct(test.loc[m, "f1_macro"]), int(test.loc[m, "fp"]), int(test.loc[m, "fn"])] for m in models],
          [width * .32, width * .20, width * .20, width * .14, width * .14])
    add(escape(findings["comparison"]), "SmallText")
    figure("model_comparison.png", "Figura 1. Teste oficial com 25.000 avaliações; limiar de decisão de 0,5.", 185)
    add("Protocolo e integridade", "Sub")
    table(["Divisão", "Avaliações", "Papel"], [[s, info["n"], role] for s, role in [
        ("train", "Vocabulários e pesos"), ("validation", "Hiperparâmetros e checkpoint"),
        ("calibration", "Uma temperatura por modelo"), ("test", "Avaliação final e análise descritiva")]
        for info in [manifest["splits"][s]]], [width * .24, width * .20, width * .56])
    audit = manifest["audit"]
    add(f"Fonte: stanfordnlp/imdb. Excluídas {audit['train_rows_matching_test_text']} linhas do treino que repetiam textos do teste e {audit['duplicate_train_rows_excluded']} repetições internas do treino. Não há texto normalizado idêntico entre divisões. O teste oficial retém {audit['duplicate_test_rows_retained']} repetições internas.", "SmallText")

    title(2, "Modelos, seleção e custo observado")
    table(["Modelo", "Representação e arquitetura", "Escolha e orçamento"], [
        [names["tfidf"], "Unigramas e bigramas; até 80 mil características. Negação preservada.", f"C escolhido em 0,5 / 2 / 8: {metadata['tfidf']['selected_C']}. Texto completo."],
        [names["cnn"], "Embeddings de 128 dimensões, filtros 3/4/5, pooling com máscara de padding e dropout.", f"Até 400 palavras; cinco épocas; checkpoint de maior F1 na validação: época {metadata['cnn']['best_epoch']}."],
        [names["bert"], "BERT-base-uncased pré-treinado, com ajuste integral e cabeça de duas classes.", f"Até 192 WordPieces; duas épocas; batch 32; lr 2e-5. Melhor época: {metadata['bert']['best_epoch']}."],
    ], [width * .19, width * .42, width * .39])
    add("O teste não escolhe modelos. A validação escolhe C e a época; a calibração ajusta probabilidades depois dessa escolha. Uma semente por modelo e uma busca pequena limitam as conclusões à configuração executada.")
    add("Treino e seleção", "Sub")
    table(["Modelo", "Parâmetros ajustados", "Treino (min)", "Inferência* (min)", "Hardware"], [[
        names[m], f"{metadata[m]['trainable_parameters']:,}", f"{metadata[m]['training_seconds']/60:.2f}",
        f"{metadata[m]['all_splits_inference_seconds']/60:.2f}",
        "Apple M1" if m != "bert" else "NVIDIA GB10"] for m in models],
        [width * .21, width * .23, width * .18, width * .21, width * .17])
    add(f"*Inferência sobre as quatro divisões, com entradas já vetorizadas/tokenizadas. O tempo do TF-IDF já inclui {metadata['tfidf'].get('featurization_seconds',0):.1f} s de vetorização e a busca de C. Tempos de parede incluem validação no treino das redes. Os dispositivos e ambientes diferem; esta tabela não mede eficiência das arquiteturas em hardware idêntico.", "SmallText")
    add("Primeira época e checkpoint escolhido", "Sub")
    table(["Modelo", "Época", "Loss de treino", "F1 de validação"], [[names[m], h["epoch"], f"{h['training_loss']:.4f}", pct(h["validation_f1_macro"])]
          for m in ("cnn", "bert") for h in metadata[m]["history"] if h["epoch"] in {1, metadata[m]["best_epoch"]}], [width * .28, width * .16, width * .26, width * .30])
    add("<b>Comparabilidade.</b> O baseline vê a avaliação inteira; CNN e BERT veem prefixos diferentes. Diferenças de qualidade resultam também do pré-treinamento, do orçamento de entrada e da otimização. O código salva revisões de origem, configuração, pesos e previsões por exemplo.", "SmallText")

    title(3, "Decisão e confiança são avaliações distintas")
    add("Ajustamos uma temperatura positiva por modelo em uma divisão separada, minimizando log loss. A transformação preserva a classe escolhida com limiar 0,5; sua finalidade é melhorar a interpretação das probabilidades.")
    figure("calibration.png", "Figura 2. Confiabilidade antes e depois da temperatura; dez intervalos fixos.", 225)
    rows = []
    for m in models:
        for view in ("raw", "calibrated"):
            r = metrics.loc[(metrics.model == m) & (metrics.split == "test") & (metrics.probabilities == view)].iloc[0]
            rows.append([names[m], "Bruta" if view == "raw" else "Calibrada", f"{r.log_loss:.4f}", f"{r.brier:.4f}", f"{r.ece_10:.4f}"])
    table(["Modelo", "Probabilidade", "Log loss", "Brier", "ECE (10)"], rows, [width * .28, width * .24, width * .16, width * .16, width * .16])
    add("Como ler", "Sub")
    add("Acurácia e F1 avaliam decisões; log loss e Brier penalizam probabilidades erradas, especialmente excesso de confiança. ECE compara confiança e frequência observada, mas depende dos intervalos escolhidos. Um valor pequeno de ECE isoladamente não garante calibração em cada subgrupo.")
    add("<b>Não há rótulo neutro.</b> O IMDb utilizado contém avaliações polarizadas. Uma probabilidade próxima de 0,5 significa incerteza do classificador. Os exemplos neutros das demonstrações finais são sintéticos e não constituem um teste de detecção de neutralidade.")

    title(4, "Falsos positivos e falsos negativos")
    add("Classe positiva = avaliação positiva. <b>FP / tipo I:</b> avaliação negativa prevista como positiva. <b>FN / tipo II:</b> avaliação positiva prevista como negativa. As taxas condicionam no rótulo real: FPR = FP/(FP+TN); FNR = FN/(FN+TP).")
    figure("confusion_matrices.png", "Figura 3. Contagens no teste; há 12.500 avaliações de cada classe.", 180)
    figure("length_performance.png", "Figura 4. Desempenho por comprimento do texto completo, medido em palavras.", 190)
    length_rows = []
    for value in ("<=100", "101-250", "251-500", ">500"):
        part = subgroups.loc[(subgroups.group == "length_group") & (subgroups.value.astype(str) == value)].set_index("model")
        length_rows.append([value, int(part.iloc[0].n), *[pct(part.loc[m, "accuracy"]) for m in models]])
    table(["Palavras", "n", "TF-IDF", "CNN", "BERT"], length_rows)
    add("Avaliações longas podem conter ressalvas, narrativas e conclusões depois do limite das redes. A associação entre comprimento e erro é descritiva: não isola o efeito causal do truncamento. As tabelas completas registram denominadores e intervalos de Wilson por classe.", "SmallText")

    title(5, "Grupos semânticos e associações com erros")
    add("Um encoder independente, all-MiniLM-L6-v2, representa os textos. K-means aprende oito grupos em 2.500 avaliações do treino sem rótulos; os 25 mil exemplos do teste recebem esses grupos. Termos característicos também vêm do treino.")
    figure("semantic_error_rates.png", "Figura 5. Taxas condicionais por grupo. Termos são descritores automáticos, não categorias humanas validadas.", 230)
    figure("error_marker_contrasts.png", "Figura 6. Frequência do marcador nos erros menos frequência nos acertos da mesma classe real.", 210)
    add("FP é comparado a TN entre avaliações negativas; FN é comparado a TP entre positivas. Isso evita confundir características de uma classe com características de erro. Marcadores de negação, contraste, mistura lexical e possível ironia são heurísticos e sobrepostos. As associações geram hipóteses para leitura dos exemplos; não demonstram causas.", "SmallText")

    title(6, "Quando o modelo de deep learning falha?")
    add("Abaixo, um FP e um FN de cada modelo, selecionados entre erros de alta confiança. São casos diagnósticos; não estimam a frequência dos padrões. Os textos completos e mais 30 exemplos, incluindo amostra aleatória de erros, acompanham os resultados.")
    examples = read_json(ROOT / "results" / "analysis" / "error_examples.json")
    annotations_path = ROOT / "results" / "analysis" / "qualitative_review.json"
    annotations = read_json(annotations_path) if annotations_path.exists() else {}
    case_rows = []
    for m in models:
        for outcome in ("FP", "FN"):
            ex = next(e for e in examples if e["model"] == m and e["outcome"] == outcome)
            snippet = ex["text"][:220].replace("\n", " ")
            if len(ex["text"]) > 220:
                snippet += " [...]"
            note = annotations.get(f"{m}:{ex['id']}", "Leia o texto completo e confronte o rótulo com o prefixo recebido pela rede antes de atribuir um mecanismo ao erro.")
            case_rows.append([f"{names[m]} / {outcome}\n{ex['id']}\nP+ = {ex['p_positive']:.4f}\n{ex['word_count']} palavras", snippet, note])
    table(["Modelo / erro", "Início da avaliação", "Hipótese de leitura do texto completo"], case_rows,
          [width * .21, width * .37, width * .42])
    add("Confiança não é explicação. Termos positivos podem descrever atores, outro filme ou a premissa dentro de uma crítica negativa; ressalvas negativas podem aparecer numa avaliação globalmente positiva. Conferir o texto completo e o prefixo recebido pela rede é necessário antes de atribuir um erro a um mecanismo específico.", "SmallText")

    title(7, "RFM / AGOP: controlar a saída do BERT")
    add("Adaptação de Beaglehole et al. ao token CLS de um encoder classificador. Ajusta-se regressão com kernel Laplaciano e atualiza-se a métrica pelo produto externo médio dos gradientes (AGOP). A direção é seu autovetor principal, orientado pela correlação com o rótulo positivo.")
    add("K(x,z) = exp[-sqrt((x-z)<super>T</super>M(x-z))/L]<br/>M seguinte = média de grad f(h) grad f(h)<super>T</super>", "Sub")
    add(f"Seleção só na validação: bloco {smeta['selected_layer']}, largura {selection['selected']['bandwidth']}, iteração {selection['selected']['selected_iteration']}. As atualizações recursivas seguintes pioraram a AUC de validação. Usam-se 512 exemplos de treino, 256 de validação e 256 de teste; AUC da sonda no teste: {selection['heldout_test_rfm_auc']:.3f}. A AGOP não é a Hessiana exata.")
    figure("steering.png", "Figura 7. Intervenções com a mesma norma; média e faixa de cinco direções aleatórias. Texto de entrada idêntico.", 230)
    part = steering.loc[(steering.split == "test") & (steering.method == "rfm_agop") & steering.alpha.isin([-4, 0, 4])].sort_values("alpha")
    table(["Alfa", "P(positivo) média bruta", "Decisões alteradas"], [[r.alpha, f"{r.mean_p_positive_raw:.3f}", pct(r.prediction_flip_fraction)] for _, r in part.iterrows()])
    add("<b>O que foi testado.</b> Intensidade zero reproduz a saída original; o caminho com prefixo em cache coincide com o forward completo; hooks são removidos após a intervenção. Há controles de diferença de médias, sonda logística e direções aleatórias com a mesma escala aprendida no treino.", "SmallText")
    n_minus = int(round(part.loc[part.alpha == -4, "prediction_flip_fraction"].iloc[0] * part.loc[part.alpha == -4, "n"].iloc[0]))
    n_plus = int(round(part.loc[part.alpha == 4, "prediction_flip_fraction"].iloc[0] * part.loc[part.alpha == 4, "n"].iloc[0]))
    add(f"<b>Interpretação.</b> O efeito é modesto: {n_minus} / {n_plus} decisões mudaram em alfa -4 / +4 no conjunto de teste. A diferença de médias teve efeito igual ou maior, sem evidência de vantagem da RFM. Texto fixo não transforma o sentimento do autor. Probabilidades sob intervenção não têm calibração validada; gradientes e AGOP são do preditor kernel, não a Hessiana completa do BERT.", "SmallText")

    title(8, "Edição de palavras e limites do exercício")
    add("Para modificar o texto de fato, usa-se uma etapa separada com exatamente o encoder ajustado do classificador. A transformação e o bias MLM vêm do pré-treinamento; a projeção de saída continua ligada aos embeddings ajustados. Duas palavras de opinião são mascaradas e preenchidas em sequência, sem reranqueamento por sentimento. Cinco sondas, sete intensidades e um controle aleatório produzem 70 condições, todas preservadas.")
    shown = edits.loc[(edits.method == "rfm_agop") & edits.id.isin(["neutral_a", "negative_a"]) & edits.alpha.isin([-4, 0, 4])].sort_values(["id", "alpha"])
    table(["Sonda / alfa", "Texto editado", "Juiz TF-IDF", "Juiz BERT"], [[f"{r.id} / {r.alpha:g}", r.edited,
          f"{r.tfidf_edited_p_positive:.3f}", f"{r.bert_edited_p_positive:.3f}"] for _, r in shown.iterrows()],
          [width * .20, width * .48, width * .16, width * .16])
    add("Os juízes acima avaliam as edições sem hooks. Todas as condições, as curvas agregadas e a similaridade semântica ao original estão no notebook e no relatório HTML.", "SmallText")
    add("Compare cada edição a alfa=0: o mascaramento por si só já troca palavras. Transferir uma direção de CLS para todos os tokens e para uma cabeça MLM pode falhar. Fluência, sentimento humano e preservação dos fatos exigem leitura; escores dos classificadores não substituem essa avaliação. Este exercício não é reescrita livre validada.", "SmallText")
    add("<b>Leitura qualitativa.</b> Alfa=+4 gerou elogios claros em negative_a, mas alfa=0 já gerava good/good. Em alfa=-4, acting was suspended muda fatos ou perde naturalidade. Houve controle parcial das escolhas lexicais, sem reescrita bidirecional confiável.", "SmallText")
    add("Limites e continuidade", "Sub")
    add("Uma semente, modelos com diferentes limites de entrada, teste com repetições internas e amostras pequenas de steering. Próximos passos justificáveis: avaliar mais sementes, comparar limites de texto equivalentes e revisar amostras de erros/edições às cegas. Não selecionar novas configurações com os resultados deste teste.", "SmallText")
    add("Referências e reprodução", "Sub")
    add('<link href="https://ai.stanford.edu/~amaas/data/sentiment/">Maas et al. (2011), Learning Word Vectors for Sentiment Analysis.</link><br/>'
        '<link href="https://huggingface.co/google-bert/bert-base-uncased">Google BERT-base-uncased: modelo e tokenizer.</link><br/>'
        '<link href="https://arxiv.org/abs/2502.03708">Beaglehole et al. (2025), Toward universal steering and monitoring of AI models.</link><br/>'
        'Notebook executado, README, código PyTorch, dicionário de dados, previsões, configurações e testes acompanham o relatório. Fontes, revisões e hashes estão nos metadados.', "SmallText")

    output = ROOT / "relatorio_imdb.pdf"

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(teal)
        canvas.setLineWidth(2)
        canvas.line(42, A4[1] - 26, A4[0] - 42, A4[1] - 26)
        canvas.setFont("Body", 7)
        canvas.setFillColor(muted)
        canvas.drawString(42, 23, "Nestor / Letícia / Gabriel - Introdução a Deep Learning")
        canvas.drawRightString(A4[0] - 42, 23, f"{doc.page} / 8")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=42, rightMargin=42,
                           topMargin=42, bottomMargin=38, title="IMDb: modelos, erros e steering",
                           author="Nestor, Letícia e Gabriel")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    pages = len(PdfReader(str(output)).pages)
    if pages != 8:
        raise RuntimeError(f"Expected 8 report pages; got {pages}. Review layout before delivery.")
    return output
