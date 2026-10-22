"""Didactic report from saved Qwen artifacts; no new inference."""
from html import escape

from .common import ROOT, read_json
from .qwen_core import QROOT
from .qwen_didactic import CONDITION_NAMES, probability
from .qwen_report import NAMES


def build_pdf(d, f, folder):
    from matplotlib import font_manager
    from pypdf import PdfReader
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                   SimpleDocTemplate, Spacer, Table, TableStyle)

    for name, weight in [('QBody', 'normal'), ('QBold', 'bold')]:
        font = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight=weight))
        pdfmetrics.registerFont(TTFont(name, font))
    pdfmetrics.registerFontFamily('QBody', normal='QBody', bold='QBold', italic='QBody', boldItalic='QBold')
    ink = colors.HexColor('#19313c'); teal = colors.HexColor('#007d83')
    muted = colors.HexColor('#52656f'); line = colors.HexColor('#d5e1e4')
    styles = getSampleStyleSheet()
    for name, size, leading, font, color in [
        ('QTitle', 22, 27, 'QBold', ink), ('QSub', 11, 15, 'QBold', teal),
        ('QText', 9, 13, 'QBody', ink), ('QSmall', 7.5, 10.5, 'QBody', muted),
        ('QCell', 8, 11, 'QBody', ink), ('QSource', 8.3, 11.6, 'QBody', ink),
    ]:
        styles.add(ParagraphStyle(name, fontName=font, fontSize=size, leading=leading,
                                  textColor=color, spaceAfter=7 if name != 'QCell' else 0,
                                  spaceBefore=7 if name == 'QSub' else 0))
    styles.add(ParagraphStyle('QHead', parent=styles['QCell'], fontName='QBold', textColor=colors.white))
    width = A4[0] - 84
    story = []
    examples = read_json(QROOT/'didactic_examples.json')
    def p(text, style='QText'):
        return Paragraph(text, styles[style])
    def add(text, style='QText'):
        story.append(p(text, style))
    def title(number, text):
        if number > 1:
            story.append(PageBreak())
        add(f'QWEN3-8B / INTERVENÇÕES NAS ATIVAÇÕES / {number:02d}', 'QSmall')
        add(text, 'QTitle')
    def table(headers, rows, widths=None):
        cells = [[p(escape(str(v)), 'QHead') for v in headers]]
        cells += [[p(escape(str(v)).replace('\n', '<br/>'), 'QCell') for v in row] for row in rows]
        t = Table(cells, colWidths=widths or [width/len(headers)]*len(headers), repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), teal), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#eff5f6'), colors.white]),
            ('LINEBELOW', (0, 0), (-1, -1), .3, line),
        ]))
        story.extend([t, Spacer(1, 7)])
    def fig(name, caption, height=190):
        im = Image(str(folder/name))
        ratio = min(width/im.imageWidth, height/im.imageHeight)
        im.drawWidth = im.imageWidth*ratio; im.drawHeight = im.imageHeight*ratio
        story.append(KeepTogether([im, Spacer(1, 4), p(caption, 'QSmall')]))
    def source_box(text):
        t = Table([[p(escape(text), 'QSource')]], colWidths=[width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eff5f6')),
            ('LINEBEFORE', (0, 0), (0, -1), 3, teal),
            ('LEFTPADDING', (0, 0), (-1, -1), 12), ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 9), ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ]))
        story.extend([t, Spacer(1, 8)])
    def output_box(row):
        color = {'rfm_-4': '#a0444c', 'rfm_+0': '#52656f', 'rfm_+4': '#007d83'}[row['condition']]
        t = Table([[p(escape(CONDITION_NAMES[row['condition']]), 'QSub')],
                   [p(escape(row['generated']), 'QText')]], colWidths=[width])
        t.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), .5, line),
            ('LINEBEFORE', (0, 0), (0, -1), 3, colors.HexColor(color)),
            ('LEFTPADDING', (0, 0), (-1, -1), 12), ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 1), ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ]))
        story.extend([KeepTogether([t]), Spacer(1, 8)])

    title(1, 'O que muda quando alteramos uma ativação?')
    add('Nestor, Letícia e Gabriel | Introdução a Deep Learning', 'QSub')
    add('<b>Pergunta didática.</b> Mantendo os pesos e a entrada fixos, o que acontece com as decisões e com o texto gerado quando deslocamos uma representação interna de um modelo de linguagem?')
    add('A proposta estende a discussão de intervenções em espaços latentes de modelos de imagem. O objetivo é compreender e demonstrar o mecanismo e seus limites. <b>Não é necessário mostrar que steering supera prompting.</b>')
    fig('activation_intervention.png', 'Figura 1. Esquema do ponto de intervenção. Na geração, o ciclo continua token por token; nenhum peso do Qwen é atualizado.', 180)
    table(['Papel do mesmo Qwen congelado', 'O que observamos ou fazemos'], [
        ['Classificar', 'O prompt pede negative ou positive. Lemos os escores desses dois próximos tokens.'],
        ['Fornecer representações', 'Capturamos vetores contextuais de 4.096 dimensões em camadas internas.'],
        ['Gerar texto', 'O modelo escreve uma crítica. Somamos a direção às ativações a cada passo.'],
    ], [width*.35, width*.65])
    add('O que foi aprendido?', 'QSub')
    add('Um preditor auxiliar RFM aprende sobre ativações de 512 resenhas rotuladas; dele extraímos uma direção associada a sentimento. O Qwen3-8B mantém seus pesos. Não adicionamos uma cabeça de classificação nem fazemos fine-tuning.')
    add('Este relatório reorganiza resultados existentes. Não houve nova inferência, alteração dos exemplos gerados ou seleção de uma nova direção a partir dos resultados apresentados.', 'QSmall')

    title(2, 'Do vetor interno à intervenção')
    add('Uma ativação não é apenas o embedding da palavra', 'QSub')
    add('O transformer calcula um vetor contextual por posição e por camada. Para aprender a direção, extraímos a última posição do prompt depois dos blocos 12, 24 e 32. Cada vetor tem 4.096 coordenadas e depende da resenha e das instruções. Não manipulamos a tabela de embeddings de entrada.')
    add('Como aprendemos uma direção', 'QSub')
    add('1. Usamos ativações de 512 exemplos de treino com rótulos de sentimento.<br/>2. Ajustamos um preditor auxiliar com kernel, a RFM.<br/>3. Calculamos a AGOP, média dos produtos externos dos gradientes desse preditor, e extraímos seu autovetor principal.<br/>4. Orientamos o vetor para correlação positiva com os rótulos de treino. Camada e parâmetros são escolhidos na validação.')
    add('São gradientes do preditor auxiliar, não a Hessiana do Qwen. A direção é operacional e dependente do contexto, sem garantia de um eixo universal de sentimento.', 'QSmall')
    add('h modificado = h original + alfa × s × v', 'QSub')
    table(['Elemento', 'Significado nesta execução'], [
        ['h', 'Ativação depois do bloco 24; as camadas 25 a 36 continuam o processamento.'],
        ['v', 'Direção unitária de 4.096 dimensões, extraída da AGOP.'],
        ['s', 'Escala calculada somente nas ativações de treino.'],
        ['alfa', 'Sentido e intensidade: -4, 0 e +4 nos exemplos centrais. Zero não desloca a ativação.'],
        ['Onde somamos', 'Todas as posições válidas do prompt e cada novo token durante a geração.'],
    ], [width*.18, width*.82])
    add('A ligação com GANs e normalizing flows', 'QSub')
    table(['Modelo', 'O que a analogia permite dizer'], [
        ['GAN', 'Uma direção no código latente pode alterar a imagem produzida pelo gerador.'],
        ['Normalizing flow', 'Há uma transformação invertível entre dados e representação latente.'],
        ['Transformer aqui', 'Intervimos nas representações internas por token e camada. Não pressupomos um único código global ou uma transformação invertível.'],
    ], [width*.26, width*.74])

    title(3, 'Mesma resenha, outra decisão')
    add('O prompt pede uma palavra: negative ou positive. Lemos os dois logits de próximo token e normalizamos somente entre eles: P(positivo) = exp(l+) / [exp(l+) + exp(l-)]. Isso não é JSON nem uma nova cabeça classificadora.')
    add('Os textos abaixo são completos e não foram truncados. A resenha e o prompt são iguais nas três condições. O que muda é a soma nas ativações. As probabilidades sob intervenção são escores restritos, sem calibração validada.', 'QSmall')
    for sample in examples['classification']:
        add(sample['id']+' | rótulo original: '+('positivo' if sample['label'] else 'negativo'), 'QSub')
        source_box(sample['source'])
        table(['Condição', 'Alfa', 'Logit l+ - l-', 'Escore P(+)', 'Decisão'], [
            [('Negativa' if r['alpha'] < 0 else 'Positiva' if r['alpha'] > 0 else 'Sem steering'),
             f'{r["alpha"]:+g}', f'{r["steered_logit"]:.2f}', probability(r['p_positive_raw']),
             'positive' if r['steered_logit'] >= 0 else 'negative'] for r in sample['outputs']],
            [width*.28, width*.12, width*.21, width*.19, width*.20])
    add('<b>O que isso mostra?</b> A intervenção altera a resposta do modelo. A opinião expressa pelo autor no texto original permanece igual. Mudar a decisão não demonstra, por si só, controle sobre uma sequência gerada.')
    add('Seleção: primeiro exemplo de cada rótulo na ordem do subconjunto reservado, com 40 a 150 palavras e sem truncamento. A escolha não depende do efeito do steering. Escores extremos são mostrados com desigualdades para evitar confundir arredondamento com certeza.', 'QSmall')

    generation_titles = ['Geração: descrição sem opinião', 'Geração: resenha negativa', 'Geração: resenha positiva']
    for page_number, sample, heading in zip([4, 5, 6], examples['generation'], generation_titles):
        title(page_number, heading)
        add('<b>Prompt fixo:</b> escrever duas frases em inglês avaliando o filme, preservando nomes, acontecimentos e cenário. O prompt não pede sentimento positivo nem negativo. Geração gulosa, até 96 novos tokens.', 'QSmall')
        add('Mesma fonte: '+escape(sample['id']), 'QSub')
        source_box(sample['source'])
        for row in sample['outputs']:
            output_box(row)
        add('O que observar', 'QSub')
        add(escape(sample['reading']))
        add('“Positiva” e “negativa” nomeiam o sentido da intervenção; não garantem o sentimento da saída. Fonte e saídas são integrais e sem edição. Leitura qualitativa do assistente, sem anotação humana independente.', 'QSmall')
        if page_number == 4:
            add('As fontes demonstrativas são neutral_01 e a primeira resenha oficial de cada rótulo na ordem do manifesto. A escolha não depende da mudança de sentimento. A fonte neutra foi escrita para a demonstração, não é uma classe neutra do IMDb.', 'QSmall')

    title(7, 'Intervir na decisão e na geração')
    fig('classification_steering.png', 'Figura 2. Mesmas 256 resenhas, texto idêntico. As curvas mostram decisões e escores sob intervenção, não mudança da opinião original.', 180)
    add('Com alfa -4, todas as decisões ficaram negativas; com alfa +4, todas positivas. Em relação à execução normal, foram alteradas 127 e 129 decisões, respectivamente. A direção por diferença de médias também produziu esse efeito. O controle com zero reproduz os escores originais.')
    fig('didactic_generation.png', 'Figura 3. Mudança média dos escores das saídas em relação à geração sem steering da mesma fonte. Há 32 fontes por condição; os avaliadores rodam sem intervenção.', 180)
    add('Na geração livre, a polaridade não mudou consistentemente. A direção foi aprendida em prompts de classificação e aplicada também em prompts de escrita. A dependência de contexto é uma hipótese explicativa, não uma causa demonstrada.')
    add('A conclusão para o seminário', 'QSub')
    add('<b>Ler uma propriedade nas ativações, alterar uma decisão e controlar o significado de um texto inteiro são tarefas diferentes.</b> O experimento demonstra uma intervenção interna real e os limites observados nesta configuração. Não precisamos eleger uma técnica vencedora para explicar esse mecanismo.')

    title(8, 'Escopo, controles e reprodução')
    add('Onde entram as instruções diretas?', 'QSub')
    add('Servem para comparar o local da intervenção: entrada textual versus estado interno. Nesses controles, o prompt pede explicitamente sentimento positivo ou negativo e não há hook. Eles não integram o trio de prompt idêntico. Também preservamos controles por diferença de médias, sonda logística e três direções aleatórias.')
    add('O que foi medido', 'QSub')
    add('Qwen3-8B em BF16 e modo sem raciocínio. RFM com 512 exemplos de treino; escolhas em 256 da validação; intervenções de classificação em 256 do teste. Geração: 24 resenhas oficiais curtas e oito descrições neutras sintéticas, em 17 condições, totalizando 544 saídas. Nenhuma saída foi descartada ou reranqueada.')
    comp = d['comparison'].set_index('model')
    table(['Contexto: teste oficial', 'Acurácia', 'F1 macro'], [
        [NAMES[m], f'{comp.loc[m,"accuracy"]:.2%}', f'{comp.loc[m,"f1_macro"]:.2%}'] for m in NAMES],
        [width*.54, width*.23, width*.23])
    add('São 25.000 avaliações. Qwen recebe até 768 tokens da fonte; BERT usa 192 WordPieces e ajuste supervisionado. A comparação não isola arquitetura, pré-treinamento ou orçamento de entrada. O resultado é contexto, não a pergunta central.', 'QSmall')
    add('Limites e verificação', 'QSub')
    add('Uma configuração e conjunto pequeno de geração. Os juízes automáticos não estabelecem fluência ou preservação factual. Não houve anotação humana independente. O teste foi reutilizado de uma etapa anterior; possível exposição ao IMDb no pré-treinamento não foi auditada. A camada foi escolhida por classificação, não por geração.')
    add('A execução original verificou hook zero, restauração dos hooks e equivalência entre continuação em cache e execução completa. Esta revisão confere as saídas com os arquivos de origem e executa o notebook de leitura, sem carregar o modelo.', 'QSmall')
    add('Como usar e reproduzir', 'QSub')
    add('O notebook executado mostra os trios lado a lado e permite escolher outra fonte entre as 32. O HTML mantém a galeria completa, métricas e auditorias no apêndice. O protocolo está em QWEN_EXPERIMENTO.md. O notebook lê os resultados incluídos sem GPU; repetir a inferência exige CUDA e os pesos públicos.', 'QSmall')
    add('Fontes', 'QSub')
    add('<link href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B: modelo oficial.</link> '
        '<link href="https://arxiv.org/abs/2505.09388">Qwen3 Technical Report.</link><br/>'
        '<link href="https://arxiv.org/abs/2502.03708">Beaglehole et al.: RFM / AGOP.</link> '
        '<link href="https://github.com/dmbeaglehole/neural_controllers">Implementação dos autores.</link><br/>'
        '<link href="https://ai.stanford.edu/~amaas/data/sentiment/">Maas et al.: Large Movie Review Dataset (IMDb).</link>', 'QSmall')

    path = ROOT/'relatorio_qwen3.pdf'
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(teal); canvas.setLineWidth(2)
        canvas.line(42, A4[1]-26, A4[0]-42, A4[1]-26)
        canvas.setFont('QBody', 7); canvas.setFillColor(muted)
        canvas.drawString(42, 23, 'Nestor / Letícia / Gabriel - Introdução a Deep Learning')
        canvas.drawRightString(A4[0]-42, 23, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=42, rightMargin=42,
                            topMargin=42, bottomMargin=38,
                            title='Qwen3-8B: intervenção nas ativações', author='Nestor, Letícia e Gabriel')
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    pages = len(PdfReader(str(path)).pages)
    if pages != 8:
        raise RuntimeError(f'Expected eight didactic pages, got {pages}; inspect and adjust layout.')
    return path
