#!/usr/bin/env python3
"""Publish the frozen generation experiment from saved outputs, without inference."""
from pathlib import Path
import hashlib, html, json, os, shutil, sys, base64
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
sys.path.insert(0,str(ROOT/'src'))
from imdb_lab.common import read_json,save_json,file_sha256
import numpy as np

Q=ROOT/'results/qwen_improved'
FIG=Q/'figures'
E=html.escape
def pct(value):return f'{value:.1%}'.replace('.',',')
def ci(value):return f"{pct(value['mean'])} [{pct(value['ci_low'])}; {pct(value['ci_high'])}]"


def main():
    frozen=read_json(Q/'selection_frozen.json');best=frozen['selected']
    analysis=read_json(Q/'test_analysis.json');semantic=read_json(Q/'semantic_analysis.json')
    notes=read_json(Q/'manual_reading.json');manifest=read_json(Q/'manual_review_selection.json')
    assert {r['id'] for r in notes['cases']}=={r['id'] for r in manifest['cases']}
    rows=read_json(Q/'test/scored.json');data=read_json(Q/'datasets.json')
    by={(r['id'],r['condition']):r for r in rows}
    names=[best['negative']['name'],'baseline',best['positive']['name']]
    for case in notes['cases']:
        for name,digest in case['output_sha256'].items():
            assert hashlib.sha256(by[case['id'],name]['generated'].encode()).hexdigest()==digest
    assert read_json(Q/'test_prefix_checks.json')['status']=='passed'
    assert read_json(Q/'regression_policy.json')['selection_sha256']==file_sha256(Q/'selection_frozen.json')
    assert len(read_json(Q/'regression/scored.json'))==160
    examples=read_json(Q/'presentation_examples.json')['cases']
    entries=read_json(Q/'selected_training_features.json')['directions']
    method='diferença de médias' if all(r['method']=='means' for r in entries) else 'preditor auxiliar com covariância dos gradientes'
    layer=', '.join(str(r['layer']) for r in entries)
    context=entries[0]['context'];n=entries[0]['n']
    target_all=semantic['paired_comparisons']['all']['joint_target_success']
    selected_stats=[next(s for s in semantic['conditions'] if s['condition']==name and s['group']=='all') for name in [names[0],names[2]]]
    note_by={r['id']:r for r in notes['cases']}
    FIG.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    fig,ax=plt.subplots(figsize=(8.4,2.5));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',bbox={'boxstyle':'round,pad=.65','fc':color,'ec':'#80979d'},fontsize=10)
    box(.2,1.1,1.6,1,'Fonte e prompt\niguais','#edf3f4')
    box(2.6,1.1,2,1,'FACT gerado\nsem intervenção','#edf3f4')
    for y,text,color in [(3,'Subtrair v\nTendência à crítica','#fae8e8'),(1.65,'Zero\nExecução normal','#edf3f4'),(.3,'Somar v\nTendência ao elogio','#e0f2ee')]:
        box(6.6,y-.3,2.4,.6,text,color)
        ax.annotate('',xy=(6.45,y),xytext=(4.8,1.6),arrowprops={'arrowstyle':'->','color':'#56717b'})
    ax.annotate('',xy=(2.55,1.6),xytext=(1.95,1.6),arrowprops={'arrowstyle':'->','color':'#56717b'})
    ax.text(5.2,3.8,'Após REVIEW:, intervir no bloco '+layer,ha='center',fontsize=10,color='#007d83')
    fig.savefig(FIG/'opinion_intervention.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,3.1));x=np.arange(3);w=.24
    for k,(label,key,color) in enumerate([('Nova direção, só opinião','selected','#007d83'),('Direção original, todas as posições','original','#a04a53'),('Aleatório, só opinião (média de 3)','random','#8a979d')]):
        values=[]
        for group in ['all','real','neutral']:
            result=semantic['paired_comparisons'][group]['joint_target_success']
            values.append(result[key]['mean'] if key!='random' else np.mean([result[f'random_{i}']['mean'] for i in range(1,4)]))
        ax.bar(x+(k-1)*w,values,w,label=label,color=color)
    ax.set_xticks(x,['Todas as fontes (64)','Resenhas reais (32)','Enredos neutros (32)'])
    ax.set(ylim=(0,1.12),ylabel='Ambos os sentidos com opinião-alvo')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.24),ncol=1,frameon=False)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    fig.savefig(FIG/'controller_quality.png',dpi=180,bbox_inches='tight');plt.close(fig)

    backup=Q/'previous_didactic';backup.mkdir(exist_ok=True)
    for rel in ['relatorio_qwen3.pdf','relatorio_qwen3.html','notebooks/qwen3_8b_steering.ipynb']:
        src=ROOT/rel;dst=backup/src.name
        if src.exists() and not dst.exists():shutil.copy2(src,dst)

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
    from matplotlib import font_manager
    for name,weight in [('IBody','normal'),('IBold','bold')]:
        pdfmetrics.registerFont(TTFont(name,font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight=weight))))
    pdfmetrics.registerFontFamily('IBody',normal='IBody',bold='IBold',italic='IBody',boldItalic='IBold')
    ink=colors.HexColor('#19313c');teal=colors.HexColor('#007d83');line=colors.HexColor('#d5e1e4')
    styles={key:ParagraphStyle(key,fontName=font,fontSize=size,leading=leading,textColor=color,spaceAfter=7) for key,font,size,leading,color in [
        ('title','IBold',21,26,ink),('sub','IBold',11,15,teal),('body','IBody',9,12.8,ink),
        ('small','IBody',7.5,10.5,colors.HexColor('#52656f')),('cell','IBody',8,11,ink),('head','IBold',8,11,colors.white)]}
    story=[];width=A4[0]-84
    def p(text,style='body'):return Paragraph(text,styles[style])
    def add(text,style='body'):story.append(p(text,style))
    def page(number,title):
        if number>1:story.append(PageBreak())
        add(f'QWEN3-8B / INTERVENÇÕES NAS ATIVAÇÕES / {number:02d}','small');add(title,'title')
    def table(headers,records,widths=None):
        t=Table([[p(E(str(v)),'head') for v in headers]]+[[p(E(str(v)).replace('\n','<br/>'),'cell') for v in row] for row in records],colWidths=widths or [width/len(headers)]*len(headers),repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),teal),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eff5f6'),colors.white]),
            ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,-1),.3,line)]))
        story.extend([t,Spacer(1,7)])
    def figure(name,maxheight=190):
        im=Image(str(FIG/name));ratio=min(width/im.imageWidth,maxheight/im.imageHeight)
        im.drawWidth=im.imageWidth*ratio;im.drawHeight=im.imageHeight*ratio
        story.extend([im,Spacer(1,8)])
    def text_box(text,label,color='#007d83'):
        t=Table([[p(E(label),'sub')],[p(E(text).replace('\n','<br/>'),'body')]],colWidths=[width])
        t.setStyle(TableStyle([('BOX',(0,0),(-1,-1),.4,line),('LINEBEFORE',(0,0),(0,-1),3,colors.HexColor(color)),
            ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
        story.extend([KeepTogether([t]),Spacer(1,6)])

    page(1,'O que muda quando deslocamos uma ativação?')
    add('Nestor, Letícia e Gabriel | Introdução a Deep Learning','sub')
    add('<b>Pergunta didática:</b> com a fonte, o prompt e os pesos do modelo fixos, como uma soma no estado interno altera o texto gerado? A analogia com modelos de imagem é a intervenção em representações. Não pressupomos um código global como o de uma GAN nem uma transformação invertível como a de um normalizing flow.')
    figure('opinion_intervention.png',180)
    add('O mesmo Qwen, três papéis','sub')
    table(['Papel','Como usamos'],[['Classificar','Um prompt pede negative ou positive; lemos os logits desses próximos tokens.'],['Revelar representações','Capturamos vetores contextuais de 4.096 dimensões depois de blocos do transformer.'],['Gerar','O modelo escreve FACT e REVIEW; alteramos as ativações somente ao produzir a opinião.']],[width*.25,width*.75])
    add('O Qwen3-8B permanece congelado: 8.190.735.360 parâmetros, nenhum treinável nesta execução. Não adicionamos uma cabeça de classificação. A primeira direção veio de uma RFM auxiliar; a direção de geração agora selecionada usa <b>'+method+'</b>. São procedimentos diferentes.')
    add('O objetivo é demonstrar o mecanismo e seus limites, sem exigir que steering seja superior a uma instrução direta. Este relatório incorpora novas inferências e preserva o primeiro experimento no histórico.','small')

    page(2,'Como descobrimos as direções?')
    add('Primeiro experimento: RFM e AGOP','sub')
    add('1. Extraímos ativações de 512 resenhas IMDb rotuladas, em prompts de classificação.<br/>2. Treinamos um preditor auxiliar RFM sobre essas ativações.<br/>3. Calculamos seus gradientes em relação às ativações.<br/>4. Formamos a AGOP, média de g g<super>T</super>, e extraímos seu autovetor principal.<br/>5. Orientamos o sinal pela correlação com os rótulos de treino. O bloco 24 foi escolhido na validação.')
    add('São gradientes do preditor auxiliar, não a Hessiana do Qwen. A AGOP identifica um eixo de sensibilidade; seu sinal precisa ser orientado. Somar e subtrair o mesmo vetor produz os dois sentidos de intervenção.','small')
    add('Direção selecionada para a geração: '+method,'sub')
    if context=='intent':
        add(f'Para cada um de 128 enredos de treino, criamos dois pedidos: escrever uma crítica positiva ou negativa. Extraímos a última ativação do prompt, antes de gerar. São {n} estados no bloco {layer}. A direção é a média dos estados com pedido positivo menos a média com pedido negativo, normalizada. As frases de opinião das referências não entram nessa extração.')
    else:
        add(f'A seleção usa {n} estados de treino no contexto {E(context)}, bloco {layer}. O arquivo de treinamento selecionado permite inspecionar as ativações e os rótulos usados. O método registrado é {method}.')
    add('<b>Na demonstração, o pedido de sentimento é removido.</b> Fonte e prompt são iguais nos três sentidos; somente o vetor muda. Os contrastes rotulados pertencem à etapa de descoberta da direção.')
    add('h alterado = h + alfa x s x v','sub')
    table(['Elemento','Configuração selecionada'],[['v','Vetor unitário, 4.096 coordenadas; bloco '+layer],['alfa',f"-{abs(best['positive']['gain']):g}, 0, +{abs(best['positive']['gain']):g}; escolhidos na validação"],['s',', '.join(f'{float(s):.4f}' for l,k,s in best['positive']['layers'])+'; escala fixada com dados de treino'],['Onde','Após o token gerado REVIEW:, nos próximos passos de geração. FACT e o marcador são gerados sem esse deslocamento.']],[width*.18,width*.82])
    add('Ler sentimento, alterar uma decisão e controlar uma sequência são problemas diferentes. Nenhuma destas direções é um eixo universal de sentimento.','small')

    for number,case,label in zip([3,4,5],examples,['Um enredo sem opinião','Uma resenha originalmente negativa','Uma resenha originalmente positiva']):
        page(number,label)
        add('Mesma fonte: '+E(case['id']),'sub');text_box(case['source'],'Fonte')
        for name,title,color in zip(names,['Intervenção negativa','Sem steering','Intervenção positiva'],['#a0444c','#52656f','#007d83']):
            text_box(by[case['id'],name]['generated'],title,color)
        add(E(note_by[case['id']]['reading']))
        add('Saídas integrais, sem edição ou seleção por qualidade. Os três casos foram definidos pela ordem do manifesto: primeiro enredo e primeira resenha de cada rótulo. A leitura qualitativa é do assistente, sem anotação humana independente.','small')

    page(6,'O efeito se mantém em fontes reservadas?')
    add('Configuração congelada antes de gerar o teste: 64 fontes novas para esta etapa, sendo 32 resenhas reais e 32 enredos neutros sintéticos. Onze condições produzem 704 textos. Todas as saídas são mantidas; não há reranqueamento nem novas tentativas por qualidade.')
    figure('controller_quality.png',185)
    add('Figura: proporção de fontes em que os dois sentidos foram identificados como as opiniões-alvo. Diagnóstico automático pelo mesmo Qwen sem steering. Nos textos estruturados, ele lê o campo REVIEW; FACT é avaliado separadamente quanto a suporte factual. Não é acurácia contra avaliação humana independente.','small')
    table(['Condição','Ambos os sentidos: média [IC 95%]'],[['Nova direção, só opinião',ci(target_all['selected'])],['Direção original, todas as posições',ci(target_all['original'])]],[width*.53,width*.47])
    table(['Verificação','Negativa','Positiva'],[
        ['Opinião-alvo reconhecida',pct(selected_stats[0]['target_rate']),pct(selected_stats[1]['target_rate'])],
        ['Sem alerta factual pelo juiz automático',f"{selected_stats[0]['supported_facts_n']}/64",f"{selected_stats[1]['supported_facts_n']}/64"],
        ['Trecho factual e seus tokens iguais ao zero','64/64','64/64'],
        ['Limites de geração atingidos',next(s for s in analysis['conditions'] if s['condition']==names[0])['hit_token_limit_n'],next(s for s in analysis['conditions'] if s['condition']==names[2])['hit_token_limit_n']]
    ],[width*.54,width*.23,width*.23])
    add('Intervalos descritivos: Wilson para proporções; 5.000 reamostragens pareadas das fontes para diferenças. O prompt é igual entre os controles, mas direção, bloco, amplitude e local de aplicação mudam juntos em relação ao controlador original. TF-IDF, CNN e BERT leem também o texto completo.','small')

    page(7,'O que aprendemos e o que permanece limitado')
    add('O treino pequeno explica a primeira falha?','sub')
    add('A direção original já separava bem os rótulos em classificação: AUC de aproximadamente 0,977 na validação. Isso enfraquece a hipótese de que faltou apenas treinar o scorer. A transferência para escrita era mais fraca; contexto, posição e orçamento de entrada mudavam juntos. Não isolamos uma causa única.')
    add('Testamos 128, 512 e 2.048 exemplos auxiliares, diferentes blocos, contextos e posições. Mais dados ajudaram algumas representações, mas não resolveram sozinhos a geração. A solução selecionada combina uma direção encontrada no contexto de escrita com a aplicação localizada na opinião.')
    add('As novas variantes RFM usam covariância de gradientes centrados, diferente da AGOP bruta original; suas curvas não equivalem a apenas aumentar o treino da mesma sonda original.','small')
    add('Fatos preservados pelo deslocamento não equivalem a fatos corretos','sub')
    add('O prefixo FACT ficou idêntico entre zero e as intervenções. Isso impede que o steering mude esse trecho, mas o modelo sem steering pode ter resumido a fonte incorretamente. A frase REVIEW ainda pode acrescentar detalhes indevidos. Por isso, avaliamos fatos e opinião separadamente e mantemos os erros no material.')
    add('Na leitura dos casos reservados, a versão positiva de imdb_test_05223 inventa uma cidade pequena e uma época mágica do ano. Em imdb_test_22261, o resumo comum fica ambíguo sobre quem critica a aparência de Harrison Ford. São problemas preservados na nova opinião e no resumo sem intervenção.','small')
    add('O juiz sinalizou dez opiniões positivas por fatos. Na releitura do assistente, nove alertas parecem confundir uma avaliação diferente com erro factual, como elogiar atuações que a fonte criticava. Não corrigimos automaticamente os números: o juiz é um diagnóstico, não uma garantia de verdade.','small')
    add('Controles e leitura','sub')
    add('Zero reproduz a execução normal; retirar os hooks restaura o modelo. Verificamos independência entre linhas do lote e identidade dos tokens factuais. Três direções aleatórias têm a mesma norma e o mesmo local de aplicação da nova direção. O assistente leu os 12 trios definidos antes do teste. As 32 fontes antigas também foram reavaliadas após congelar a seleção.')
    add('Limites do alcance','sub')
    add('Um modelo, geração gulosa curta e um formato com campos explícitos. O resultado não demonstra controle geral de texto livre. O juiz semântico é o mesmo modelo-base; seus escores não são confiança calibrada. As resenhas vêm de um benchmark já usado no projeto, e exposição no pré-treinamento não foi auditada. Há busca extensa na validação, mas o novo teste de geração não escolheu a configuração.')
    add('Reprodução','sub')
    add('O notebook lê resultados reais e reconstrói a direção selecionada em CPU; não precisa carregar o Qwen. Repetir a inferência requer os pesos públicos e CUDA. As fontes, prompts, tokens, vetores, escolhas congeladas e hashes acompanham o pacote. O primeiro relatório e notebook ficam em results/qwen_improved/previous_didactic.')
    add('<link href="https://huggingface.co/Qwen/Qwen3-8B">Modelo oficial Qwen3-8B</link> | <link href="https://arxiv.org/abs/2505.09388">Qwen3 Technical Report</link> | <link href="https://arxiv.org/abs/2502.03708">RFM / AGOP</link> | <link href="https://arxiv.org/abs/2312.06681">Contrastive Activation Addition</link> | <link href="https://ai.stanford.edu/~amaas/data/sentiment/">IMDb Stanford</link>','small')
    def footer(canvas,doc):
        canvas.setFont('IBody',7);canvas.setFillColor(colors.HexColor('#52656f'))
        canvas.drawString(42,26,'Introdução a Deep Learning | Experimento didático, Qwen3-8B congelado')
        canvas.drawRightString(A4[0]-42,26,str(doc.page))
    pdf=ROOT/'relatorio_qwen3.pdf'
    SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=34,bottomMargin=42,title='Qwen3-8B: intervenções nas ativações',author='Nestor, Letícia e Gabriel').build(story,onFirstPage=footer,onLaterPages=footer)

    chart='data:image/png;base64,'+base64.b64encode((FIG/'controller_quality.png').read_bytes()).decode()
    intro=f'<h1>Intervenções nas ativações do Qwen3-8B</h1><p>Pesos e prompt fixos. Direção selecionada: {E(method)}, bloco {layer}, aplicada após REVIEW:.</p><p>Teste reservado: 64 fontes, 704 saídas. Ambos os sentidos com opinião-alvo: {pct(target_all["selected"]["mean"])}; direção original: {pct(target_all["original"]["mean"])}. Diagnóstico automático pelo mesmo modelo sem hooks; não é avaliação humana independente.</p><img src="{chart}" alt="Resultados pareados por grupo"><p>Todos os textos abaixo são saídas integrais e sem edição. O formato FACT/REVIEW é comum às três condições, sem pedido de sentimento-alvo. O prefixo factual igual não garante que seu resumo seja correto. A galeria mostra os 192 textos dos 64 trios; as 704 saídas, incluindo controles, acompanham o pacote.</p>'
    gallery=[]
    for source in data['generation_test']:
        cards=''.join(f'<article><h3>{label}</h3><pre>{E(by[source["id"],name]["generated"])}</pre></article>' for name,label in zip(names,['Intervenção negativa','Sem steering','Intervenção positiva']))
        note='<p>'+E(note_by[source['id']]['reading'])+'</p>' if source['id'] in note_by else ''
        gallery.append(f'<details><summary>{E(source["id"])} | {E(source["kind"])}</summary><p class="source">{E(source["source"])}</p><div class="trio">{cards}</div>{note}</details>')
    css='body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#19313c}h1,h3{color:#007d83}img{max-width:760px;width:100%}.trio{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}article,.source{background:#eff5f6;padding:16px}pre{white-space:pre-wrap;font:15px/1.5 system-ui}details{border-top:1px solid #ccd9dd;padding:14px 0}summary{cursor:pointer;font-weight:600}@media(max-width:850px){.trio{grid-template-columns:1fr}}'
    (ROOT/'relatorio_qwen3.html').write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Qwen3-8B: steering</title><style>'+css+'</style>'+intro+'<h2>Os 64 trios da direção selecionada</h2>'+''.join(gallery)+'</html>')
    build_notebook()
    save_json(Q/'report_build.json',{'pdf_sha256':file_sha256(pdf),'html_sha256':file_sha256(ROOT/'relatorio_qwen3.html'),
        'notebook_sha256':file_sha256(ROOT/'notebooks/qwen3_8b_steering.ipynb'),'selection_sha256':file_sha256(Q/'selection_frozen.json'),
        'test_analysis_sha256':file_sha256(Q/'test_analysis.json'),'manual_reading_sha256':file_sha256(Q/'manual_reading.json')})
    print('Updated report, complete gallery and reading notebook from frozen saved results.')


def build_notebook():
    import nbformat as nb
    from textwrap import dedent
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(s))
    def code(s):cells.append(nb.v4.new_code_cell(dedent(s).strip()))
    md('# Qwen3-8B: intervir nas ativações\n\n**Nestor, Letícia e Gabriel - Introdução a Deep Learning.** O objetivo é compreender o mecanismo, sem precisar provar que steering supera prompting. Este notebook lê saídas reais, preserva falhas e funciona sem GPU. O Qwen permanece congelado. A direção original veio de uma RFM/AGOP; a seleção da geração usa o método identificado abaixo.')
    code('''
        from pathlib import Path
        import json, hashlib, html
        import numpy as np
        import pandas as pd
        from IPython.display import display, HTML
        roots=[Path.cwd(),Path.cwd().parent]
        ROOT=next(p for p in roots if (p/'results/qwen_improved/selection_frozen.json').exists())
        Q=ROOT/'results/qwen_improved'
        def read(p):return json.loads(p.read_text())
        frozen=read(Q/'selection_frozen.json');best=frozen['selected']
        rows=read(Q/'test/scored.json');data=read(Q/'datasets.json')
        names=[best['negative']['name'],'baseline',best['positive']['name']]
        labels=['Intervenção negativa','Sem steering','Intervenção positiva']
        by={(r['id'],r['condition']):r for r in rows}
        display(pd.DataFrame(read(Q/'selected_training_features.json')['directions'])[['name','context','n','layer','method']])
        print('Textos gerados:',len(rows),'| parâmetros treináveis:',read(Q/'test/model_metadata.json')['trainable_parameters'])
    ''')
    md('## 1. Como veio a direção original?\n\nCapturamos ativações de 512 resenhas rotuladas em prompts de classificação. Um preditor RFM aprende a relação entre ativação e rótulo; seus gradientes formam $M=\\frac{1}{n}\\sum_i g_i g_i^T$, a AGOP. Extraímos seu autovetor principal e orientamos o sinal pela correlação com os rótulos de treino. **Não são gradientes de treino dos pesos do Qwen, nem sua Hessiana.** A direção selecionada original foi a do bloco 24.\n\nA AGOP determina um eixo, sem sinal preferido: +v e -v são os dois sentidos do mesmo eixo. A célula abaixo reconstrói esse eixo a partir do fator salvo.')
    code('''
        rfm=np.load(ROOT/'results/qwen/steering/selected_rfm.npz')
        B=rfm['agop_factor']
        eigenvalues,U=np.linalg.eigh(B@B.T)
        axis=B.T@U[:,-1];axis/=np.linalg.norm(axis)
        original=np.load(ROOT/'results/qwen/steering/directions.npz')['rfm_agop']
        alignment=abs(float(axis@original))
        assert alignment>1-1e-8
        display({'dimensões':len(axis),'norma':float(np.linalg.norm(original)),'alinhamento do eixo reconstruído':alignment})
    ''')
    md('## 2. A direção escolhida para escrever\n\nA escolha usa somente a validação, antes do novo teste. Para uma **diferença de médias**, calculamos $v=(\\mu_+-\\mu_-)/\\|\\mu_+-\\mu_-\\|$. No contexto `intent`, são 128 pares do mesmo enredo com instruções positiva/negativa, extraindo o último estado do prompt **antes** de gerar. Assim, não é necessário treinar um scorer para essa alternativa. O pedido de sentimento aparece na descoberta da direção; é removido na demonstração.\n\nNo contexto `factual`, são continuações contrastivas com fatos iguais e opiniões opostas, extraindo estados da resposta. O campo `context` registra qual caso foi realmente selecionado. As variantes exploratórias RFM desta segunda busca usam gradientes centrados, uma covariância de gradientes; isso difere da AGOP bruta do primeiro experimento.')
    code('''
        archive=np.load(Q/'directions.npz');features=np.load(Q/'selected_training_features.npz')
        info=read(Q/'selected_training_features.json')['directions']
        checks=[]
        for entry in info:
            layer=entry['layer'];x=features[f'train_L{layer}'].astype(float);y=features[f'train_labels_L{layer}']
            v=archive[entry['name']]
            result={'bloco':layer,'norma':float(np.linalg.norm(v)),'método':entry['method']}
            if entry['method']=='means':
                reconstructed=x[y==1].mean(0)-x[y==0].mean(0)
                reconstructed/=np.linalg.norm(reconstructed)
                assert np.allclose(reconstructed,v,atol=1e-10,rtol=1e-10)
                result['direção reconstruída dos estados de treino']=True
            checks.append(result)
        display(pd.DataFrame(checks))
    ''')
    md('## 3. Onde ocorre a soma?\n\nUsamos $h_t \\leftarrow h_t+\\alpha s v$. O modelo gera FACT sem steering e só começa a receber a soma depois de produzir o marcador REVIEW:. A geração continua token a token. Não substituímos nem editamos o trecho factual. Fonte, prompt e pesos são iguais nos três sentidos. O formato explícito é parte desta configuração; o resultado não demonstra controle universal de texto livre.\n\nA célula demonstra a aritmética sobre um estado real de treino; as gerações apresentadas depois foram executadas no modelo completo, com GPU.')
    code('''
        layer,name,scale=best['positive']['layers'][0]
        h=features[f'train_L{layer}'][0].astype(float);v=archive[name]
        gain=abs(best['positive']['gain'])
        display(pd.DataFrame([{'alfa':a,'projeção original':float(h@v),'projeção após soma':float((h+a*scale*v)@v),'norma do deslocamento':float(np.linalg.norm(a*scale*v))} for a in [-gain,0,gain]]))
        from IPython.display import Image as DisplayImage
        display(DisplayImage(filename=str(Q/'figures/opinion_intervention.png')))
    ''')
    md('## 4. Mesma fonte, três saídas reais\n\nCasos definidos pela ordem dos dados, sem escolher pelo efeito: primeiro enredo neutro, primeira resenha negativa e primeira positiva. Leia a **opinião sobre a qualidade do filme** separadamente de fatos tristes ou felizes do enredo. Os textos são integrais.')
    code('''
        def gallery_html(source_id):
            source=next(r for r in data['generation_test'] if r['id']==source_id)
            cards=''.join('<td style="vertical-align:top;width:33%;padding:12px;border:1px solid #ccd9dd"><b>'+html.escape(label)+'</b><pre style="white-space:pre-wrap;font:14px/1.5 sans-serif">'+html.escape(by[source_id,name]['generated'])+'</pre></td>' for name,label in zip(names,labels))
            return '<h3>'+html.escape(source_id)+'</h3><p>'+html.escape(source['source'])+'</p><table><tr>'+cards+'</tr></table>'
        def show(source_id):display(HTML(gallery_html(source_id)))
        for case in read(Q/'presentation_examples.json')['cases']:show(case['id'])
    ''')
    md('## 5. Explore as 64 fontes\n\nO seletor percorre o teste completo, incluindo os casos em que o efeito é parcial. Ele lê resultados salvos; não gera uma nova versão.')
    code('''
        options=''.join('<option value="'+html.escape(r['id'])+'">'+html.escape(r['id']+' | '+r['kind'])+'</option>' for r in data['generation_test'])
        payload=json.dumps({r['id']:gallery_html(r['id']) for r in data['generation_test']},ensure_ascii=False).replace('<',chr(92)+'u003c')
        display(HTML('<select id="qwenpick" style="width:90%;padding:8px">'+options+'</select><div id="qwenview">'+gallery_html(data['generation_test'][0]['id'])+'</div><script>(()=>{const payload='+payload+';document.getElementById("qwenpick").onchange=function(){document.getElementById("qwenview").innerHTML=payload[this.value];};})();</script>'))
        print("Se o visualizador bloquear o seletor, execute show('identificador') ou abra a galeria do relatório HTML.")
    ''')
    md('## 6. Avaliação pareada e controles\n\nO teste contém 32 resenhas e 32 enredos sintéticos novos para esta etapa de geração. Comparamos a nova direção, a original e três direções aleatórias com norma e local de aplicação iguais aos da nova. Há 11 condições e 704 saídas; nenhuma foi descartada ou regenerada por qualidade.\n\nO juiz semântico é o **mesmo Qwen congelado sem hooks**, não um avaliador humano independente. Para o formato com campos, ele avalia a nova opinião em REVIEW e os fatos no texto completo. TF-IDF/CNN/BERT avaliam também o texto completo. Os intervalos são descritivos: Wilson para proporções e 5.000 reamostragens pareadas de fontes para diferenças. O prompt é igual; direção, bloco, amplitude e posição de aplicação mudam juntos na comparação com o controlador original. Os controles aleatórios mantêm esses outros fatores iguais aos do novo controlador.')
    code('''
        semantic=read(Q/'semantic_analysis.json');analysis=read(Q/'test_analysis.json')
        display(pd.DataFrame([r for r in semantic['conditions'] if r['condition'] in names]))
        display(pd.DataFrame(semantic['paired_comparisons']['all']['joint_target_success']).T)
        display(pd.DataFrame(analysis['conditions']))
        display(pd.DataFrame({j:v['paired_improvement'] for j,v in analysis['results']['all'].items()}).T)
        display(DisplayImage(filename=str(Q/'figures/controller_quality.png')))
    ''')
    md('## 7. Fatos e falhas\n\nA identidade do prefixo factual mostra que a intervenção não alterou esse trecho. Não prova que o resumo sem steering estava correto. A opinião gerada também pode acrescentar fatos indevidos. A leitura qualitativa do assistente cobre os 12 trios fixados antes de abrir o teste; suas observações não são anotação humana independente. As fontes antigas foram reavaliadas depois de congelar a seleção.\n\nO juiz sinalizou dez textos positivos por fatos. Uma releitura pós-hoc desses alertas identifica uma invenção clara de cenário e nove casos que parecem confundir discordância de opinião com erro factual. Mantemos os rótulos automáticos e as leituras; não estimamos uma taxa factual corrigida.')
    code('''
        prefix=read(Q/'test_prefix_checks.json');assert prefix['status']=='passed'
        display(pd.DataFrame(prefix['checks']))
        manual=read(Q/'manual_reading.json')
        display(pd.DataFrame([{'fonte':r['id'],'leitura':r['reading']} for r in manual['cases']]))
        display(pd.DataFrame(read(Q/'fact_flag_reading.json')['cases'])[['id','classification','reading']])
        display(pd.DataFrame(read(Q/'regression_summary.json')))
    ''')
    md('### Novas gerações das fontes antigas\n\nMaya, Volcano e OSS117 foram definidos para releitura antes de abrir estas novas saídas. A fonte é a mesma da primeira rodada; agora o formato FACT/REVIEW e a intervenção selecionada são aplicados. Isso permite inspecionar os exemplos conhecidos, sem tratá-los como um novo teste independente.')
    code('''
        legacy=read(Q/'regression/scored.json')
        for case in read(Q/'regression_review_selection.json')['cases']:
            source_id=case['id'];part={r['condition']:r for r in legacy if r['id']==source_id}
            cards=''.join('<td style="vertical-align:top;width:33%;padding:12px;border:1px solid #ccd9dd"><b>'+html.escape(label)+'</b><pre style="white-space:pre-wrap;font:14px/1.5 sans-serif">'+html.escape(part[name]['generated'])+'</pre></td>' for name,label in zip(names,labels))
            display(HTML('<h3>'+html.escape(source_id)+'</h3><p>'+html.escape(part['baseline']['source'])+'</p><table><tr>'+cards+'</tr></table>'))
    ''')
    md('## 8. O que podemos concluir?\n\nUma sonda que reconhece sentimento não garante que seu eixo controle uma sequência. A direção original já tinha AUC alta na classificação; mais dados não eram uma solução suficiente. Nesta configuração, uma direção de contexto de escrita e uma intervenção localizada melhoram o controle da opinião. Não isolamos uma causa única nem demonstramos superioridade sobre prompting.\n\nLimites: um modelo, formato explícito, textos curtos, validação reutilizada na seleção, juízes automáticos, benchmark já consultado no projeto e pré-treinamento não auditado. A configuração foi congelada antes das novas gerações de teste.\n\n[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | [RFM / AGOP](https://arxiv.org/abs/2502.03708) | [Adição contrastiva, referência](https://arxiv.org/abs/2312.06681) | [IMDb](https://ai.stanford.edu/~amaas/data/sentiment/)')
    code('''
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        assert sha(Q/'directions.npz')==frozen['directions_sha256']
        assert sha(Q/'datasets.json')==frozen['datasets_sha256']
        assert sha(Q/'selected_training_features.npz')==frozen['selected_training_features_sha256']
        frame=pd.DataFrame(rows)
        assert frame.groupby('id').prompt_sha256.nunique().eq(1).all()
        assert not frame.duplicated(['id','condition']).any()
        print('Verificados: escolha congelada, estados de treino, mesma fonte/prompt e todas as condições salvas.')
    ''')
    out=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nb.write(out,ROOT/'notebooks/qwen3_8b_steering.ipynb')


if __name__=='__main__':main()
