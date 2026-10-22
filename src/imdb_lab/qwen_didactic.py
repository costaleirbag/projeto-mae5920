"""Teaching views of existing results. No model inference or outcome selection."""
from __future__ import annotations

from html import escape
import textwrap

import numpy as np
import pandas as pd

from .common import ROOT, file_sha256, read_json, save_json
from .qwen_core import QROOT

CONDITIONS = ['rfm_-4', 'rfm_+0', 'rfm_+4']
CONDITION_NAMES = {
    'rfm_-4': 'Intervenção negativa | alfa = -4',
    'rfm_+0': 'Sem steering | alfa = 0',
    'rfm_+4': 'Intervenção positiva | alfa = +4',
}
GEN_NOTES = {
    'neutral_01': 'A fonte descreve acontecimentos, sem avaliar o filme. A intervenção negativa ainda produz elogios como “effectively captures” e “enhances”. A positiva fica mais descritiva. Há mudança de formulação, mas não uma inversão confiável de sentimento.',
    'negative': 'As três saídas mantêm uma avaliação negativa. A intervenção positiva não transforma a crítica em elogio. O texto muda, mas o sinal da intervenção não determina o sentimento observado.',
    'positive': 'As três saídas continuam favoráveis. Mesmo sob intervenção negativa aparecem elogios como “delightful” e “remarkable”. A direção aprendida no contexto de classificação não controla consistentemente o tom nesta tarefa de escrita.',
}


def build_examples():
    """Select by input order/label/length, never by a steering outcome."""
    sources = read_json(QROOT/'generation/sources.json')
    scored = read_json(QROOT/'evaluation/scored_generations.json')
    by = {(r['id'], r['condition']): r for r in scored}
    selected = [next(s for s in sources if s['id'] == 'neutral_01')]
    selected += [next(s for s in sources if s['kind'] == 'official_test_review' and s['source_label'] == label)
                 for label in [0, 1]]
    generation = []
    for source in selected:
        kind = 'neutral_01' if source['id'] == 'neutral_01' else ('positive' if source['source_label'] else 'negative')
        outputs = [by[(source['id'], condition)] for condition in CONDITIONS]
        assert len({r['prompt_sha256'] for r in outputs}) == 1
        assert all(r['source'] == source['source'] for r in outputs)
        generation.append({**source, 'outputs': outputs, 'reading': GEN_NOTES[kind],
                           'reading_provenance': 'Leitura qualitativa do assistente; sem anotação humana independente.'})

    manifest = read_json(QROOT/'features/subset_manifest.json')
    test = pd.read_parquet(ROOT/'data/test.parquet').set_index('id')
    predictions = pd.read_parquet(QROOT/'classification/predictions.parquet').set_index('id')
    interventions = pd.read_parquet(QROOT/'steering/interventions.parquet')
    classification = []
    for label in [0, 1]:
        id_ = next(id_ for id_ in manifest['test']
                   if int(test.loc[id_, 'label']) == label
                   and 40 <= len(test.loc[id_, 'clean_text'].split()) <= 150
                   and not bool(predictions.loc[id_, 'review_truncated']))
        source = test.loc[id_, 'clean_text']
        output = interventions.loc[(interventions.split == 'test') & (interventions.id == id_)
                                  & (interventions.method == 'rfm_agop')
                                  & interventions.alpha.isin([-4, 0, 4])].sort_values('alpha')
        assert len(output) == 3 and output.input_text_unchanged.all()
        classification.append({'id': id_, 'label': label, 'source': source,
                               'input_complete': True, 'outputs': output.to_dict('records')})
    result = {
        'purpose': 'Demonstração didática de intervenções nas ativações, sem nova inferência.',
        'selection_rule': 'Geração: neutral_01 e primeira fonte oficial de cada rótulo na ordem do manifesto. Classificação: primeiro exemplo de cada rótulo na ordem do subconjunto reservado, com 40 a 150 palavras e sem truncamento. Nenhuma seleção depende do efeito do steering.',
        'condition_caveat': 'Positiva e negativa designam o sentido da intervenção, não um sentimento garantido na saída.',
        'generation': generation, 'classification': classification,
        'sources': {str(p.relative_to(ROOT)): file_sha256(p) for p in [
            QROOT/'generation/sources.json', QROOT/'evaluation/scored_generations.json',
            QROOT/'steering/interventions.parquet', QROOT/'features/subset_manifest.json',
            QROOT/'classification/predictions.parquet', ROOT/'data/test.parquet']},
    }
    save_json(QROOT/'didactic_examples.json', result)
    return result


def probability(value):
    if value < .001:
        return '<0,001'
    if value > .999:
        return '>0,999'
    return f'{value:.3f}'.replace('.', ',')


def generation_comparison_html(source, rows=None):
    outputs = source['outputs'] if rows is None else rows
    by = {r['condition']: r for r in outputs}
    cards = []
    for condition in CONDITIONS:
        row = by[condition]
        color = {'rfm_-4': '#a0444c', 'rfm_+0': '#52656f', 'rfm_+4': '#007d83'}[condition]
        cards.append(f'<div style="flex:1;min-width:190px;padding:14px;border:1px solid #d5e1e4;border-top:4px solid {color};border-radius:5px">'
                     f'<b>{CONDITION_NAMES[condition]}</b><p lang="en">{escape(row["generated"])}</p></div>')
    return (f'<section class="paired-example"><h3>{escape(source["id"])}</h3>'
            '<p><b>Mesma fonte nas três condições:</b></p>'
            f'<blockquote lang="en">{escape(source["source"])}</blockquote>'
            '<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:stretch">'+''.join(cards)+'</div>'
            + (f'<p><b>O que observar:</b> {escape(source["reading"])}</p>' if 'reading' in source else '')+'</section>')


def classification_comparison_html(source):
    rows = []
    for row in source['outputs']:
        alpha = row['alpha']
        name = {-4: 'Intervenção negativa', 0: 'Sem steering', 4: 'Intervenção positiva'}[alpha]
        label = 'positive' if row['steered_logit'] >= 0 else 'negative'
        rows.append(f'<tr><td>{name}</td><td>{alpha:+g}</td><td>{row["steered_logit"]:.2f}</td>'
                    f'<td>{escape(probability(row["p_positive_raw"]))}</td><td>{label}</td></tr>')
    return (f'<h3>{escape(source["id"])} | rótulo original: '+('positivo' if source['label'] else 'negativo')+'</h3>'
            f'<blockquote lang="en">{escape(source["source"])}</blockquote>'
            '<table><tr><th>Condição</th><th>Alfa</th><th>Logit positivo - negativo</th>'
            '<th>Escore P(positivo)</th><th>Decisão do modelo</th></tr>'+''.join(rows)+'</table>')


def teaching_figures(d, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    ink, teal, purple = '#19313c', '#007d83', '#8c42a5'
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set(xlim=(0, 10), ylim=(0, 3.5)); ax.axis('off')
    def box(x, y, w, h, label, face='#eff5f6', edge=teal):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.06',
                                   facecolor=face, edgecolor=edge, linewidth=1.3))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', color=ink, fontsize=9)
    box(.1, 1.4, 1.6, .95, 'Mesmo texto\ne mesmo prompt')
    box(2.05, 1.4, 1.8, .95, 'Qwen congelado\nblocos 1 a 24')
    box(4.2, 1.4, 1.6, .95, 'Ativação interna\nh: 4.096 números', face='#f5eef8', edge=purple)
    box(6.15, 1.4, 1.6, .95, 'Qwen congelado\nblocos 25 a 36')
    box(8.1, 1.4, 1.75, .95, 'Escores de rótulos\nou próximo token')
    for start, end in [(1.75, 2), (3.9, 4.15), (5.85, 6.1), (7.8, 8.05)]:
        ax.annotate('', xy=(end, 1.87), xytext=(start, 1.87), arrowprops={'arrowstyle': '->', 'color': ink})
    box(3.65, .12, 2.7, .7, 'Intervenção: h ← h + alfa × s × v', face='#f5eef8', edge=purple)
    ax.annotate('', xy=(5, 1.32), xytext=(5, .85), arrowprops={'arrowstyle': '->', 'color': purple, 'lw': 1.5})
    ax.text(5, 3.07, 'Alteramos as ativações durante a execução; os pesos permanecem iguais.',
            ha='center', color=ink, fontsize=11, weight='bold')
    ax.text(5, 2.65, 'Na geração, a intervenção se repete a cada novo token.', ha='center', color=ink, fontsize=10)
    fig.tight_layout(); fig.savefig(folder/'activation_intervention.png', dpi=190, bbox_inches='tight'); plt.close(fig)

    gen = d['generation'].loc[d['generation'].group == 'all'].set_index('condition')
    conditions = ['rfm_-4', 'rfm_-2', 'rfm_+0', 'rfm_+2', 'rfm_+4']
    x = [-4, -2, 0, 2, 4]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for judge, color, name in [('tfidf', '#0072b2', 'TF-IDF'), ('cnn', '#d55e00', 'CNN'), ('bert', teal, 'BERT')]:
        ax.plot(x, gen.loc[conditions, judge+'_mean_delta_from_unsteered'], 'o-', color=color, label=name)
    ax.axhline(0, color='gray', lw=.8); ax.axvline(0, color='gray', lw=.6)
    ax.set(xticks=x, ylim=(-1, 1), xlabel='Alfa da intervenção nas ativações',
           ylabel='Mudança média no escore P(positivo)',
           title='Geração: diferença em relação à saída sem steering da mesma fonte')
    ax.legend(title='Avaliador automático', fontsize=8, title_fontsize=8, loc='lower right')
    ax.text(.01, .04, '32 fontes por condição; escores automáticos, sem avaliação humana independente.',
            transform=ax.transAxes, fontsize=8, color=ink)
    fig.tight_layout(); fig.savefig(folder/'didactic_generation.png', dpi=190, bbox_inches='tight'); plt.close(fig)


def html_intro(d, examples, folder):
    from .report import image_html
    selected = d['selection']['selected']['layer']
    parts = [
        '<h1>Qwen3-8B: intervindo nas ativações</h1><p class="lead">Nestor, Letícia e Gabriel | Introdução a Deep Learning</p>',
        '<p class="lead"><b>Pergunta didática:</b> mantendo os pesos e a entrada fixos, o que acontece com as decisões e com o texto gerado quando deslocamos uma representação interna do modelo?</p>',
        '<p>O objetivo é ilustrar intervenções em representações de um modelo de linguagem e relacioná-las à discussão de espaços latentes em modelos de imagem. Demonstrar superioridade sobre prompting não é requisito de sucesso.</p>',
        '<h2>1. O mesmo modelo, três papéis</h2><p><b>Classificação:</b> um prompt pede <code>negative</code> ou <code>positive</code>; lemos os escores desses dois próximos tokens. <b>Extração:</b> capturamos vetores internos de 4.096 dimensões. <b>Geração:</b> o Qwen escreve uma crítica, token por token. Não houve fine-tuning nem inclusão de uma cabeça de classificação.</p>',
        image_html(folder/'activation_intervention.png'),
        '<h2>2. Qual é o espaço que modificamos?</h2><p>São ativações contextuais depois de um bloco do transformer, não os embeddings de entrada das palavras. Para aprender a direção, extraímos a última posição do prompt nos blocos 12, 24 e 32 de 512 resenhas rotuladas de treino. Um preditor auxiliar RFM usa esses vetores; os pesos do Qwen ficam congelados.</p>',
        '<p>A AGOP resume produtos externos dos gradientes do preditor auxiliar. Seu autovetor principal fornece uma direção, orientada para correlação positiva com os rótulos de treino. Ela é uma direção operacional associada ao sentimento neste experimento, sem garantia de um eixo universal de sentimento. AGOP não é a Hessiana do Qwen.</p>',
        f'<p><b>Intervenção no bloco {selected}: h′ = h + alfa × s × v.</b> O vetor v tem norma 1; s ajusta a escala com dados de treino; alfa controla sentido e intensidade. A soma atua em todas as posições válidas do prompt e em cada novo token. As camadas seguintes processam a ativação modificada.</p>',
        '<p>Em uma GAN, costuma-se manipular um código latente que o gerador transforma em imagem; em um normalizing flow, a transformação entre dados e latente é invertível. Aqui há um vetor por token e por camada: a analogia é a intervenção na representação, sem supor um único código global ou uma transformação invertível.</p>',
        '<h2>3. Mesma resenha, outra decisão</h2><p>O texto abaixo não é reescrito. Mudamos a ativação e lemos novamente os escores dos mesmos dois rótulos. P(positivo) é normalizado apenas entre essas opções; sob intervenção, não tem calibração validada.</p>',
    ]
    parts += [classification_comparison_html(source) for source in examples['classification']]
    parts += [
        '<p>Conduzir a resposta do modelo para outro rótulo não muda a opinião expressa na resenha original. Este é um experimento sobre o comportamento do modelo.</p>',
        '<h2>4. Mesma fonte, três gerações</h2><p>O prompt pede duas frases em inglês avaliando o filme, preservando nomes, acontecimentos e cenário. Nas três condições, não informa um sentimento-alvo. Usamos geração gulosa, até 96 novos tokens. A única diferença entre as condições é a intensidade da soma nas ativações.</p>',
        '<p><b>Como ler os rótulos:</b> “intervenção positiva” e “intervenção negativa” identificam o sentido do vetor. Não significam que a saída necessariamente será positiva ou negativa. Os textos em inglês abaixo são as saídas integrais salvas, sem edição.</p>',
        '<p class="small">'+escape(examples['selection_rule'])+'</p>',
    ]
    parts += [generation_comparison_html(source) for source in examples['generation']]
    parts += [
        '<h2>5. O que a demonstração permite aprender</h2>',
        image_html(folder/'classification_steering.png'),
        '<p>A intervenção altera fortemente as decisões sobre textos fixos. Nos 256 exemplos reservados, alfa -4 levou todas as decisões a negativo e alfa +4 a positivo. Direções de diferença de médias também produziram esse efeito.</p>',
        image_html(folder/'didactic_generation.png'),
        '<p>A mudança na decisão não se transferiu de modo consistente ao tom da geração livre. A direção foi aprendida no contexto de classificação e aplicada também no contexto de escrita; a dependência do contexto é uma hipótese para investigar, não uma causa demonstrada.</p>',
        '<p><b>Conclusão didática:</b> ler uma propriedade nas ativações, alterar uma decisão e controlar o significado de um texto inteiro são tarefas diferentes. O experimento mostra uma intervenção real e seus limites. O controle por instrução direta serve para comparar onde se intervém - na entrada ou no estado interno - sem transformar a apresentação numa disputa de desempenho.</p>',
        '<p class="small">Leituras qualitativas do assistente; não houve avaliação humana independente. Todas as 544 saídas e os controles estão preservados. O apêndice reúne métricas, auditorias, fontes e a galeria completa.</p>',
    ]
    return ''.join(parts)


def build_notebook():
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    md = nbf.v4.new_markdown_cell
    def code(source):
        return nbf.v4.new_code_cell(textwrap.dedent(source).strip())
    nb.cells = [
        md('# Qwen3-8B: intervenção nas ativações\n\nNestor, Letícia e Gabriel | Introdução a Deep Learning\n\n**Pergunta didática:** mantendo os pesos e a entrada fixos, o que muda quando deslocamos uma representação interna do modelo? O objetivo é ilustrar o mecanismo e seus limites, sem exigir superioridade sobre prompting.\n\nEste notebook executa a leitura e a análise de resultados reais já salvos. Não carrega os pesos de 8B, não faz nova inferência, não treina o Qwen e não precisa de GPU. Execute as células na ordem. O protocolo de inferência está em `QWEN_EXPERIMENTO.md`.'),
        code('''
            from pathlib import Path
            import sys
            import numpy as np
            import pandas as pd
            from IPython.display import display, Image, HTML, Markdown
            ROOT = Path.cwd()
            if not (ROOT/'qwen_config.json').exists():
                ROOT = ROOT.parent
            sys.path.insert(0, str(ROOT/'src'))
            from imdb_lab.common import read_json
            from imdb_lab.qwen_core import QROOT
            from imdb_lab.qwen_didactic import (CONDITIONS, CONDITION_NAMES,
                generation_comparison_html, classification_comparison_html)
            examples = read_json(QROOT/'didactic_examples.json')
            cfg = read_json(ROOT/'qwen_config.json')
            meta = read_json(QROOT/'classification/metadata.json')
            assert meta['trainable_backbone_parameters'] == 0
            print('Modelo:', cfg['model_id'])
            print('Parâmetros do Qwen treinados:', meta['trainable_backbone_parameters'])
            print('Esta execução lê resultados salvos; não gera novos textos.')
        '''),
        md('## 1. Como um modelo gerador foi usado para classificar?\n\nO prompt pede uma única palavra: `negative` ou `positive`. Ambas são tokens unitários neste tokenizer. Lemos seus **logits de próximo token** e escolhemos a maior pontuação. Não acrescentamos uma cabeça de classificação nem usamos JSON como mecanismo de decisão.\n\n$$p_+ = \\frac{\\exp(\\ell_+)}{\\exp(\\ell_+) + \\exp(\\ell_-)}$$\n\nEsse é um escore condicionado às duas opções, não uma probabilidade normalizada sobre todo o vocabulário nem um detector de neutralidade. A temperatura ajustada na calibração pertence à avaliação de classificação; as intervenções abaixo usam escores brutos sem calibração validada.'),
        code('''
            from transformers import AutoTokenizer
            from imdb_lab.qwen_core import classification_prompts, label_ids
            tokenizer = AutoTokenizer.from_pretrained(ROOT/'artifacts/qwen/tokenizer', local_files_only=True)
            sample = examples['classification'][0]
            prompts, info = classification_prompts(tokenizer, [sample['source']])
            print('Tokens dos rótulos [negative, positive]:', label_ids(tokenizer))
            print('Prompt real reconstruído com o tokenizer salvo (sem pesos):')
            print(prompts[0])
            assert not info[0]['review_truncated']
        '''),
        md('## 2. Embeddings de entrada e ativações internas\n\nCada camada do transformer calcula um vetor contextual por posição. Extraímos um vetor de **4.096 dimensões na última posição do prompt**, depois dos blocos 12, 24 e 32. Esses estados dependem tanto da resenha quanto das instruções. Não são apenas os embeddings estáticos das palavras.\n\nA RFM é um **preditor auxiliar** treinado sobre ativações de 512 resenhas rotuladas. Seus gradientes formam a AGOP; o autovetor principal fornece uma direção associada ao sentimento. Qwen permanece congelado. A AGOP usa gradientes do preditor auxiliar, não é a Hessiana do Qwen.\n\nA camada e os parâmetros da direção foram escolhidos na validação. A direção não é um eixo universal de sentimento. Sua utilidade pode depender do prompt, da camada, da posição e da tarefa.'),
        code('''
            selection = read_json(QROOT/'steering/layer_selection.json')
            directions = np.load(QROOT/'steering/directions.npz')
            v = directions['rfm_agop']
            scale = float(directions['scale'])
            layer = int(directions['layer'])
            assert v.shape == (4096,) and np.isclose(np.linalg.norm(v), 1)
            display(pd.DataFrame(selection['all_candidates']))
            print('Bloco escolhido:', layer, '| dimensão:', v.shape[0])
            print('Norma da direção:', np.linalg.norm(v), '| escala de treino:', scale)
            print('AUC da sonda no subconjunto de teste:', selection['heldout_probe_auc'])
        '''),
        md('## 3. A intervenção durante a execução\n\n$$h^{(24)}_t \\leftarrow h^{(24)}_t + \\alpha\\,s\\,v$$\n\n- $v$: direção unitária aprendida; $s$: escala calculada no treino.\n- $\\alpha$: intensidade e sentido; zero reproduz a execução sem deslocamento.\n- A soma ocorre **depois do bloco 24**, em todas as posições válidas do prompt e em cada novo token gerado. As camadas 25 a 36 recebem o estado modificado.\n- Os pesos, a fonte, o prompt e a regra de decodificação permanecem iguais entre as três condições de ativação.\n\n**Analogia com modelos de imagem:** manipulamos uma representação e observamos a saída. Uma GAN costuma receber um código latente; um flow define uma transformação invertível. O transformer tem estados por token e por camada, sem pressupor um único código global ou uma transformação invertível.'),
        code('''
            display(Image(filename=str(QROOT/'figures/activation_intervention.png')))
            # Deslocamentos calculados a partir da direção salva, sem executar o Qwen.
            # Estes valores ilustram a soma em precisão NumPy, não uma nova medição BF16.
            delta = {alpha: alpha*scale*v for alpha in [-4, 0, 4]}
            display(pd.DataFrame({'coordenada': np.arange(8), 'direção_v': v[:8],
                'delta_negativo': delta[-4][:8], 'delta_zero': delta[0][:8],
                'delta_positivo': delta[4][:8]}))
            for alpha, shift in delta.items():
                assert np.isclose(np.linalg.norm(shift), abs(alpha)*scale)
                print(f'Alfa {alpha:+d}: norma do deslocamento por posição = {np.linalg.norm(shift):.4f}')
            assert np.array_equal(delta[-4], -delta[4])
        '''),
        md('## 4. Mesma resenha: sem steering, intervenção negativa e positiva\n\nOs dois textos completos abaixo pertencem ao subconjunto reservado de classificação. A seleção usa ordem, rótulo e comprimento, nunca a resposta à intervenção. **A resenha não é reescrita.** A tabela mostra como a resposta do modelo muda; a opinião expressa pelo autor continua a mesma.\n\nOs nomes “positiva” e “negativa” identificam o sentido da intervenção. Escores perto de 0 ou 1 são apresentados com desigualdades para evitar confundir arredondamento com certeza absoluta.'),
        code('''
            print(examples['selection_rule'])
            interventions = pd.read_parquet(QROOT/'steering/interventions.parquet')
            for sample in examples['classification']:
                display(HTML(classification_comparison_html(sample)))
                actual = interventions.loc[(interventions.id == sample['id']) &
                    (interventions.method == 'rfm_agop') & interventions.alpha.isin([-4,0,4])].sort_values('alpha')
                assert actual.input_text_unchanged.all()
                np.testing.assert_allclose(actual.p_positive_raw, [r['p_positive_raw'] for r in sample['outputs']])
            steering = pd.read_csv(QROOT/'steering/intervention_summary.csv')
            assert (steering.loc[steering.alpha == 0].prediction_flip_n == 0).all()
            display(Image(filename=str(QROOT/'figures/classification_steering.png')))
            display(steering.loc[(steering.split == 'test') & (steering.method == 'rfm_agop') &
                steering.alpha.isin([-4,0,4]), ['alpha','n','mean_p_positive_raw','prediction_flip_n']])
        '''),
        md('## 5. Mesma fonte: três textos realmente gerados\n\nAgora o Qwen escreve duas frases em inglês, preservando nomes, acontecimentos e cenário. **Nas condições de ativação, o prompt não informa um sentimento-alvo e é idêntico para alfa -4, 0 e +4.** A intervenção se repete durante a geração. A decodificação é gulosa e permite até 96 novos tokens.\n\nAs três fontes a seguir são fixadas pela ordem do manifesto: primeira descrição neutra, primeira resenha negativa e primeira positiva. Os textos exibidos são as saídas integrais salvas. “Intervenção positiva” não significa “saída necessariamente positiva”. Leia sentimento, fatos e fluência separadamente; as observações são do assistente, sem avaliação humana independente.'),
        code('''
            rows = pd.read_parquet(QROOT/'evaluation/scored_generations.parquet')
            sources = read_json(QROOT/'generation/sources.json')
            assert len(rows) == 32*17 and rows.groupby('condition').size().eq(32).all()
            assert not rows.duplicated(['id','condition']).any()
            for sample in examples['generation']:
                part = rows.loc[(rows.id == sample['id']) & rows.condition.isin(CONDITIONS)]
                assert part.prompt_sha256.nunique() == 1
                assert part.source.nunique() == 1
                saved = part.set_index('condition').generated.to_dict()
                assert all(saved[r['condition']] == r['generated'] for r in sample['outputs'])
                display(HTML(generation_comparison_html(sample)))
                display(Markdown('---'))
        '''),
        md('### Explorar outra fonte sem escolher apenas acertos\n\nAltere `SOURCE_ID` para qualquer identificador da lista. Isso exibe resultados existentes, sem gerar novas saídas. O relatório HTML também contém a galeria completa.'),
        code('''
            print('Fontes disponíveis:', ', '.join(s['id'] for s in sources))
            SOURCE_ID = 'neutral_02'
            source = next(s for s in sources if s['id'] == SOURCE_ID)
            part = rows.loc[(rows.id == SOURCE_ID) & rows.condition.isin(CONDITIONS)]
            assert part.prompt_sha256.nunique() == 1
            display(HTML(generation_comparison_html(source, part.to_dict('records'))))
        '''),
        md('## 6. O que mudou e o que não ficou controlado?\n\nNos 256 exemplos de classificação, as intervenções fortes conduziram todas as decisões a um dos rótulos. Na geração livre, a polaridade não mudou de forma consistente. A direção foi aprendida em um contexto de classificação e aplicada em outro, de escrita: essa diferença é uma hipótese explicativa, não uma causa demonstrada.\n\n**Ler uma propriedade, alterar uma decisão e controlar o significado de uma sequência são tarefas diferentes.** Esse contraste é parte do conteúdo didático, sem exigir que steering supere prompting.'),
        code('''
            display(Image(filename=str(QROOT/'figures/didactic_generation.png')))
            summary = pd.read_csv(QROOT/'evaluation/generation_summary.csv')
            display(summary.loc[(summary.group == 'all') & summary.condition.isin(CONDITIONS),
                ['condition','n','tfidf_mean_delta_from_unsteered','cnn_mean_delta_from_unsteered',
                 'bert_mean_delta_from_unsteered','three_judge_target_consensus_rate']])
            print('Escores de avaliadores automáticos; não são taxas de sucesso humano.')
        '''),
        md('## 7. Onde entram as instruções diretas?\n\nSão um controle de **onde intervimos**: mudar a instrução de entrada ou modificar estados internos. Nesses dois controles, o prompt pede explicitamente sentimento positivo ou negativo e não há hook. Não fazem parte do trio de prompt idêntico. O objetivo do seminário não é provar superioridade de uma técnica.\n\nTambém existem controles por diferença de médias, sonda logística e três direções aleatórias. Todas as 544 saídas foram preservadas, inclusive falhas. Escores automáticos, similaridade e retenção de entidades não garantem fluência ou preservação factual.'),
        code('''
            from imdb_lab.qwen_experiment import rewriting_prompt
            sample = examples['generation'][0]
            fixed = rewriting_prompt(tokenizer, sample['source'])
            explicit = rewriting_prompt(tokenizer, sample['source'], target='negative')
            assert 'Make the overall opinion of the film negative.' not in fixed
            assert 'Make the overall opinion of the film negative.' in explicit
            print('Trecho acrescentado apenas no controle por prompt:')
            print('Make the overall opinion of the film negative.')
            display(summary.loc[summary.group == 'all',
                ['condition','n','bert_mean_delta_from_unsteered','semantic_cosine_mean',
                 'three_judge_target_consensus_rate']])
        '''),
        md('## Apêndice: classificação e verificações\n\nA comparação dos sistemas é contexto, não a pergunta principal. Qwen recebe até 768 tokens da fonte; BERT usa 192 WordPieces e ajuste supervisionado. O resultado não isola arquitetura, pré-treinamento ou orçamento de entrada. O teste foi reutilizado de uma etapa anterior e a possível exposição ao IMDb no pré-treinamento não foi auditada.'),
        code('''
            from imdb_lab.metrics import evaluate
            pred = pd.read_parquet(QROOT/'classification/predictions.parquet')
            metrics = read_json(QROOT/'classification/metrics.json')
            for split, part in pred.groupby('split'):
                actual = evaluate(part.label, part.p_positive)
                assert all(np.isclose(actual[k], metrics[split]['calibrated'][k])
                    for k in ['accuracy','f1_macro','log_loss','brier','ece_10','fp','fn'])
            assert len(pred.loc[pred.split == 'test']) == 25000
            display(pd.read_csv(QROOT/'evaluation/model_comparison.csv')
                [['model','n','accuracy','f1_macro','fp','fn']])
            checks = read_json(QROOT/'steering/metadata.json')
            assert checks['hook_restoration_exact'] and checks['nonzero_hook_changes_logits']
            assert max(checks['identity_max_error'].values()) < 1e-4
            assert checks['full_hook_cached_max_error'] < 1e-4
            print('Hook zero, restauração e equivalência cache/execução completa: verificados na execução original.')
            worksheet = pd.read_csv(QROOT/'evaluation/qualitative_review_blind.csv')
            print('Revisão humana independente: não realizada; planilha preparada com',len(worksheet),'casos.')
        '''),
        md('## Roteiro para explicar no seminário\n\n1. Mostrar uma resenha e a decisão normal do Qwen congelado.\n2. Localizar o vetor interno por token e explicar como aprendemos a direção.\n3. Mostrar a soma $h + \\alpha s v$, com zero e os dois sentidos.\n4. Comparar as decisões sobre a mesma resenha e as três gerações da mesma fonte.\n5. Concluir que a intervenção é real, mas controlar uma sequência exige mais que alterar um rótulo.\n\n**Reprodução:** o notebook lê artefatos incluídos e dispensa GPU. A inferência original exige CUDA e os pesos públicos; consulte `QWEN_EXPERIMENTO.md`. `python execute_notebook.py --notebook notebooks/qwen3_8b_steering.ipynb` executa esta análise local. A verificação científica completa também usa dados e artefatos omitidos do pacote leve.\n\n**Fontes:** [modelo Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), [RFM / AGOP](https://arxiv.org/abs/2502.03708), [implementação dos autores](https://github.com/dmbeaglehole/neural_controllers), [IMDb Stanford](https://ai.stanford.edu/~amaas/data/sentiment/).'),
    ]
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    path = ROOT/'notebooks/qwen3_8b_steering.ipynb'
    nbf.write(nb, path)
    return path
