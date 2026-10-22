"""Reports from measured Qwen results; never launches model inference."""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

from .common import ROOT, file_sha256, read_json, save_json
from .qwen_core import QROOT, qconfig
from .report import html_table, image_html

NAMES = {'tfidf': 'TF-IDF + logística', 'cnn': 'CNN', 'bert': 'BERT-base ajustado', 'qwen3_8b': 'Qwen3-8B congelado'}
METHODS = {'rfm_agop': 'RFM / AGOP', 'difference_in_means': 'Diferença de médias',
           'logistic_probe': 'Sonda logística', 'prompt_only': 'Instrução direta'}
COLORS = {'rfm_agop': '#8c42a5', 'difference_in_means': '#008837', 'logistic_probe': '#0072b2'}


def data():
    return {
        'comparison': pd.read_csv(QROOT/'evaluation/model_comparison.csv'),
        'classification': read_json(QROOT/'classification/metrics.json'),
        'classification_meta': read_json(QROOT/'classification/metadata.json'),
        'selection': read_json(QROOT/'steering/layer_selection.json'),
        'steering': pd.read_csv(QROOT/'steering/intervention_summary.csv'),
        'generation': pd.read_csv(QROOT/'evaluation/generation_summary.csv'),
        'contrasts': pd.read_csv(QROOT/'evaluation/generation_method_contrasts.csv'),
        'generation_meta': read_json(QROOT/'generation/metadata.json'),
        'evaluation_meta': read_json(QROOT/'evaluation/metadata.json'),
        'outputs': read_json(QROOT/'evaluation/scored_generations.json'),
    }


def findings(d):
    test=d['comparison'].set_index('model')
    delta=100*(test.loc['qwen3_8b','accuracy']-test.loc['bert','accuracy'])
    selected=d['selection']['selected']
    audit=pd.read_csv(QROOT/'classification/validation_prompt_audit.csv')
    normal_acc=float(((audit.normal_logit>=0)==audit.label).mean())
    short_acc=float(((audit.prefix_192_logit>=0)==audit.label).mean())
    order_flips=int(((audit.reversed_label_order_logit>=0)!=(audit.normal_logit>=0)).sum())
    pilot=read_json(QROOT/'pilot/metadata.json')
    part=d['steering'].loc[d['steering'].split=='test']
    rfm=part.loc[part.method=='rfm_agop'].set_index('alpha')
    means=part.loc[part.method=='difference_in_means'].set_index('alpha')
    gen=d['generation'].loc[d['generation'].group=='all'].set_index('condition')
    contrasts=d['contrasts'].loc[(d['contrasts'].group=='all')&(d['contrasts'].judge=='bert')].set_index('control')
    def contrast(control):
        r=contrasts.loc[control]
        return f"{r.rfm_minus_control:+.3f} (IC de 95% {r.ci_low:+.3f} a {r.ci_high:+.3f})"
    def effect(condition):
        r=gen.loc[condition]
        return f"{r.bert_mean_delta_from_unsteered:+.3f} (IC descritivo de 95% {r.bert_delta_ci_low:+.3f} a {r.bert_delta_ci_high:+.3f})"
    result={
        'didactic_question': 'Mantendo os pesos e a entrada fixos, o que muda quando deslocamos uma representação interna do modelo? O objetivo é ilustrar a intervenção nas ativações, sem exigir superioridade sobre prompting.',
        'classification': f"Qwen3-8B, sem ajuste dos pesos: acurácia de {test.loc['qwen3_8b','accuracy']:.2%} nas 25.000 avaliações oficiais. Diferença observada em relação ao BERT: {delta:+.2f} pontos percentuais.",
        'comparability': 'Qwen recebe até 768 tokens da avaliação e instruções; BERT recebe até 192 WordPieces e foi ajustado no IMDb. Uma execução por configuração. A comparação mede os sistemas executados, sem isolar tamanho, pré-treinamento ou orçamento de entrada. Possível exposição prévia ao IMDb no pré-treinamento não foi auditada.',
        'input_audit': f"Em {len(audit)} exemplos de validação, o Qwen com prefixo de 768 tokens teve acurácia de {normal_acc:.2%}; com 192 tokens, {short_acc:.2%}. Inverter a ordem dos rótulos mudou {order_flips} decisões, apesar de preservar a acurácia agregada. A amostra é pequena e os tokens Qwen não equivalem aos WordPieces do BERT.",
        'restricted_score': f"Na descrição neutra do piloto, o escore restrito foi P(positivo)={pilot['p_positive_raw'][2]:.3f}, embora a massa dos dois rótulos no vocabulário inteiro fosse apenas {100*pilot['allowed_label_mass'][2]:.4f}%. O escore é condicional às duas opções, não confiança geral nem detecção de neutralidade. No conjunto IMDb polarizado, a massa média dos rótulos foi próxima de 1.",
        'selection': f"Bloco {selected['layer']} escolhido pela resposta direcional na validação. Kernel de largura {selected['bandwidth']:g}, iteração {selected['selected_iteration']}; AUC da sonda no teste reservado de 256 exemplos: {d['selection']['heldout_probe_auc']:.3f}.",
        'recursion': ('A configuração escolhida usa a primeira iteração. As atualizações recursivas adicionais desse ajuste não melhoraram sua AUC de validação; esta execução não sustenta um ganho da recursão.' if selected['selected_iteration']==1 else f"A configuração escolhida usa a iteração {selected['selected_iteration']}. Essa escolha foi feita em validação e não demonstra, isoladamente, ganho geral da recursão."),
        'steering': f"Com texto idêntico, RFM em alfa -4 / 0 / +4 produz P(positivo) média de {rfm.loc[-4,'mean_p_positive_raw']:.3f} / {rfm.loc[0,'mean_p_positive_raw']:.3f} / {rfm.loc[4,'mean_p_positive_raw']:.3f}; muda {int(rfm.loc[-4,'prediction_flip_n'])} / {int(rfm.loc[4,'prediction_flip_n'])} das 256 decisões nos extremos. A diferença de médias produz {means.loc[-4,'mean_p_positive_raw']:.3f} / {means.loc[4,'mean_p_positive_raw']:.3f}. Esses escores sob intervenção não têm calibração validada.",
        'generation': f"Foram gerados {d['generation_meta']['outputs_n']} textos: 32 fontes, 17 condições. No juiz BERT sem hooks, a mudança média em P(positivo) em relação à reescrita sem steering é {effect('rfm_-4')} em alfa -4 e {effect('rfm_+4')} em alfa +4.",
        'prompt_control': f"Concordância dos três classificadores com o alvo: instrução positiva {gen.loc['prompt_positive','three_judge_target_consensus_rate']:.1%}, instrução negativa {gen.loc['prompt_negative','three_judge_target_consensus_rate']:.1%}; RFM +4 {gen.loc['rfm_+4','three_judge_target_consensus_rate']:.1%}, RFM -4 {gen.loc['rfm_-4','three_judge_target_consensus_rate']:.1%}. São diagnósticos automáticos, não taxas de sucesso humano.",
        'paired_controls': f"No juiz BERT, a diferença RFM menos médias na separação direcional é {contrast('difference_in_means')}; RFM menos instrução direta, {contrast('prompt_only')}. A separação é [P(positivo,+4) - P(positivo,-4)]/2 por fonte, ou os dois prompts no controle. Esse indicador mede polarização automática e não recompensa fluência ou fatos.",
        'quality': f"Nas 544 saídas, {int(gen.empty_output_n.sum())} vazias e {int(gen.hit_token_limit_n.sum())} atingiram o limite de tokens. Similaridade de embeddings, repetição e retenção de entidades são indicadores auxiliares. Não estabelecem fluência nem preservação factual. Há uma planilha cega de 60 casos, ainda sem anotações humanas.",
        'scope': 'Extensão educacional exploratória com modelo congelado e geração real. O teste já foi consultado na etapa original; esta reutilização não é uma nova validação externa independente. Não houve busca de novos coeficientes usando as saídas geradas do teste, reranqueamento por sentimento nem descarte de falhas. AGOP usa gradientes do preditor auxiliar, não a Hessiana do LLM. A apresentação distingue a leitura das ativações, a mudança de decisão e o controle da geração; superioridade sobre prompting não é o objetivo didático.',
    }
    save_json(QROOT/'findings.json',result)
    return result


def figures(d,classification_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    folder=QROOT/'figures'
    folder.mkdir(exist_ok=True)
    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(folder/name,dpi=180,bbox_inches='tight')
        plt.close(fig)
    comp=d['comparison'].set_index('model')
    order=list(NAMES)
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for ax,metric,title in zip(axes,['accuracy','f1_macro'],['Acurácia','F1 macro']):
        vals=comp.loc[order,metric]
        ax.bar(np.arange(4),vals,color=['#0072b2','#d55e00','#009e73','#8c42a5'])
        ax.set(xticks=np.arange(4),xticklabels=['TF-IDF','CNN','BERT','Qwen3-8B'],ylim=(.7,1),ylabel=title)
        for i,v in enumerate(vals): ax.text(i,v+.007,f'{v:.2%}',ha='center',fontsize=9)
    fig.suptitle('Mesmo teste oficial: 25.000 avaliações; orçamentos de entrada diferentes',y=1.03)
    finish(fig,'models.png')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for view,color,label in [('raw','#d55e00','Bruta'),('calibrated','#8c42a5','Temperatura')]:
        bins=d['classification']['test']['reliability_'+view]
        axes[0].plot([r['mean_p_positive'] for r in bins],[r['fraction_positive'] for r in bins],'o-',color=color,label=label)
    axes[0].plot([0,1],[0,1],'k--',alpha=.35)
    axes[0].set(xlabel='P(positivo) prevista',ylabel='Fração realmente positiva',xlim=(0,1),ylim=(0,1),title='Qwen: confiabilidade no teste')
    axes[0].legend()
    old=pd.read_csv(ROOT/'results/analysis/subgroup_metrics.csv')
    new=pd.read_csv(QROOT/'evaluation/subgroup_metrics.csv')
    length=pd.concat([old,new])
    for model,color in [('bert','#009e73'),('qwen3_8b','#8c42a5')]:
        rows=length.loc[(length.model==model)&(length.group=='length_group')].set_index('value').reindex(['<=100','101-250','251-500','>500'])
        axes[1].plot(np.arange(4),rows.accuracy,'o-',color=color,label=NAMES[model])
    axes[1].set(xticks=np.arange(4),xticklabels=['<=100','101-250','251-500','>500'],xlabel='Palavras no texto completo',ylabel='Acurácia',ylim=(.7,1),title='Associação com comprimento')
    axes[1].legend(fontsize=8)
    finish(fig,'calibration_length.png')
    if classification_only:
        return folder
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    candidates=d['selection']['all_candidates']
    axes[0].bar([str(r['layer']) for r in candidates],[r['validation_directional_response'] for r in candidates],color='#8c42a5')
    axes[0].set(xlabel='Bloco do decoder (1 a 36)',ylabel='Média P(+2) - P(-2)',title='Escolha da camada: só validação')
    rfm_sel=read_json(QROOT/'rfm/selection.json')
    for row in rfm_sel['all_candidates']:
        if row['layer']==d['selection']['selected']['layer']:
            axes[1].plot([h['iteration'] for h in row['history']],[h['validation_auc'] for h in row['history']],'o-',label=f"Largura {row['bandwidth']:g}")
    axes[1].set(xlabel='Iteração RFM',ylabel='AUC na validação',xticks=[1,2,3,4],title='Recursão no bloco escolhido')
    axes[1].legend()
    finish(fig,'selection.png')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    held=d['steering'].loc[d['steering'].split=='test']
    for method,color in COLORS.items():
        rows=held.loc[held.method==method].sort_values('alpha')
        for ax,col in zip(axes,['mean_p_positive_raw','prediction_flip_fraction']):
            ax.plot(rows.alpha,rows[col],'o-',color=color,label=METHODS[method],markersize=4)
    random=held.loc[held.method.str.startswith('random_')].groupby('alpha')
    for ax,col in zip(axes,['mean_p_positive_raw','prediction_flip_fraction']):
        avg,lo,hi=random[col].mean(),random[col].min(),random[col].max()
        ax.plot(avg.index,avg,'--',color='gray',label='3 direções aleatórias')
        ax.fill_between(avg.index,lo,hi,color='gray',alpha=.15)
        ax.axvline(0,color='black',lw=.6)
        ax.set(xlabel='Intensidade alfa; mesma norma por token',ylim=(-.01,1.01))
    axes[0].set_ylabel('Média de P(positivo) bruta')
    axes[1].set_ylabel('Fração de decisões alteradas')
    axes[0].legend(fontsize=7)
    fig.suptitle('Intervenção na classificação: mesmas 256 avaliações; texto idêntico',y=1.03)
    finish(fig,'classification_steering.png')
    gen=d['generation'].loc[d['generation'].group=='all'].set_index('condition')
    chosen=['rfm_-4','rfm_-2','rfm_+0','rfm_+2','rfm_+4','difference_in_means_-4','difference_in_means_+4','logistic_probe_-4','logistic_probe_+4','prompt_negative','prompt_positive']
    labels=['RFM -4','RFM -2','RFM 0','RFM +2','RFM +4','Médias -4','Médias +4','Logística -4','Logística +4','Instrução negativa','Instrução positiva']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,judge,title in zip(axes,['tfidf','bert'],['Juiz TF-IDF','Juiz BERT']):
        part=gen.loc[chosen]
        vals=part[judge+'_mean_delta_from_unsteered'].to_numpy()
        low=part[judge+'_delta_ci_low'].to_numpy(); high=part[judge+'_delta_ci_high'].to_numpy()
        ax.errorbar(vals,np.arange(len(part)),xerr=np.maximum(0,np.vstack([vals-low,high-vals])),fmt='o',color='#8c42a5',capsize=3)
        ax.axvline(0,color='gray',lw=.8)
        ax.set(yticks=np.arange(len(part)),yticklabels=labels,xlabel='Mudança média em P(positivo) vs. reescrita sem steering',title=title,xlim=(-1,1))
    axes[0].invert_yaxis()
    fig.suptitle('Geração real: 32 fontes por condição; IC descritivo por reamostragem pareada',y=1.01)
    finish(fig,'generation_effects.png')
    chosen=['rfm_-4','rfm_+0','rfm_+4','difference_in_means_-4','difference_in_means_+4','prompt_negative','prompt_positive']
    labels=['RFM -4','RFM 0','RFM +4','Médias -4','Médias +4','Instrução negativa','Instrução positiva']
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    part=gen.loc[chosen]
    axes[0].bar(np.arange(len(part)),part.semantic_cosine_mean,color='#007d83')
    axes[1].bar(np.arange(len(part)),part.required_entities_retained,color='#d55e00')
    required=int(part.required_entity_n.max())
    for i,(_,row) in enumerate(part.iterrows()):
        axes[1].text(i,row.required_entities_retained+.08,f'{int(row.required_entities_retained)}/{int(row.required_entity_n)}',ha='center',fontsize=8)
    for ax in axes:
        ax.set_xticks(np.arange(len(part)),labels,rotation=30,ha='right',fontsize=8)
    axes[0].set(ylim=(0,1),ylabel='Cosseno médio com a fonte',title='Similaridade não garante fatos preservados')
    axes[1].set(ylim=(0,required+1),yticks=np.arange(required+1),ylabel='Alvos retidos (nomes e anos)',title='6 alvos em 4 fontes sintéticas; verificação parcial')
    finish(fig,'generation_quality.png')
    return folder


def build_html(d,f,folder):
    comp=d['comparison'].copy()
    comp['model']=comp.model.map(NAMES)
    comp=comp[['model','accuracy','f1_macro','log_loss','brier','ece_10','fp','fn']]
    gen=d['generation'].loc[d['generation'].group=='all']
    cols=['condition','n','tfidf_mean_delta_from_unsteered','cnn_mean_delta_from_unsteered','bert_mean_delta_from_unsteered','semantic_cosine_mean','hit_token_limit_n','three_judge_target_consensus_rate']
    examples=[]
    sources=read_json(QROOT/'generation/sources.json')
    outputs={(r['id'],r['condition']):r for r in d['outputs']}
    review_path=QROOT/'evaluation/assistant_qualitative_review.json'
    review=read_json(review_path) if review_path.exists() else {'cases':[],'general_notes':[]}
    notes={(r['id'],r['condition']):r['note'] for r in review['cases']}
    reference_path=QROOT/'evaluation/source_reference.json'
    references={r['id']:r for r in read_json(reference_path)['cases']} if reference_path.exists() else {}
    classification_notes={r['id']:r['note'] for r in review.get('classification_cases',[])}
    error_parts=[]
    for row in read_json(QROOT/'evaluation/classification_error_examples.json'):
        note=classification_notes.get(row['id'],'')
        error_parts.append(f'<details><summary>{escape(row["outcome"])} - {escape(row["id"])} - P+={row["p_positive"]:.4f}</summary><p class="small">{row["word_count"]} palavras; {row["original_review_tokens"]} tokens Qwen; prefixo de {row["used_review_tokens"]}; truncamento: {row["review_truncated"]}.</p><blockquote>{escape(row["text"])}</blockquote><p><b>Prefixo recebido pelo Qwen:</b></p><blockquote>{escape(row["qwen_input_prefix"])}</blockquote><p>{escape(note)}</p></details>')
    for source in sources:
        parts=[f'<details><summary>{escape(source["id"])} - {escape(source["kind"])}</summary><blockquote>{escape(source["source"])}</blockquote>']
        if source['id'] in references:
            ref=references[source['id']]
            parts.append('<p class="small">Leitura da fonte pelo assistente, preparada antes de examinar as saídas. Sem verificação externa de fatos do filme.</p><ul>'+''.join('<li>'+escape(s)+'</li>' for s in ref['facts']+ref['pitfalls'])+'</ul>')
        for condition in ['rfm_-4','rfm_-2','rfm_+0','rfm_+2','rfm_+4','prompt_negative','prompt_positive']:
            row=outputs[(source['id'],condition)]
            note=notes.get((source['id'],condition),'')
            parts.append(f'<article><b>{escape(condition)}</b><p>{escape(row["generated"])}</p><p class="small">P+ TF-IDF / CNN / BERT: {row["tfidf_p_positive"]:.3f} / {row["cnn_p_positive"]:.3f} / {row["bert_p_positive"]:.3f}; cosseno com a fonte {row["semantic_cosine_to_source"]:.3f}; limite de tokens: {row["hit_token_limit"]}.</p><p>{escape(note)}</p></article>')
        parts.append('</details>'); examples.append(''.join(parts))
    audit=pd.read_csv(QROOT/'classification/validation_prompt_audit.csv')
    audit_rows=[]
    for name in ['normal','reversed_label_order','prefix_192']:
        pred=audit[name+'_logit']>=0
        audit_rows.append({'condição':name,'n':len(audit),'acurácia':float((pred==audit.label).mean()),
                           'decisões_diferentes_do_prompt_original':int((pred!=(audit.normal_logit>=0)).sum())})
    sections=[
        '<h1>Qwen3-8B: classificação e controle do tom de resenhas</h1><p class="lead">Nestor, Letícia e Gabriel | Introdução a Deep Learning | Extensão exploratória com modelo congelado.</p>',
        '<h2>Resultados principais</h2>'+''.join('<p>'+escape(f[k])+'</p>' for k in ['classification','comparability','steering','generation','prompt_control','paired_controls','quality']),
        html_table(comp,['accuracy','f1_macro'])+image_html(folder/'models.png'),
        '<h2>Dados, probabilidades e orçamento</h2><p>Mesmo protocolo de divisões do experimento original: 19.823 treino, 2.478 validação, 2.478 calibração e 25.000 teste. Nenhum texto normalizado atravessa divisões. A RFM usa 512 exemplos do treino e 256 da validação; suas intervenções são medidas em 256 exemplos reservados do teste. As 24 resenhas de geração vêm do teste, com 40 a 150 palavras e classes balanceadas. Oito fontes neutras foram escritas para a demonstração e não são uma classe do IMDb.</p><p>O escore bruto é softmax restrito aos logits dos tokens negative e positive. Não é confiança declarada em linguagem natural nem probabilidade normalizada sobre todo o vocabulário. Uma temperatura é ajustada apenas na calibração. O prompt e o modo sem raciocínio são fixos; a fonte é truncada antes de inserir as instruções.</p>',
        f'<p>Temperatura: {d["classification_meta"]["temperature"]:.4f}. Massa média do vocabulário atribuída aos dois rótulos, nas três divisões: {d["classification_meta"]["allowed_label_mass_mean"]:.4f}. Taxa de avaliações truncadas: {d["classification_meta"]["review_truncation_rate"]:.2%}. O audit abaixo usa só validação; 192 tokens do Qwen não equivalem a 192 WordPieces do BERT.</p><p class="small">As métricas usam probabilidades recortadas para [1e-7, 1-1e-7], como no projeto original. O recorte limita a penalidade de log loss em erros de extrema confiança. Os valores originais ficam nas previsões; a temperatura é ajustada diretamente nos logits.</p>',
        html_table(pd.DataFrame(audit_rows),['acurácia'])+'<p>'+escape(f['input_audit'])+'</p><p>'+escape(f['restricted_score'])+'</p>'+image_html(folder/'calibration_length.png'),
        '<h2>Falsos positivos e falsos negativos do Qwen</h2><p>FP: rótulo negativo, previsão positiva. FN: rótulo positivo, previsão negativa. São três erros de alta confiança e três sorteados por tipo, sem representar a frequência de uma explicação. A leitura compara o texto completo com o prefixo recebido; as notas são do assistente e não demonstram causas.</p>'+''.join(error_parts),
        '<h2>RFM / AGOP e intervenção causal no programa</h2><p>Kernel Laplaciano K(x,z)=exp(-sqrt((x-z)ᵀM(x-z))/L), regressão regularizada e atualização M=média de grad f grad fᵀ. Os gradientes são do preditor kernel. A AGOP não é a Hessiana exata do LLM. A primeira iteração usa a geometria euclidiana completa; fatores posteriores representam exatamente a métrica de baixa ordem, sem PCA ou esboço aleatório.</p><p>Ativações no último token do prompt após blocos 12, 24 e 32. Largura e iteração são escolhidas pela AUC de validação por bloco; o bloco é escolhido pela mudança direcional em validação. Todas as direções têm norma unitária e compartilham uma escala aprendida somente no treino. A soma atua em todas as posições não preenchidas por padding e em cada novo token da geração. Hook zero, restauração e equivalência entre execução completa e continuação em cache são verificados.</p>',
        '<p>'+escape(f['selection'])+'</p><p>'+escape(f['recursion'])+'</p>'+image_html(folder/'selection.png')+image_html(folder/'classification_steering.png'),
        '<h2>Geração de texto, controles e avaliação</h2><p>32 fontes × 17 condições = 544 saídas. RFM em -4/-2/0/+2/+4; diferença de médias, sonda logística e três direções aleatórias em -4/+4; instruções diretas positiva e negativa sem hooks. O mesmo prompt sem alvo é usado em todas as intervenções de ativação. Decodificação gulosa, até 96 novos tokens, sem raciocínio. Todos os resultados são mantidos.</p><p>Os juízes TF-IDF, CNN e BERT avaliam os textos fora dos hooks. As mudanças são pareadas à reescrita sem steering da mesma fonte; IC de 95% por 2.000 reamostragens de fontes, descritivos e sem correção para comparações múltiplas. AUC da sonda, mudança na decisão e mudança em texto gerado são resultados diferentes.</p>',
        image_html(folder/'generation_effects.png')+image_html(folder/'generation_quality.png')+html_table(gen[cols]),
        '<h3>RFM menos cada controle: comparação pareada</h3><p>Separação direcional = [P(positivo,+4) - P(positivo,-4)]/2 por fonte; nos prompts, positivo menos negativo. Diferença positiva favorece RFM apenas nesse diagnóstico de polarização. ICs descritivos, sem ajuste para múltiplas comparações.</p>'+html_table(d['contrasts'].loc[d['contrasts'].group=='all']),
        '<h2>Leitura dos textos completos</h2><p>Abra cada fonte para comparar as mesmas condições. Os arquivos de resultados incluem também todos os controles. As notas abaixo, quando presentes, são leituras do assistente e não anotações humanas independentes.</p>'+''.join('<p>'+escape(n)+'</p>' for n in review['general_notes'])+''.join(examples),
        '<h2>Limites e uso no seminário</h2><p>'+escape(f['scope'])+'</p><p>O conjunto de geração é pequeno e contém oito fontes sintéticas. Não houve avaliação humana independente. A escolha da camada usou classificação, não geração. O modelo e o prompt são uma única configuração; resultados não generalizam automaticamente a outras tarefas ou modelos. A planilha cega de 60 exemplos permite uma etapa posterior de revisão pelo grupo, com sentimento, fluência e fatos avaliados separadamente.</p>',
        '<h2>Reprodução e fontes</h2><p>Use o notebook executado em notebooks/qwen3_8b_steering.ipynb para inspecionar resultados. O protocolo está em QWEN_EXPERIMENTO.md; a verificação está em results/qwen/verification.json. A inferência exige CUDA e os pesos públicos do Qwen3-8B; o relatório não exige carregar esses pesos.</p><p><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B: modelo oficial</a> · <a href="https://arxiv.org/abs/2505.09388">Qwen3 Technical Report</a> · <a href="https://arxiv.org/abs/2502.03708">Beaglehole et al.: RFM / AGOP</a> · <a href="https://github.com/dmbeaglehole/neural_controllers">Implementação dos autores</a> · <a href="https://ai.stanford.edu/~amaas/data/sentiment/">Large Movie Review Dataset</a></p>',
    ]
    from .qwen_didactic import html_intro
    sections[0]=html_intro(d,read_json(QROOT/'didactic_examples.json'),folder)
    sections[1]='<details class="appendix"><summary>Apêndice: métricas, controles, auditorias e todas as fontes</summary>'+sections[1].replace('Resultados principais','Resultados de apoio e limites')
    sections[-1]+='</details>'
    style='body{font:16px/1.6 system-ui,sans-serif;color:#19313c;max-width:1100px;margin:auto;padding:40px 24px;background:#fff}h1{font-size:36px;line-height:1.18}h2{margin-top:42px;color:#007d83}.lead{font-size:19px}.small{font-size:13px;color:#52656f}img{max-width:100%;height:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:12px;display:block;overflow:auto}td,th{padding:9px;border-bottom:1px solid #d5e1e4;text-align:left}th{background:#eaf3f4}blockquote{border-left:4px solid #007d83;padding:10px 18px;margin:16px 0;background:#f3f7f8}details{margin:14px 0;padding:14px;border:1px solid #d5e1e4}summary{cursor:pointer;font-weight:650}article{padding:10px 0;border-bottom:1px solid #d5e1e4}a{color:#0072b2}'
    path=ROOT/'relatorio_qwen3.html'
    path.write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen3-8B: intervenção nas ativações</title><style>'+style+'</style><body>'+''.join(sections)+'</body></html>')
    return path


def build_notebook():
    from .qwen_didactic import build_notebook as build_teaching_notebook
    return build_teaching_notebook()


def run():
    from .qwen_didactic import build_examples, teaching_figures
    d=data()
    examples=build_examples()
    f=findings(d)
    folder=figures(d)
    teaching_figures(d,folder)
    html=build_html(d,f,folder)
    notebook=build_notebook()
    from .qwen_pdf import build_pdf
    pdf=build_pdf(d,f,folder)
    inputs=[ROOT/'qwen_config.json',ROOT/'results/analysis/subgroup_metrics.csv']
    inputs += [QROOT/name for name in [
        'didactic_examples.json','pilot/metadata.json','classification/predictions.parquet','classification/metrics.json',
        'classification/metadata.json','classification/validation_prompt_audit.csv',
        'features/subset_manifest.json','rfm/selection.json','steering/layer_selection.json',
        'steering/intervention_summary.csv','steering/metadata.json','steering/directions.npz',
        'generation/sources.json','generation/conditions.json','generation/generations.json',
        'generation/metadata.json','evaluation/model_comparison.csv',
        'evaluation/generation_summary.csv','evaluation/generation_method_contrasts.csv',
        'evaluation/scored_generations.json','evaluation/scored_generations.parquet',
        'evaluation/subgroup_metrics.csv','evaluation/metadata.json',
        'evaluation/classification_error_examples.json','evaluation/assistant_qualitative_review.json']]
    if (QROOT/'evaluation/source_reference.json').exists(): inputs.append(QROOT/'evaluation/source_reference.json')
    save_json(QROOT/'report_manifest.json',{'html':str(html.relative_to(ROOT)),
              'notebook':str(notebook.relative_to(ROOT)),'pdf':str(pdf.relative_to(ROOT)),
              'sources':{str(path.relative_to(ROOT)):file_sha256(path) for path in inputs}})
    print('Qwen reports built from saved results',flush=True)
