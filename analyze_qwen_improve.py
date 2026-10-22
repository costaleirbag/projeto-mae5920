#!/usr/bin/env python3
"""Paired held-out evaluation of generation steering; no model inference."""
from pathlib import Path
import sys
import json
import hashlib
import re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from imdb_lab.common import read_json,save_json,file_sha256
Q=ROOT/'results/qwen_improved'


def interval(values,seed=20261021):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    boots=a[rng.integers(0,len(a),size=(5000,len(a)))].mean(1)
    return {'mean':float(a.mean()),'n':len(a),'ci_low':float(np.quantile(boots,.025)),
            'ci_high':float(np.quantile(boots,.975))}


def proportion_interval(values):
    a=np.asarray(values,dtype=float);n=len(a);p=float(a.mean());z=1.959963984540054
    denominator=1+z*z/n;center=(p+z*z/(2*n))/denominator
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return {'mean':p,'n':n,'ci_low':max(0.,float(center-half)),
            'ci_high':min(1.,float(center+half)),'interval_method':'Wilson 95%'}


def semantic_analysis(rows,conditions,sources,selected):
    path=Q/'semantic_audit_test.json';audit=read_json(path)
    assert audit['source_sha256']==file_sha256(Q/'test/scored.json')
    generated={(r['id'],r['condition']):r for r in rows}
    assert len(audit['rows'])==len(generated)
    for r in audit['rows']:
        assert r['generated_sha256']==hashlib.sha256(generated[r['id'],r['condition']]['generated'].encode()).hexdigest()
    ids=[s['id'] for s in sources]
    parts={c['name']:{r['id']:r for r in audit['rows'] if r['condition']==c['name']} for c in conditions}
    summary=[]
    for c in conditions:
        for group,kind in [('all',None),('real','official_review'),('neutral','synthetic_neutral_plot')]:
            part=[parts[c['name']][s['id']] for s in sources if kind is None or s['kind']==kind]
            counts={letter:sum(r['sentiment']['label']==letter for r in part) for letter in 'ABC'}
            target='A' if c['gain']>0 else 'B'
            summary.append({'condition':c['name'],'group':group,'n':len(part),'opinion_counts':counts,
                'target_rate':counts[target]/len(part) if c['gain'] else None,
                'supported_facts_n':sum(r['facts']['label']=='A' for r in part),
                'unsupported_facts_n':sum(r['facts']['label']=='B' for r in part),
                'unclear_facts_n':sum(r['facts']['label']=='C' for r in part)})
    pairs=[('selected',selected['negative']['name'],selected['positive']['name']),
           ('original','original_exact_-1_all','original_exact_+1_all')]
    pairs += [(f'random_{i}',f'random_matched_{i}_-1',f'random_matched_{i}_+1') for i in range(1,4)]
    comparisons={}
    for group,kind in [('all',None),('real','official_review'),('neutral','synthetic_neutral_plot')]:
        chosen=[s['id'] for s in sources if kind is None or s['kind']==kind]
        joint={name:np.array([parts[negative][i]['sentiment']['label']=='B' and parts[positive][i]['sentiment']['label']=='A' for i in chosen],dtype=float)
               for name,negative,positive in pairs}
        comparisons[group]={'joint_target_success':{name:proportion_interval(value) for name,value in joint.items()},
            'paired_improvement_over_original':interval(joint['selected']-joint['original']),
            'paired_improvement_over_random_mean':interval(joint['selected']-np.mean([joint[f'random_{i}'] for i in range(1,4)],axis=0))}
    result={'provenance':audit['provenance'],'scope':audit.get('field_scope'),
        'caveat':'Automatic rubric diagnostic by the same base model, not accuracy against independent human ground truth.',
        'intervals':'Wilson intervals for individual binary rates; 5,000 paired source bootstrap resamples for differences. Descriptive intervals for this fixed source mixture.',
        'conditions':summary,'paired_comparisons':comparisons,
        'source_sha256':file_sha256(path)}
    save_json(Q/'semantic_analysis.json',result)
    return result


def main():
    frozen=read_json(Q/'selection_frozen.json');sources=read_json(Q/'datasets.json')['generation_test']
    conditions=read_json(Q/'test_conditions.json');rows=read_json(Q/'test/scored.json')
    assert file_sha256(Q/'directions.npz')==frozen['directions_sha256']
    assert file_sha256(Q/'datasets.json')==frozen['datasets_sha256']
    df=pd.DataFrame(rows);assert len(df)==len(sources)*len(conditions)
    assert not df.duplicated(['id','condition']).any()
    assert df.groupby('id').prompt_sha256.nunique().eq(1).all()
    selected=frozen['selected'];ids=[r['id'] for r in sources]
    parts={name:part.set_index('id').reindex(ids) for name,part in df.groupby('condition')}
    baseline=parts['baseline'];negative=parts[selected['negative']['name']];positive=parts[selected['positive']['name']]
    oldminus=parts['original_exact_-1_all'];oldplus=parts['original_exact_+1_all']
    assert all(part.source.tolist()==baseline.source.tolist() for part in parts.values())
    from run_qwen_improve import check_fact_prefixes
    check_fact_prefixes(rows,conditions,Q/'test_prefix_checks.json')
    semantic=semantic_analysis(rows,conditions,sources,selected)
    results={}
    groups={'all':np.ones(len(ids),dtype=bool),'real':baseline.kind.to_numpy()=='official_review',
            'neutral':baseline.kind.to_numpy()=='synthetic_neutral_plot'}
    for group,mask in groups.items():
        group_results={}
        for judge in ['tfidf','cnn','bert']:
            col=judge+'_p_positive'
            new_delta=(positive[col]-negative[col]).to_numpy()[mask]
            old_delta=(oldplus[col]-oldminus[col]).to_numpy()[mask]
            group_results[judge]={'new_separation':interval(new_delta),
                'old_separation':interval(old_delta),'paired_improvement':interval(new_delta-old_delta),
                'positive_vs_zero':interval((positive[col]-baseline[col]).to_numpy()[mask]),
                'negative_vs_zero':interval((negative[col]-baseline[col]).to_numpy()[mask]),
                'positive_target_rate':float((positive[col].to_numpy()[mask]>=.5).mean()),
                'negative_target_rate':float((negative[col].to_numpy()[mask]<.5).mean())}
        results[group]=group_results
    conditions_summary=[]
    for name,part in parts.items():
        positive_target=float(part.gain.iloc[0])>0
        hit=part[['tfidf_p_positive','cnn_p_positive','bert_p_positive']].ge(.5)
        consensus=hit.all(axis=1) if positive_target else (~hit).all(axis=1)
        opinions=[re.search(r'(?is)\bREVIEW\s*:(.*)$',t) for t in part.generated]
        opinion_texts=[m.group(1).strip() for m in opinions if m]
        conditions_summary.append({'condition':name,'n':len(part),'gain':float(part.gain.iloc[0]),
            'three_judge_target_consensus':float(consensus.mean()) if name!='baseline' else None,
            'quality_ok':float(part.quality_ok.mean()),'empty_n':int(part['empty'].sum()),
            'hit_token_limit_n':int(part.hit_token_limit.sum()),'mean_repetition':float(part.repetition.mean()),
            'neutral_entity_retention':float(part.loc[part.kind=='synthetic_neutral_plot','entity_retention'].mean()),
            'opinion_field_missing_n':len(part)-len(opinion_texts),'distinct_opinion_n':len(set(opinion_texts)),
            'distinct_generation_fraction':float(part.generated.nunique()/len(part))})
    # Choose display cases by input order, never by outcomes.
    examples=[next(r for r in sources if r['kind']=='synthetic_neutral_plot')]
    examples += [next(r for r in sources if r['kind']=='official_review' and r['source_label']==y) for y in [0,1]]
    selected_rows=[]
    raw_by={(r['id'],r['condition']):r for r in rows}
    for s in examples:
        selected_rows.append({**s,'outputs':[raw_by[s['id'],name] for name in [
            selected['negative']['name'],'baseline',selected['positive']['name']]]})
    save_json(Q/'presentation_examples.json',{'selection':'First new test plot and first new official review of each label, in dataset order; no outcome selection.','cases':selected_rows})
    save_json(Q/'test_analysis.json',{'results':results,'conditions':conditions_summary,
        'bootstrap':'5,000 paired resamples of source IDs; descriptive 95% intervals; sources are the independent unit.',
        'selection':selected,'tests_used_for_settings':False,
        'sources':{str(p.relative_to(ROOT)):file_sha256(p) for p in [Q/'selection_frozen.json',Q/'test/scored.json',Q/'test_conditions.json',Q/'datasets.json']}})
    print(json.dumps(results['all'],indent=2),flush=True)
    print('Semantic joint-target success:',json.dumps(semantic['paired_comparisons']['all'],indent=2))
    print('Same source and prompt verified for every condition; complete held-out paired analysis saved.')

if __name__=='__main__':main()
