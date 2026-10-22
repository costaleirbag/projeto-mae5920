#!/usr/bin/env python3
"""Audit the saved Qwen results without loading its 8B backbone."""
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from imdb_lab.common import file_sha256, read_json, save_json
from imdb_lab.data import load_splits
from imdb_lab.factorized_rfm import factor_gradients
from imdb_lab.metrics import evaluate, fit_temperature
from imdb_lab.qwen_core import QROOT, classification_prompts, qconfig
from imdb_lab.qwen_experiment import (bootstrap_mean_ci, feature_frames, generation_conditions,
                                     generation_sources, rewriting_prompt, saved_probe_predict)


def main():
    cfg=qconfig()
    evidence={}
    frames=load_splits()
    manifest=read_json(ROOT/'data/manifest.json')
    for split,frame in frames.items():
        assert file_sha256(ROOT/f'data/{split}.parquet')==manifest['splits'][split]['sha256']
        assert frame.id.is_unique
        for other,other_frame in frames.items():
            if split!=other: assert not set(frame.text_sha256)&set(other_frame.text_sha256)
    assert len(frames['test'])==25000
    evidence['data']='Source hashes match; disjoint normalized texts across splits; official 25,000-row test retained.'

    pred=pd.read_parquet(QROOT/'classification/predictions.parquet')
    meta=read_json(QROOT/'classification/metadata.json')
    saved=read_json(QROOT/'classification/metrics.json')
    assert meta['revision']==cfg['revision'] and meta['model_id']==cfg['model_id']
    assert meta['trainable_backbone_parameters']==0 and meta['calibration_fit_split']=='calibration'
    assert set(pred.split)=={'validation','calibration','test'}
    calibration=pred.loc[pred.split=='calibration']
    temperature=fit_temperature(calibration.logit,calibration.label)
    assert np.isclose(temperature,meta['temperature'],atol=1e-10)
    for split in ['validation','calibration','test']:
        rows=pred.loc[pred.split==split].set_index('id').reindex(frames[split].id)
        assert rows.index.is_unique and len(rows)==len(frames[split])
        assert np.array_equal(rows.label,frames[split].label)
        assert np.array_equal(rows.prediction,rows.logit>=0)
        np.testing.assert_allclose(rows.p_raw,expit(rows.logit),atol=1e-12)
        np.testing.assert_allclose(rows.p_positive,expit(rows.logit/temperature),atol=1e-12)
        assert rows.allowed_label_mass.between(0,1.000001).all()
        assert (rows.used_review_tokens<=cfg['max_review_tokens']).all()
        assert np.array_equal(rows.review_truncated,rows.original_review_tokens>cfg['max_review_tokens'])
        for view,col in [('raw','p_raw'),('calibrated','p_positive')]:
            actual=evaluate(rows.label,rows[col])
            for key in actual:
                if actual[key] is not None:
                    assert np.isclose(actual[key],saved[split][view][key],atol=1e-10),(split,view,key)
    evidence['classification']='All 29,956 IDs, labels, decisions, probabilities and split metrics recomputed; temperature refitted from calibration only.'

    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'artifacts/qwen/tokenizer',local_files_only=True)
    tokenizer.padding_side='left'
    # Reconstruct metadata with the saved tokenizer, without downloading weights.
    sample=pd.concat([pred.groupby('split',sort=True).head(3),pred.groupby('split',sort=True).tail(3)])
    for _,row in sample.iterrows():
        source=frames[row.split].set_index('id').loc[row.id,'clean_text']
        _,info=classification_prompts(tokenizer,[source])
        for key,value in info[0].items(): assert row[key]==value
    audit=pd.read_csv(QROOT/'classification/validation_prompt_audit.csv')
    expected=feature_frames()
    assert audit.id.tolist()==expected['validation'].id.tolist()
    assert all(np.isfinite(audit[c+'_logit']).all() for c in ['normal','reversed_label_order','prefix_192'])
    evidence['input_and_prompt']='Input truncation metadata reconstructed with saved tokenizer; prompt-order and prefix audits contain validation IDs only.'

    subset=read_json(QROOT/'features/subset_manifest.json')
    features=np.load(QROOT/'features/activations.npz')
    assert subset['dataset_manifest_sha256']==file_sha256(ROOT/'data/manifest.json')
    assert subset['revision']==cfg['revision']
    for split in ['train','validation','test']:
        assert subset[split]==expected[split].id.tolist()==features[split+'_ids'].tolist()
        assert np.array_equal(features[split+'_labels'],expected[split].label)
        for other in ['train','validation','test']:
            if split!=other: assert not set(subset[split])&set(subset[other])
        for layer in cfg['layers']:
            assert features[f'{split}_layer_{layer}'].shape==(len(expected[split]),4096)
            assert np.isfinite(features[f'{split}_layer_{layer}']).all()
    selection=read_json(QROOT/'steering/layer_selection.json')
    rfm_selection=read_json(QROOT/'rfm/selection.json')
    for candidate in rfm_selection['candidates']:
        layer=candidate['layer']
        path=QROOT/f'rfm/rfm_layer_{layer}.npz'
        auc=roc_auc_score(features['validation_labels'],saved_probe_predict(path,features[f'validation_layer_{layer}']))
        assert np.isclose(auc,candidate['validation_auc'])
        choices=[r for r in rfm_selection['all_candidates'] if r['layer']==layer]
        assert candidate['validation_auc']==max(r['validation_auc'] for r in choices)
        for fit in choices:
            best=max(fit['history'],key=lambda h:(h['validation_auc'],-h['validation_mse']))
            assert best['iteration']==fit['selected_iteration']
    best=max(selection['all_candidates'],key=lambda r:(r['validation_directional_response'],r['validation_auc']))
    assert best==selection['selected']
    layer=best['layer']
    selected_path=QROOT/'steering/selected_rfm.npz'
    assert file_sha256(selected_path)==file_sha256(QROOT/f'rfm/rfm_layer_{layer}.npz')
    rfm=np.load(selected_path)
    metric=None if bool(rfm['metric_is_identity']) else rfm['metric_factor']
    gradients=factor_gradients(rfm['centers'],rfm['centers'],metric,rfm['alpha'],float(rfm['bandwidth']))
    np.testing.assert_allclose(gradients/np.sqrt(len(gradients)),rfm['agop_factor'],atol=1e-9,rtol=1e-6)
    eigvals,eigvectors=np.linalg.eigh(rfm['agop_factor']@rfm['agop_factor'].T)
    np.testing.assert_allclose(eigvals,rfm['eigenvalues'],atol=1e-10,rtol=1e-6)
    top=rfm['agop_factor'].T@eigvectors[:,-1]; top/=np.linalg.norm(top)
    assert abs(top@rfm['direction'])>.999999
    assert eigvals.min()>-1e-10
    x=features[f'train_layer_{layer}']
    assert np.cov(x@rfm['direction'],features['train_labels'])[0,1]>0
    assert np.isclose(np.std(x@rfm['direction'],ddof=1),float(rfm['projection_std']))
    heldout=roc_auc_score(features['test_labels'],saved_probe_predict(selected_path,features[f'test_layer_{layer}']))
    assert np.isclose(heldout,selection['heldout_probe_auc'])
    evidence['rfm']='Subset roles checked; validation probe AUC and selections recomputed; exact saved AGOP factor recomputed from gradients; PSD, principal direction and train-only scale verified.'

    directions=np.load(QROOT/'steering/directions.npz')
    methods=[k for k in directions.files if k not in {'layer','scale'}]
    assert len(methods)==3+cfg['random_controls'] and int(directions['layer'])==layer
    scale=float(directions['scale'])
    for method in methods: assert np.isclose(np.linalg.norm(directions[method]),1)
    np.testing.assert_allclose(directions['rfm_agop'],rfm['direction'],atol=1e-12)
    summary=pd.read_csv(QROOT/'steering/intervention_summary.csv')
    individuals=pd.read_parquet(QROOT/'steering/interventions.parquet')
    assert len(summary)==2*len(methods)*len(cfg['intervention_alphas'])
    assert not individuals.duplicated(['split','id','method','alpha']).any()
    for _,row in summary.iterrows():
        part=individuals.loc[(individuals.split==row.split)&(individuals.method==row.method)&(individuals.alpha==row.alpha)]
        assert set(part.id)==set(subset[row.split]) and len(part)==row.n
        assert part.input_text_unchanged.all()
        np.testing.assert_allclose(part.p_positive_raw,expit(part.steered_logit),atol=1e-12)
        assert np.isclose(row.mean_p_positive_raw,part.p_positive_raw.mean())
        assert np.isclose(row.mean_delta_p_raw,(part.p_positive_raw-expit(part.base_logit)).mean())
        flips=((part.steered_logit>=0)!=(part.base_logit>=0))
        assert flips.sum()==row.prediction_flip_n
        assert np.isclose(flips.mean(),row.prediction_flip_fraction)
        assert np.isclose(row.perturbation_l2_per_token,abs(row.alpha)*scale)
        if row.alpha==0: assert not flips.any() and np.array_equal(part.steered_logit,part.base_logit)
    smeta=read_json(QROOT/'steering/metadata.json')
    assert smeta['hook_restoration_exact'] and smeta['nonzero_hook_changes_logits']
    assert max(smeta['identity_max_error'].values())<1e-4 and smeta['full_hook_cached_max_error']<1e-4
    assert smeta['subset_manifest_sha256']==file_sha256(QROOT/'features/subset_manifest.json')
    evidence['classification_interventions']='All norm-matched methods and coefficients retained; per-ID changes match summaries; zero identity, actual full-hook/cache equivalence and restoration recorded.'

    comparison=pd.read_csv(QROOT/'evaluation/model_comparison.csv').set_index('model')
    for model in ['tfidf','cnn','bert','qwen3_8b']:
        all_rows=pred if model=='qwen3_8b' else pd.read_parquet(ROOT/f'results/{model}/predictions.parquet')
        rows=all_rows.loc[all_rows.split=='test'].set_index('id').reindex(frames['test'].id)
        assert len(rows)==25000 and np.array_equal(rows.label,frames['test'].label)
        actual=evaluate(rows.label,rows.p_positive)
        for key,value in actual.items():
            if value is not None: assert np.isclose(value,comparison.loc[model,key],atol=1e-10)
    evidence['comparison']='All four reported system metrics recomputed on the exact same official test IDs.'

    sources=read_json(QROOT/'generation/sources.json')
    conditions=read_json(QROOT/'generation/conditions.json')
    assert sources==generation_sources() and conditions==generation_conditions()
    generated=read_json(QROOT/'generation/generations.json')
    assert len(generated)==len(sources)*len(conditions)==544
    by={(r['id'],r['condition']):r for r in generated}
    assert len(by)==len(generated)
    source_map={r['id']:r for r in sources}
    for condition in conditions:
        assert {r['id'] for r in generated if r['condition']==condition['condition']}==set(source_map)
    for row in generated:
        source=source_map[row['id']]
        assert row['source']==source['source'] and row['kind']==source['kind']
        assert row['generated']==tokenizer.decode(row['output_token_ids'],skip_special_tokens=True).strip()
        assert row['output_tokens']==len(row['output_token_ids'])<=cfg['max_new_tokens']
        target=row['target'] if row['method']=='prompt_only' else 'none'
        prompt=rewriting_prompt(tokenizer,row['source'],target)
        assert hashlib.sha256(prompt.encode()).hexdigest()==row['prompt_sha256']
        expected_norm=abs(row['alpha'])*scale if row['uses_hook'] else 0
        assert np.isclose(row['perturbation_l2_per_token'],expected_norm)
        assert row['hook_calls_in_batch']==row['prefill_hook_calls_in_batch']+row['decode_hook_calls_in_batch']
        if row['uses_hook']:
            assert row['prefill_hook_calls_in_batch']>0 and row['decode_hook_calls_in_batch']>0
        else: assert row['hook_calls_in_batch']==0
    gmeta=read_json(QROOT/'generation/metadata.json')
    assert gmeta['zero_generation_replay_exact'] and gmeta['all_hooks_removed'] and gmeta['backbone_frozen']
    assert gmeta['trainable_backbone_parameters']==0 and not gmeta['thinking_enabled']
    pilot=read_json(QROOT/'pilot/metadata.json')
    for key in ['zero_hook_exact','hook_restoration_exact','nonzero_hook_changes_logits','zero_generation_exact','nonzero_generation_cached_vs_full_exact']:
        assert pilot[key]
    evidence['generation']='All 32 x 17 outputs retained; token IDs decode exactly; source/prompts and norm-matched conditions reconstructed; prefill and decode hooks active; zero replay and nonzero cached/full generation verified on real 8B.'

    scored=read_json(QROOT/'evaluation/scored_generations.json')
    evalmeta=read_json(QROOT/'evaluation/metadata.json')
    assert len(scored)==544 and not evalmeta['judging_hooks_used']
    assert not evalmeta['human_annotations_completed']
    for kind,path in {'tfidf':'artifacts/tfidf/model.joblib','cnn':'artifacts/cnn/model.pt','bert':'artifacts/bert/model.safetensors'}.items():
        assert evalmeta['judge_checkpoint_sha256'][kind]==file_sha256(ROOT/path)
    scored_frame=pd.DataFrame(scored)
    sums=pd.read_csv(QROOT/'evaluation/generation_summary.csv')
    for _,row in sums.iterrows():
        part=scored_frame.loc[scored_frame.condition==row.condition]
        if row.group!='all': part=part.loc[part.kind==row.group]
        assert len(part)==row.n
        assert row.hit_token_limit_n==part.hit_token_limit.sum()
        for judge in ['tfidf','cnn','bert']:
            assert part[judge+'_p_positive'].between(0,1).all()
            delta=part[judge+'_p_positive']-part[judge+'_unsteered_p_positive']
            assert np.isclose(delta.mean(),row[judge+'_mean_delta_from_unsteered'])
            lo,hi=bootstrap_mean_ci(delta,cfg['seed'])
            assert np.isclose(lo,row[judge+'_delta_ci_low']) and np.isclose(hi,row[judge+'_delta_ci_high'])
    worksheet=pd.read_csv(QROOT/'evaluation/qualitative_review_blind.csv')
    contrasts=pd.read_csv(QROOT/'evaluation/generation_method_contrasts.csv')
    assert len(contrasts)==3*3*(3+cfg['random_controls'])
    for _,row in contrasts.iterrows():
        part=scored_frame if row.group=='all' else scored_frame.loc[scored_frame.kind==row.group]
        pivot=part.pivot(index='id',columns='condition',values=row.judge+'_p_positive')
        plus,minus=('prompt_positive','prompt_negative') if row.control=='prompt_only' else (row.control+'_+4',row.control+'_-4')
        rfm_sep=(pivot['rfm_+4']-pivot['rfm_-4'])/2
        control_sep=(pivot[plus]-pivot[minus])/2
        difference=rfm_sep-control_sep
        assert np.isclose(difference.mean(),row.rfm_minus_control)
        lo,hi=bootstrap_mean_ci(difference,cfg['seed'])
        assert np.isclose(lo,row.ci_low) and np.isclose(hi,row.ci_high)
    assert len(worksheet)==60 and worksheet.blind_id.is_unique
    assert worksheet[['sentiment_negative_neutral_positive','fluency_1_to_5','facts_preserved_1_to_5','notes']].isna().all().all()
    key=pd.read_csv(QROOT/'evaluation/qualitative_review_key.csv')
    joined=worksheet.merge(key,on='blind_id',validate='one_to_one')
    assert len(joined)==60
    for _,row in joined.iterrows():
        original=by[(row.id,row.condition)]
        assert row.source==original['source'] and row.generated==original['generated']
    qualitative=read_json(QROOT/'evaluation/assistant_qualitative_review.json')
    assert qualitative['human_annotations']==False and qualitative['provenance']=='assistant qualitative reading; not independent human annotation'
    assert qualitative['generations_sha256']==file_sha256(QROOT/'generation/generations.json')
    assert qualitative['classification_errors_sha256']==file_sha256(QROOT/'evaluation/classification_error_examples.json')
    assert len(qualitative['cases'])==60
    assert {(r['id'],r['condition']) for r in qualitative['cases']}==set(zip(key.id,key.condition))
    for row in qualitative['cases']:
        original=by[(row['id'],row['condition'])]
        assert row['generated_sha256']==hashlib.sha256(original['generated'].encode()).hexdigest()
    assert qualitative['source_reference_sha256']==file_sha256(QROOT/'evaluation/source_reference.json')
    references=read_json(QROOT/'evaluation/source_reference.json')
    assert not references['human_annotations']
    assert {r['id'] for r in references['cases']}==set(key.id)
    for row in references['cases']:
        assert row['source_sha256']==hashlib.sha256(source_map[row['id']]['source'].encode()).hexdigest()
    evidence['evaluation']='Independent frozen judge checkpoint hashes verified; all paired summary means and bootstrap intervals recomputed; blank human worksheet distinguished from assistant qualitative reading.'

    import nbformat
    from pypdf import PdfReader
    nb=nbformat.read(ROOT/'notebooks/qwen3_8b_steering.ipynb',as_version=4)
    cells=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in cells)
    assert not any(o.output_type=='error' for c in cells for o in c.outputs)
    assert (ROOT/'relatorio_qwen3.html').stat().st_size>10000
    report_manifest=read_json(QROOT/'report_manifest.json')
    for path,digest in report_manifest['sources'].items(): assert file_sha256(ROOT/path)==digest
    pdf=PdfReader(ROOT/'relatorio_qwen3.pdf')
    assert 7<=len(pdf.pages)<=9 and all(len(page.extract_text())>100 for page in pdf.pages)
    for stage in ['pilot','features','rfm','classify','steer','generate','evaluate','report']:
        status=read_json(ROOT/f'logs/qwen_{stage}_status.json')
        assert status['status']=='complete'
        assert status['config_sha256']==file_sha256(ROOT/'qwen_config.json')
    evidence['deliverables']=f'{len(cells)} notebook code cells executed without errors; standalone HTML and {len(pdf.pages)}-page PDF; every experiment stage complete.'
    save_json(QROOT/'verification.json',{'status':'passed','evidence':evidence,
              'limitations':['No independent human annotations.','No audit of possible IMDb pretraining contamination.',
                             'Exploratory extension reuses the official test already inspected in the previous stage.',
                             'Single model/prompt/configuration; small generation set.','Automated sentiment and semantic proxies do not establish human fluency or factual preservation.']})
    print('Qwen scientific result verification passed',flush=True)


if __name__=='__main__': main()
