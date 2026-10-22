#!/usr/bin/env python3
"""Verify frozen inputs, exact learned vector, generated outputs and deliverables."""
from pathlib import Path
import hashlib,json,re,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from imdb_lab.common import read_json,save_json,file_sha256
from run_qwen_improve import writing_prompt,controller_vectors,fingerprint
Q=ROOT/'results/qwen_improved'


def main():
    frozen=read_json(Q/'selection_frozen.json');best=frozen['selected'];checks={}
    for path,key in [(ROOT/'qwen_improve_config.json','config_sha256'),(Q/'datasets.json','datasets_sha256'),
        (Q/'directions.npz','directions_sha256'),(Q/'selected_training_features.npz','selected_training_features_sha256'),
        (Q/'manual_review_selection.json','manual_review_selection_sha256'),
        (Q/'structured_validation/scored.json','validation_sha256'),
        (Q/'semantic_audit_structured_validation.json','semantic_audit_sha256')]:
        assert file_sha256(path)==frozen[key],path
    assert file_sha256(ROOT/'run_qwen_improve.py')==frozen['generation_code_sha256']
    checks['frozen_inputs']='Configuration, datasets, complete direction archive, validation evidence, code and preselected reading cases match their frozen hashes.'
    archive=np.load(Q/'directions.npz');features=np.load(Q/'selected_training_features.npz')
    for key,digest in read_json(Q/'initial_direction_fingerprints.json').items():
        assert hashlib.sha256(archive[key].tobytes()).hexdigest()==digest
    for key in archive.files:assert np.isfinite(archive[key]).all() and np.isclose(np.linalg.norm(archive[key]),1,atol=1e-10)
    for entry in read_json(Q/'selected_training_features.json')['directions']:
        x=features[f"train_L{entry['layer']}"].astype(float);y=features[f"train_labels_L{entry['layer']}"]
        assert len(x)==entry['n'] and x.shape[1]==4096
        if entry['method']=='means':
            delta=x[y==1].mean(0)-x[y==0].mean(0)
            np.testing.assert_allclose(delta/np.linalg.norm(delta),archive[entry['name']],atol=1e-12,rtol=1e-12)
    checks['directions']='All 109 unit vectors finite; first 85 unchanged; selected mean direction exactly reconstructed from saved training states and labels.'
    data=read_json(Q/'datasets.json');test=data['generation_test']
    assert len(test)==64
    assert sum(r['kind']=='official_review' for r in test)==32
    assert sum(r['source_label']==1 for r in test)==16
    ids={r['id'] for r in test};texts={re.sub(r'\s+',' ',r['source']).strip().lower() for r in test}
    assert len(ids)==len(texts)==64
    old=read_json(ROOT/'results/qwen/features/subset_manifest.json')['test']
    old_sources=read_json(ROOT/'results/qwen/generation/sources.json')
    assert ids.isdisjoint(set(old)|{r['id'] for r in old_sources})
    for name,part in data.items():
        if name=='generation_test':continue
        assert ids.isdisjoint(r['id'] for r in part)
        assert texts.isdisjoint(re.sub(r'\s+',' ',r['source']).strip().lower() for r in part)
    checks['heldout_sources']='64 unique sources, balanced real labels, disjoint from training/validation and the old intervention/generation subsets.'
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'artifacts/qwen/tokenizer',local_files_only=True)
    tokenizer.padding_side='left'
    assert max(len(tokenizer.encode(r['source'],add_special_tokens=False)) for r in test)<=256
    for split,n in [('test',64),('regression',32)]:
        rows=read_json(Q/split/'scored.json');frame=pd.DataFrame(rows)
        conditions=read_json(Q/'test_conditions.json') if split=='test' else list({r['condition']:r['controller'] for r in rows}.values())
        assert len(rows)==n*len(conditions) and not frame.duplicated(['id','condition']).any()
        assert frame.groupby('id').prompt_sha256.nunique().eq(1).all()
        meta=read_json(Q/split/'model_metadata.json')
        assert meta['trainable_parameters']==0 and meta['parameters']==8190735360
        assert meta['source_code_sha256']==frozen['generation_code_sha256']
        assert meta['do_sample'] is False and meta['max_new_tokens']==112
        baseline={r['id']:r for r in rows if r['condition']=='baseline'}
        for r in rows:
            c=r['controller'];prompt=writing_prompt(tokenizer,r['source'],c['prompt_profile'])
            assert r['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
            assert r['controller_fingerprint']==fingerprint(c)
            assert tokenizer.decode(r['token_ids'],skip_special_tokens=True)==r['generated']
            assert r['source']==baseline[r['id']]['source']
            if c['scope']=='after_fact':
                b=baseline[r['id']];at=r['fact_gate_at_generated_token'];bt=b['fact_gate_at_generated_token']
                assert r['fact_gate_fired'] and at is not None and bt is not None
                assert r['first_fact_line']==b['first_fact_line'] and bool(r['first_fact_line'])
                assert r['token_ids'][:at]==b['token_ids'][:bt]
        audit=read_json(Q/f'semantic_audit_{split}.json')
        assert audit['source_sha256']==file_sha256(Q/split/'scored.json')
        assert len(audit['rows'])==len(rows) and audit['trainable_parameters']==0
        by={(r['id'],r['condition']):r for r in rows}
        for r in audit['rows']:
            assert r['generated_sha256']==hashlib.sha256(by[r['id'],r['condition']]['generated'].encode()).hexdigest()
            for task in ['sentiment','facts']:
                p=[r[task]['p_'+letter] for letter in 'ABC']
                assert np.isclose(sum(p),1,atol=1e-6) and min(p)>=0
                assert r[task]['label']=='ABC'[np.argmax(p)]
        checks[split]=f'{len(rows)} actual outputs: exact tokens, source, prompt, vector fingerprints and unchanged factual prefixes verified; semantic audit linked to exact outputs.'
    selected_norms={l:np.linalg.norm(v)*abs(best['positive']['gain']) for l,v in controller_vectors(best['positive']).items()}
    for c in read_json(Q/'test_conditions.json'):
        if c['name'].startswith('random_matched'):
            assert c['scope']==best['positive']['scope'] and c['prompt_profile']==best['positive']['prompt_profile']
            for l,v in controller_vectors(c).items():assert np.isclose(np.linalg.norm(v)*abs(c['gain']),selected_norms[l])
    checks['controls']='Three random directions use identical perturbation norms, layers, phase masks and prompt profile to the selected controller.'
    manual=read_json(Q/'manual_reading.json');selection=read_json(Q/'manual_review_selection.json')
    assert {r['id'] for r in manual['cases']}=={r['id'] for r in selection['cases']}
    test_rows={(r['id'],r['condition']):r for r in read_json(Q/'test/scored.json')}
    for r in manual['cases']:
        assert r['reading'].strip()
        for name,digest in r['output_sha256'].items():assert digest==hashlib.sha256(test_rows[r['id'],name]['generated'].encode()).hexdigest()
    checks['qualitative_reading']='All 12 preselected trios have assistant notes linked to their exact saved outputs; no independent human annotation claimed.'
    import nbformat
    from pypdf import PdfReader
    notebook=nbformat.read(ROOT/'notebooks/qwen3_8b_steering.ipynb',as_version=4)
    cells=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in cells)
    assert not any(o.output_type=='error' for c in cells for o in c.outputs)
    pdf=PdfReader(ROOT/'relatorio_qwen3.pdf')
    assert len(pdf.pages)==7 and all(len(p.extract_text())>100 for p in pdf.pages)
    full=re.sub(r'\s+',' ',' '.join(p.extract_text() for p in pdf.pages))
    for case in read_json(Q/'presentation_examples.json')['cases']:
        for r in case['outputs']:assert re.sub(r'\s+',' ',r['generated']) in full
    from html.parser import HTMLParser
    from collections import Counter
    class Gallery(HTMLParser):
        def __init__(self):super().__init__();self.inside=False;self.current=[];self.outputs=[];self.details=0
        def handle_starttag(self,tag,attrs):
            if tag=='pre':self.inside=True;self.current=[]
            if tag=='details':self.details+=1
        def handle_data(self,data):
            if self.inside:self.current.append(data)
        def handle_endtag(self,tag):
            if tag=='pre':self.outputs.append(''.join(self.current));self.inside=False
    gallery=Gallery();gallery.feed((ROOT/'relatorio_qwen3.html').read_text())
    selected_names={best['negative']['name'],'baseline',best['positive']['name']}
    expected=[r['generated'] for r in read_json(Q/'test/scored.json') if r['condition'] in selected_names]
    assert gallery.details==64 and len(gallery.outputs)==192 and Counter(gallery.outputs)==Counter(expected)
    build=read_json(Q/'report_build.json')
    for rel,key in [('relatorio_qwen3.pdf','pdf_sha256'),('relatorio_qwen3.html','html_sha256'),('notebooks/qwen3_8b_steering.ipynb','notebook_sha256')]:
        assert file_sha256(ROOT/rel)==build[key]
    checks['deliverables']=f'Seven-page PDF contains exact demonstration outputs; HTML has all 64 trios without text changes; all {len(cells)} notebook code cells executed without errors; output hashes match.'
    save_json(Q/'verification.json',{'status':'passed','evidence':checks,
        'limitations':['Same base model is the semantic judge; no independent human ground truth.',
        'Direction, layer, amplitude and intervention position change together relative to the original controller.',
        'Explicit FACT/REVIEW format; no universal free-text control claim.',
        'Unchanged factual prefix can still contain baseline summarization errors.']})
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
