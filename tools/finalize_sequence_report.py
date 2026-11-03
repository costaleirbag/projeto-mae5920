#!/usr/bin/env python3
"""Execute the reading notebook and check exact reported scientific values."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=ROOT/'results/sequences';record=json.loads((base/'report_manifest.json').read_text())
    for kind in ['pdf','html']: assert sha(ROOT/record[kind])==record[kind+'_sha256']
    for name,digest in record['figures'].items(): assert sha(ROOT/name)==digest
    subprocess.run([sys.executable,str(ROOT/'execute_notebook.py'),'--notebook',record['notebook']],cwd=ROOT,check=True)
    import nbformat
    from pypdf import PdfReader
    nb=nbformat.read(ROOT/record['notebook'],as_version=4);cells=[c for c in nb.cells if c.cell_type=='code']
    assert len(cells)==13 and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type=='error' for c in cells for o in c.outputs)
    pdf=PdfReader(ROOT/record['pdf']);assert len(pdf.pages)==11
    assert all(len(p.extract_text())>350 for p in pdf.pages)
    text=' '.join(p.extract_text() for p in pdf.pages)
    for term in ['TextCNN' if 'TextCNN' in text else 'CNN','BERT','DistilBERT','ModernBERT','1.538','2.048','87']:
        assert term in text
    s=json.loads((base/'summary.json').read_text())
    for r in s['groups']:
        if r['group']=='encoder':
            assert f"{100*r['accuracy_mean']:.2f}%".replace('.',',') in text
            assert f"{r['ce_mean']:.3f}" in text
    record.update(notebook_execution='passed',executed_code_cells=len(cells),notebook_sha256=sha(ROOT/record['notebook']),
                  pdf_pages=len(pdf.pages),builder_sha256=sha(ROOT/'src/imdb_lab/sequence_report.py'),
                  verification_sha256=sha(base/'verification.json'),summary_sha256=sha(base/'summary.json'))
    (base/'report_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('CNN/encoder deliverables verified: 13 executed cells, 11 pages, exact metrics and figure hashes')


if __name__=='__main__':main()
