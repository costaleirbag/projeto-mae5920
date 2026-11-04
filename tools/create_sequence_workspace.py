#!/usr/bin/env python3
"""Create an empty scientific rerun workspace without changing published results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
FILES=[
    'sequence_config.json','run_sequences.py','analyze_sequences.py','replay_sequences.py','verify_sequences.py','execute_notebook.py',
    'src/imdb_lab/__init__.py','src/imdb_lab/common.py','src/imdb_lab/models.py','src/imdb_lab/sequences.py','src/imdb_lab/sequence_report.py',
    'results/didactic/data_manifest.json','results/sequences/models_pinned.json','docs/REFERENCIAS_SEQUENCIAS.json',
    'tools/fetch_sequence_models.py','tools/sequence_audit_input.py','data/train.parquet'
]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args()
    target=args.destination.expanduser().resolve()
    if target==ROOT or target.exists(): raise ValueError('Choose a new, nonexistent destination')
    for name in FILES:
        if not (ROOT/name).is_file(): raise FileNotFoundError(f'Restore required scientific input first: {name}')
    manifest=json.loads((ROOT/'results/didactic/data_manifest.json').read_text())
    assert sha(ROOT/'data/train.parquet')==manifest['source_sha256']
    target.mkdir(parents=True)
    for name in FILES:
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
    for folder in ['notebooks','output/pdf']: (target/folder).mkdir(parents=True,exist_ok=True)
    (target/'source_workspace_manifest.json').write_text(json.dumps({'purpose':'Isolated rerun with published results preserved','files':{name:sha(target/name) for name in FILES}},indent=2)+'\n')
    print('Fresh scientific workspace created. Prepare roles, fetch pinned models, then run training.')


if __name__=='__main__': main()
