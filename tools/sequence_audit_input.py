#!/usr/bin/env python3
"""Withhold the audit input outside a run workspace, then release after freeze."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['withhold','restore'])
    p.add_argument('--file',type=Path,required=True,help='A file outside the run workspace')
    args=p.parse_args();external=args.file.expanduser().resolve()
    if external.is_relative_to(ROOT): raise ValueError('The held-out file must stay outside the run workspace')
    base=ROOT/'results/sequences';manifest=json.loads((base/'data_manifest.json').read_text())
    expected=manifest['roles']['audit']['input_sha256'];canonical=ROOT/'.cache/sequence_inputs/audit.parquet'
    if args.action=='withhold':
        if (base/'protocol_started.json').exists(): raise ValueError('Withhold the audit before starting the protocol')
        assert canonical.is_file() and sha(canonical)==expected
        if external.exists(): raise FileExistsError('Refusing to overwrite the held-out copy')
        external.parent.mkdir(parents=True,exist_ok=True);shutil.move(canonical,external)
        assert not canonical.exists() and sha(external)==expected
        print('Audit input withheld outside the run workspace')
    else:
        freeze=base/'selection_frozen.json'
        assert freeze.is_file(), 'Complete training and freeze every choice before releasing audit input'
        assert not canonical.exists(), 'The audit input is already present'
        assert external.is_file() and sha(external)==expected
        frozen=json.loads(freeze.read_text())
        assert frozen['data_manifest_sha256']==sha(base/'data_manifest.json')
        for name,digest in frozen['code'].items(): assert sha(ROOT/name)==digest
        canonical.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(external,canonical)
        assert sha(canonical)==expected
        record={'selection_existed_before_transfer':True,'audit_input_present_before_transfer':False,
                'selection_frozen_sha256':sha(freeze),'audit_input_sha256':expected,
                'rows':manifest['roles']['audit']['n'],'official_test_opened':False}
        (base/'audit_transfer_record.json').write_text(json.dumps(record,indent=2)+'\n')
        print('Audit input released after freezing selection; transfer record saved')


if __name__=='__main__': main()
