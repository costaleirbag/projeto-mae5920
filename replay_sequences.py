#!/usr/bin/env python3
"""Re-evaluate every frozen checkpoint in a fresh process, preserving first audit."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from imdb_lab.common import read_json,save_json,file_sha256
from imdb_lab import sequences as lab


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained',type=Path,default=Path('.cache/sequence_pretrained'))
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    first=ROOT/'results/sequences'; repeat=ROOT/'.cache/sequence_replay'
    assert first != repeat and not repeat.exists(), 'Keep each replay isolated'
    repeat.mkdir(parents=True)
    for name in ['selection_frozen.json','data_manifest.json','vocabulary.json','models_pinned.json']:
        shutil.copy2(first/name,repeat/name)
    (repeat/'runs').mkdir()
    for path in (first/'runs').glob('*.json'): shutil.copy2(path,repeat/'runs'/path.name)
    original_hash=file_sha256(first/'audit_predictions.npz')
    lab.SROOT=repeat
    lab.audit(args.pretrained,args.device)
    per_key={}; content_hashes={}
    with np.load(first/'audit_predictions.npz') as a,np.load(repeat/'audit_predictions.npz') as b:
        assert set(a.files)==set(b.files)
        assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['y'],b['y'])
        for key in a.files:
            if key in {'ids','y'}: continue
            per_key[key]=float(np.max(np.abs(a[key].astype(np.float64)-b[key].astype(np.float64))))
            content_hashes[key]={'first':lab.array_hash(a[key]),'replay':lab.array_hash(b[key])}
    maximum=max(per_key.values()); assert maximum<=1e-5
    assert file_sha256(first/'audit_predictions.npz')==original_hash
    save_json(first/'replay_verification.json',{'status':'passed','checkpoints':len(per_key),'max_absolute_logit_difference':maximum,
        'per_checkpoint_max_difference':per_key,'content_sha256':content_hashes,'audit_predictions_sha256':original_hash,
        'selection_frozen_sha256':file_sha256(first/'selection_frozen.json'),'code':lab.fingerprint(),
        'description':'Fresh process, same frozen weights and data, repeated all end-to-end predictions. First audit files preserved byte for byte.'})
    print('REPLAY_PASSED',len(per_key),maximum)


if __name__=='__main__': main()
