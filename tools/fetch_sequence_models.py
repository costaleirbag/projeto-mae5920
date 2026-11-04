#!/usr/bin/env python3
"""Download only the public pretrained files pinned by the sequence study."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=ROOT/'.cache/sequence_pretrained')
    args=parser.parse_args()
    from huggingface_hub import snapshot_download
    pins=json.loads((ROOT/'results/sequences/models_pinned.json').read_text())
    for model,record in pins.items():
        folder=args.destination/model.split('/')[-1]
        snapshot_download(model,revision=record['revision'],local_dir=folder,allow_patterns=list(record['files']),token=False)
        for name,info in record['files'].items():
            path=folder/name
            assert path.stat().st_size==info['bytes'] and sha(path)==info['sha256'],f'Unexpected pretrained file: {model}/{name}'
        print(model,record['revision'],'verified')


if __name__=='__main__': main()
