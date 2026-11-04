#!/usr/bin/env python3
"""Link the executed notebook to portability and completed human visual review."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from verify_seminar_notebook import verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed",action="store_true",help="Record only after reviewing the actual final figures and narrative")
    args=parser.parse_args()
    result=verify()
    base=ROOT/"results/seminar"
    portability=json.loads((base/"portability.json").read_text())
    assert portability["status"]=="passed"
    assert portability["payload_sha256"]==result["payload_sha256"]
    assert portability["reader_code_sha256"]==result["code_sha256"]
    path=ROOT/"notebooks/seminario_completo.ipynb";nb=json.loads(path.read_text())
    figures={};html_outputs=0
    for c in nb["cells"]:
        for i,o in enumerate(c.get("outputs",[])):
            data=o.get("data",{})
            html_outputs+=int("text/html" in data)
            if "image/png" in data:
                value=data["image/png"];raw=base64.b64decode(''.join(value) if isinstance(value,list) else value)
                figures[f"{c['id']}:{i}"]=hashlib.sha256(raw).hexdigest()
    assert len(figures)==16
    record=json.loads((base/"build_manifest.json").read_text())
    assert record["builder_sha256"]==hashlib.sha256((ROOT/"tools/build_seminar_notebook.py").read_bytes()).hexdigest()
    record.update(execution="passed",notebook_sha256=result["notebook_sha256"],
        code_sha256=result["code_sha256"],checks_executed=result["checks_executed"],
        portability="passed",figure_count=len(figures),html_outputs=html_outputs,
        review="passed" if args.reviewed else "pending")
    (base/"build_manifest.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    (base/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if args.reviewed:
        review={"status":"passed","notebook_sha256":result["notebook_sha256"],"figure_sha256":figures,
            "all_figures_inspected":True,"narrative_and_code_reviewed":True,
            "improvements":["Show full tokenization instead of truncated dataframe cells",
                "Explicitly label synthetic CPU exercises and separate evaluation cohorts",
                "Use a readable fixed vertical scale for the orthogonal steering projection",
                "Separate BPE learning curves and paired differences with complete axis labels",
                "Keep editable scientific source definitions visible and matched to production"],
            "scope":"Notebook figures, tables, code and reading sequence; not a new GPU model evaluation"}
        (base/"review.json").write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    print(f"Final notebook: {result['executed_code_cells']} executed cells, {result['checks_executed']} checks, {len(figures)} figures, portability passed")


if __name__=="__main__":main()
