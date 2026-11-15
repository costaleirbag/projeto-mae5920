#!/usr/bin/env python3
"""Execute a lone notebook outside the repository with Python networking blocked."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path,
                        help="Diretório com os quatro Parquets e o manifesto, para reajustar o TF-IDF real.")
    args=parser.parse_args()
    path=ROOT/"notebooks/seminario_completo.ipynb";original=nbformat.read(path,as_version=4)
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mae5920-notebook-") as temporary:
        scratch=Path(temporary);local=scratch/path.name;shutil.copy2(path,local)
        runtime=scratch/".jupyter";kernel=runtime/"kernels/seminario_cpu";kernel.mkdir(parents=True)
        (kernel/"kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
            "display_name":"Seminário CPU isolado","language":"python","env":{"CUDA_VISIBLE_DEVICES":"","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}}))
        os.environ["JUPYTER_PATH"]=str(runtime);os.environ["JUPYTER_RUNTIME_DIR"]=str(runtime/"runtime")
        os.environ["IPYTHONDIR"]=str(scratch/".ipython")
        guarded=nbformat.from_dict(json.loads(json.dumps(original)))
        guard=nbformat.v4.new_code_cell("import socket\ndef _deny_python_network(*args,**kwargs):\n    raise RuntimeError('Python network disabled during portability check')\nsocket.socket.connect=_deny_python_network\nsocket.create_connection=_deny_python_network\n")
        guarded.cells.insert(0,guard)
        commands = """
pasta = preparar_execucao(Path("reproducao-portabilidade"))
for script in ("tools/seminar_retrain.py", "tools/seminar_recompute.py"):
    resultado = subprocess.run([sys.executable, script, "--help"], cwd=pasta,
                               check=True, capture_output=True, text=True)
    assert "usage:" in resultado.stdout.lower()
assert (pasta / "referencia/derived/oficial.npz").is_file()
assert (pasta / "src/imdb_lab/training.py").is_file()
"""
        if args.corpus:
            corpus = args.corpus.resolve()
            if corpus.is_file():
                corpus = corpus.parent
            files = ("manifest.json", "train.parquet", "validation.parquet", "calibration.parquet", "test.parquet")
            for name in files:
                if not (corpus / name).is_file():
                    parser.error(f"Arquivo do corpus ausente: {name}")
            commands += "CORPUS_PATH = Path(" + repr(str(corpus)) + ")\n"
            commands += """
import shutil
shutil.copytree(CORPUS_PATH, pasta / "data", ignore=shutil.ignore_patterns("*.csv"))
processo = subprocess.run([sys.executable, "tools/seminar_retrain.py", "original",
                           "--original-models", "tfidf"], cwd=pasta, capture_output=True, text=True)
assert processo.returncode == 0, processo.stdout[-3000:] + processo.stderr[-3000:]
comparacao = json.loads((pasta / "results/seminar_retrain/original/comparacao.json").read_text())
assert len(comparacao["comparacao"]) == 16
assert all(abs(row["diferença"]) < 1e-6 for row in comparacao["comparacao"]), comparacao
"""
        guarded.cells.append(nbformat.v4.new_code_cell(commands))
        nb=guarded
        client=NotebookClient(nb,timeout=300,kernel_name="seminario_cpu",resources={"metadata":{"path":str(scratch)}})
        try: client.execute()
        except Exception:
            failure=ROOT/".cache/seminar_portability_failure.ipynb";failure.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,failure)
            raise
        cells=[c for c in nb.cells if c.cell_type=="code"]
        assert all(c.execution_count is not None for c in cells)
        assert not any(o.output_type=="error" for c in cells for o in c.outputs)
        original_code="\n".join(c.source for c in original.cells if c.cell_type=="code")
        result={"status":"passed","only_notebook_copied":True,"working_directory_outside_repository":True,
            "python_socket_connections_blocked":True,"gpu_hidden":True,
            "reader_code_cells_executed":sum(c.cell_type=="code" for c in original.cells),
            "payload_sha256":original.metadata.mae5920_seminar.payload_sha256,
            "reader_code_sha256":hashlib.sha256(original_code.encode()).hexdigest(),
            "runtime_extracted_from_notebook": True,
            "optional_full_runner_smoke": "passed: full TF-IDF with restored IMDb; 16 metrics within 1e-6"
                                        if args.corpus else "not requested",
            "scientific_benchmark_rerun":False,"elapsed_seconds":time.perf_counter()-started}
        target=ROOT/"results/seminar/portability.json";target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
