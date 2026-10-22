#!/usr/bin/env python3
"""Execute the results notebook with this exact interpreter, without retraining."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
runtime = ROOT / ".cache" / "jupyter"
kernel = runtime / "kernels" / "imdb_local"
kernel.mkdir(parents=True, exist_ok=True)
(kernel / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                               "display_name": "IMDb local", "language": "python"}))
os.environ["JUPYTER_PATH"] = str(runtime)
os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime / "runtime")
os.environ["IPYTHONDIR"] = str(ROOT / ".cache" / "ipython")
from nbclient import NotebookClient
import nbformat

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--notebook",default="notebooks/experimentos_imdb.ipynb")
args=parser.parse_args()
path = ROOT / args.notebook
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=300, kernel_name="imdb_local", resources={"metadata": {"path": str(ROOT)}})
client.execute()
# Portable notebook metadata: users may choose their own ordinary Python kernel.
notebook.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(notebook, path)
print("Notebook executed without retraining", len([c for c in notebook.cells if c.cell_type == 'code']), "code cells")
