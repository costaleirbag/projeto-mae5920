#!/usr/bin/env python3
"""Execute the new reading notebook and record verified deliverable hashes."""
from pathlib import Path
import json
import hashlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    q = ROOT / "results/didactic"
    record = json.loads((q / "report_manifest.json").read_text())
    for kind in ["pdf", "html"]:
        assert sha(ROOT / record[kind]) == record[kind + "_sha256"]
    for path, digest in record["figures"].items(): assert sha(ROOT / path) == digest
    subprocess.run([sys.executable, str(ROOT / "execute_notebook.py"), "--notebook", record["notebook"]], cwd=ROOT, check=True)
    import nbformat
    from pypdf import PdfReader
    nb = nbformat.read(ROOT / record["notebook"], as_version=4)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    assert cells and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == "error" for c in cells for o in c.outputs)
    pdf = PdfReader(ROOT / record["pdf"])
    assert len(pdf.pages) == 8 and all(len(p.extract_text()) > 200 for p in pdf.pages)
    text = " ".join(p.extract_text() for p in pdf.pages)
    assert all(word in text for word in ["Otimização", "Arquitetura", "Inicialização", "Regularização", "Double descent"])
    summary = json.loads((q / "summary.json").read_text())
    for p in [24, 64, 512]:
        assert f"{summary['double_descent'][str(p)]['test_mse_median']:.3f}" in text
    record.update(notebook_execution="passed", executed_code_cells=len(cells), notebook_sha256=sha(ROOT / record["notebook"]),
                  pdf_pages=len(pdf.pages), builder_sha256=sha(ROOT / "src/imdb_lab/didactic_report.py"),
                  summary_sha256=sha(q / "summary.json"), verification_sha256=sha(q / "verification.json"),
                  visual_review=record.get("visual_review", "pending"))
    (q / "report_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"Deliverables verified: {len(cells)} executed notebook cells, 8 PDF pages, exact summary values and unchanged figure files")


if __name__ == "__main__":
    main()
