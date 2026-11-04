#!/usr/bin/env python3
"""Verify the unified notebook, its embedded provenance and visible real code."""
from __future__ import annotations
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks/seminario_completo.ipynb"
ANCHORS = ["dados","tokens","cnn","treino","experimentos","bpe","encoders","resultados",
           "mecanismos","double-descent","qwen","originais","testes","reproduzir","referencias"]


def digest(data): return hashlib.sha256(data).hexdigest()


def verify(path=NOTEBOOK, require_execution=True):
    path = Path(path); nb = json.loads(path.read_text())
    meta = nb["metadata"]["mae5920_seminar"]
    payload = base64.b64decode(nb["cells"][0]["attachments"]["evidencias.zip"]["application/zip"])
    assert digest(payload) == meta["payload_sha256"]
    archive = zipfile.ZipFile(io.BytesIO(payload))
    assert archive.testzip() is None
    expected = {n for n in meta["assets"] if not n.startswith("source_archive/")}
    assert set(archive.namelist()) == expected
    assert len(expected) == len(archive.namelist())
    for name in expected:
        record = meta["assets"][name]; data = archive.read(name)
        assert not name.startswith(("/", "../", "data/train", "artifacts/", ".cache/"))
        assert digest(data) == record["sha256"] and len(data) == record["bytes"]
        if (ROOT / name).is_file(): assert (ROOT / name).read_bytes() == data, name
    for name, record in meta["assets"].items():
        if name.startswith("source_archive/"):
            assert digest((ROOT / record["source"]).read_bytes()) == record["sha256"]
    for source, compact in meta["aliases"].items():
        original=json.loads((ROOT/source).read_text())
        for role in original["roles"].values():
            role.pop("text_sha256",None);role.pop("excerpt_sha256",None)
        assert json.loads(archive.read(compact)) == original
    markdown = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    for anchor in ANCHORS:
        assert f'id="{anchor}"' in markdown and f'href="#{anchor}"' in markdown
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    trees = []
    definitions = {}; assignments = set()
    for cell in code_cells:
        source = "".join(cell["source"]); tree = ast.parse(source); trees.append(tree)
        if require_execution:
            assert cell.get("execution_count") is not None, cell["id"]
            assert not any(o.get("output_type") == "error" for o in cell.get("outputs",[])), cell["id"]
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                modules=[node.module or ""] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names]
                assert not any(m.startswith("imdb_lab") for m in modules)
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)): definitions[node.name] = node
            if isinstance(node,ast.Assign): assignments.update(t.id for t in node.targets if isinstance(t,ast.Name))
    for name in ["modelo","registros_seq","bpe_real","trios","verificacoes_metricas"]:
        assert name in assignments, f"Teaching example accidentally nested in a function: {name}"
    for record in meta["extracted_definitions"]:
        data=archive.read(record["source"])
        assert digest(data)==record["source_sha256"]
        original=next(n for n in ast.parse(data.decode()).body if getattr(n,"name",None)==record["symbol"])
        assert record["symbol"] in definitions, record["symbol"]
        assert ast.dump(original,include_attributes=False)==ast.dump(definitions[record["symbol"]],include_attributes=False),record["symbol"]
    flags=[n for t in trees for n in t.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=="TREINAR_COMPARACAO_COMPLETA" for v in n.targets)]
    assert len(flags)==1 and isinstance(flags[0].value,ast.Constant) and flags[0].value.value is False
    # No giant visible base64 cell: the archive belongs to a notebook attachment.
    assert max(len("".join(c["source"])) for c in code_cells) < 15_000
    if require_execution:
        assert [c["execution_count"] for c in code_cells] == list(range(1,len(code_cells)+1))
        streams="\n".join("".join(o.get("text",[])) for c in code_cells for o in c.get("outputs",[]) if o.get("output_type")=="stream")
        match=re.search(r"Fim do roteiro: (\d+) verificações passaram",streams)
        assert match and int(match[1]) > 250
        assert not any(o.get("name")=="stderr" for c in code_cells for o in c.get("outputs",[])), "Clean teaching output required"
    result={"status":"passed","notebook":path.name,"cells":len(nb["cells"]),"executed_code_cells":len(code_cells) if require_execution else 0,
            "embedded_files":len(expected),"inline_production_definitions":len(meta["extracted_definitions"]),
            "chapters":len(ANCHORS),"payload_sha256":digest(payload),"notebook_sha256":digest(path.read_bytes()),
            "code_sha256":digest("\n".join("".join(c["source"]) for c in code_cells).encode()),
            "self_contained_data":True,"architecture_matches_production":True,"full_training_disabled_by_default":True,
            "checks_executed":int(match[1]) if require_execution else None}
    return result


if __name__ == "__main__":
    result=verify()
    target=ROOT/"results/seminar/verification.json";target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
