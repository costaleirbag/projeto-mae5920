#!/usr/bin/env python3
"""Verifica o notebook de entrega, seus anexos e as definições de produção."""
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
ANCHORS = ("introducao", "dados", "cnn", "treino", "oficial", "diagnosticos",
           "falhas", "fora-do-imdb", "apoio", "reproduzir")
STAGES = (
    ("tools/seminar_retrain.py", "original"),
    ("tools/seminar_retrain.py", "encoders"),
    ("tools/seminar_retrain.py", "textcnn"),
    ("tools/seminar_retrain.py", "qwen"),
    ("tools/seminar_recompute.py", "original"),
    ("tools/seminar_recompute.py", "textcnn"),
    ("tools/seminar_recompute.py", "bert-cap"),
    ("tools/seminar_recompute.py", "qwen-youtube"),
    ("tools/seminar_steering.py",),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(path=NOTEBOOK, require_execution=True):
    path = Path(path)
    nb = json.loads(path.read_text(encoding="utf-8"))
    meta = nb["metadata"]["mae5920_seminar"]
    assert meta["format"] == 2
    payload = base64.b64decode(nb["cells"][0]["attachments"]["evidencias.zip"]["application/zip"])
    assert digest(payload) == meta["payload_sha256"]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.testzip() is None
        expected = {name for name in meta["assets"] if not name.startswith("source_archive/")}
        assert set(archive.namelist()) == expected and len(expected) == len(archive.namelist())
        for name, record in meta["assets"].items():
            if name.startswith("source_archive/"):
                assert digest((ROOT / record["source"]).read_bytes()) == record["sha256"], name
                continue
            assert not name.startswith(("/", "../", "data/train", "artifacts/", ".cache/"))
            assert ".." not in Path(name).parts
            data = archive.read(name)
            assert digest(data) == record["sha256"] and len(data) == record["bytes"], name
            if (ROOT / name).is_file():
                assert (ROOT / name).read_bytes() == data, name
        runtime = set(meta["runtime_files"])
        assert runtime <= expected
        assert {stage[0] for stage in STAGES} <= runtime
        assert {"derived/oficial.npz", "derived/papeis_originais.json"} <= expected - runtime

        markdown = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
        for anchor in ANCHORS:
            assert f'id="{anchor}"' in markdown
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        definitions = {}
        assignments = {}
        codes = ["".join(c["source"]) for c in code_cells]
        for source in codes:
            assert len(source) < 15_000, "Arquivos grandes ficam no anexo do notebook"
            for node in ast.parse(source).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    definitions[node.name] = node
                if isinstance(node, ast.Assign):
                    assignments.update({t.id: node.value for t in node.targets if isinstance(t, ast.Name)})
        assert ast.literal_eval(assignments["MODO"]) == "conferir"
        assert ast.literal_eval(assignments["ETAPAS"]) == STAGES

        for record in meta["extracted_definitions"]:
            data = archive.read(record["source"])
            assert digest(data) == record["source_sha256"], record
            definition = next(node for node in ast.parse(data.decode()).body
                              if getattr(node, "name", None) == record["symbol"])
            assert record["symbol"] in definitions, record["symbol"]
            assert ast.dump(definition, include_attributes=False) == ast.dump(
                definitions[record["symbol"]], include_attributes=False), record["symbol"]

    checks = None
    if require_execution:
        assert [c["execution_count"] for c in code_cells] == list(range(1, len(code_cells) + 1))
        outputs = [o for c in code_cells for o in c["outputs"]]
        assert not any(o["output_type"] == "error" or o.get("name") == "stderr" for o in outputs)
        streams = "\n".join("".join(o.get("text", [])) for o in outputs if o["output_type"] == "stream")
        found = re.search(r"(\d+) verificações no notebook passaram\.", streams)
        assert found and int(found[1]) >= 40
        checks = int(found[1])

    return {"status": "passed", "notebook": path.name, "cells": len(nb["cells"]),
            "executed_code_cells": len(code_cells) if require_execution else 0,
            "embedded_files": len(expected), "inline_production_definitions": len(meta["extracted_definitions"]),
            "chapters": len(ANCHORS), "payload_sha256": digest(payload), "notebook_sha256": digest(path.read_bytes()),
            "code_sha256": digest("\n".join(codes).encode()), "self_contained_data": True,
            "architecture_matches_production": True, "full_training_disabled_by_default": True,
            "reproduction_stages": [list(stage) for stage in STAGES], "checks_executed": checks}


if __name__ == "__main__":
    result = verify()
    target = ROOT / "results/seminar/verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
