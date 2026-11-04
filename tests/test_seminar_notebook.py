"""Controls for the notebook as an executable, portable teaching artifact."""
import importlib.util
import json
from pathlib import Path
import pytest


def test_master_notebook_is_complete_and_uses_the_actual_architectures():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("verify_seminar_notebook",root/"verify_seminar_notebook.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.verify()
    assert result["chapters"]==15
    assert result["executed_code_cells"]==45
    assert result["inline_production_definitions"]>=20
    assert result["embedded_files"]>100


def test_a_changed_cnn_cannot_be_presented_as_the_original_architecture(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("verify_seminar_notebook",root/"verify_seminar_notebook.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    nb=json.loads((root/"notebooks/seminario_completo.ipynb").read_text())
    cell=next(c for c in nb["cells"] if c["id"]=="textcnn-codigo-real")
    original="".join(cell["source"])
    changed=original.replace("filters * len(kernels), 2", "filters * len(kernels), 3")
    assert changed!=original
    cell["source"]=changed
    path=tmp_path/"modified.ipynb";path.write_text(json.dumps(nb))
    with pytest.raises(AssertionError,match="TextCNN"):
        module.verify(path,require_execution=False)
