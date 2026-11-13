"""Controles para a máscara das novas convoluções e a escolha do recorte."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seminario"))

from auditar_baseline import cut
from imdb_lab.models import tokenize
from treinar_cnn_comprimento import DEFAULTS, RECIPES, ReviewBatches, model_for


def test_crop_and_batch_padding_use_the_same_token_count():
    raw = "One, two, İ THREE 14. That's enough."
    for limit in (1, 3, 4, 20):
        assert tokenize(cut(raw, limit)) == tokenize(raw)[:limit]
    rows = [[2, 3, 4, 5], [6, 7], list(range(8, 22))]
    data = ReviewBatches(rows, np.array([0, 1, 1]), 9, 5, torch.device("cpu"))
    (tokens, lengths), labels = data.batch(np.array([1, 0]))
    assert tokens.shape == (2, 5)
    assert tokens[0].tolist() == [6, 7, 0, 0, 0]
    assert lengths.tolist() == [2, 4]
    assert labels.tolist() == [1, 0]
    assert data.lengths.tolist() == [4, 2, 9]


def test_all_new_cnn_variants_ignore_extra_right_padding():
    target = torch.device("cpu")
    tokens = torch.tensor([[3, 4, 5, 6, 7, 8, 9], [4, 3, 0, 0, 0, 0, 0]])
    padded = torch.nn.functional.pad(tokens, (0, 11))
    lengths = torch.tensor([7, 2])
    with torch.inference_mode():
        for name in ("max_400", "max_media_400", "max_metades_400", "dilatada_max_media_400", "largura_357_400"):
            model = model_for(22, DEFAULTS | RECIPES[name], 17, target).eval()
            torch.testing.assert_close(model(tokens, lengths), model(padded, lengths), atol=1e-6, rtol=1e-5)
