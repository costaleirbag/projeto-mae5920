"""Scientific controls for the CNN/encoder extension; no model downloads."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from torch import nn
from imdb_lab.models import TextCNN, Vocabulary
from imdb_lab.sequences import excerpt, masked_mean, orders, select_rate, train_one, Probe


def test_masked_mean_excludes_special_and_padding_tokens():
    h = torch.tensor([[[100., 100.], [2., 4.], [4., 8.], [200., 200.], [-50., -50.]]])
    mask = torch.tensor([[False, True, True, False, False]])
    torch.testing.assert_close(masked_mean(h, mask), torch.tensor([[3., 6.]]))
    assert torch.isfinite(masked_mean(h, torch.zeros_like(mask))).all()


def test_cnn_prediction_invariant_to_added_right_padding():
    torch.manual_seed(7); model = TextCNN(12, 8, 4, (3, 4, 5), dropout=0).eval()
    a = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 0, 0, 0]])
    b = torch.nn.functional.pad(a, (0, 9))
    n = torch.tensor([5, 2])
    torch.testing.assert_close(model(a, n), model(b, n), atol=1e-6, rtol=1e-6)


def test_vocabulary_heldout_word_never_fitted():
    vocab = Vocabulary.fit(["warm acting", "good acting"], max_tokens=20)
    assert vocab.encode("unseenword", 10) == [1]
    assert excerpt("one   two\nthree four", 3) == "one two three"


def test_learning_rate_uses_mean_validation_not_best_seed():
    records = [{"spec": {"lr": lr}, "best_validation_ce": ce} for lr, ce in [(0.1, 0.1), (0.1, 0.9), (0.01, 0.3), (0.01, 0.3)]]
    rate, means = select_rate(records)
    assert rate == 0.01 and means["0.1"] == 0.5


def test_paired_epoch_orders_cover_every_example_once():
    a = orders(129, 3, 17); b = orders(129, 3, 17)
    assert all(np.array_equal(x, y) and np.array_equal(np.sort(x), np.arange(129)) for x, y in zip(a, b))
    assert not np.array_equal(a[0], a[1])


def test_checkpoint_epoch_and_predictions_follow_validation_ce():
    torch.manual_seed(4); x = torch.randn(32, 3); y = (x[:, 0] > 0).long()
    v = torch.randn(16, 3); vy = (v[:, 0] > 0).long()
    model = Probe(3)
    spec = {"run_id": "unit", "optimizer": "adam", "lr": .03, "epochs": 4, "batch_size": 8}
    out = train_one(model, ({"x": x}, y), ({"x": v}, vy), spec, 17, torch.device("cpu"))
    expected = min(out["history"][1:], key=lambda h: h["validation_ce"])
    assert out["best_epoch"] == expected["epoch"]
    model.load_state_dict(out["best_state"]); model.eval()
    torch.testing.assert_close(model(v), torch.tensor(out["best_predictions"]["validation"]))
    assert out["history"][-1]["updates"] == 16
