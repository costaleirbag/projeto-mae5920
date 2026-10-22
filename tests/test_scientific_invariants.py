import numpy as np
import pandas as pd
import pytest
import torch

from imdb_lab.data import normalize_frame, remove_training_leakage
from imdb_lab.metrics import evaluate, fit_temperature
from imdb_lab.models import TextCNN
from imdb_lab.rfm import (ActivationSteering, LaplacianRFM, agop_from_gradients,
                          prediction_gradients, bert_tail_logits)


def test_leakage_audit_removes_normalized_duplicates_and_conflicts():
    train = normalize_frame(pd.DataFrame({"text": ["Great<br />film", "great film", "Bad film", "Bad film", "Only here"],
                                         "label": [1, 1, 0, 1, 0]}), "train")
    test = normalize_frame(pd.DataFrame({"text": [" GREAT film "], "label": [1]}), "test")
    clean, audit = remove_training_leakage(train, test)
    assert clean.text.tolist() == ["Only here"]
    assert audit["train_rows_matching_test_text"] == 2
    assert audit["conflicting_train_rows_excluded"] == 2


def test_cnn_prediction_invariant_to_extra_padding():
    torch.manual_seed(0)
    model = TextCNN(40, embedding_dim=8, filters=4).eval()
    for sequence in ([2, 4], [2, 4, 8, 3, 9, 1]):
        lengths = torch.tensor([len(sequence)])
        short = torch.tensor([sequence + [0] * max(0, 5 - len(sequence))])
        long = torch.tensor([sequence + [0] * 20])
        assert torch.allclose(model(short, lengths), model(long, lengths), atol=1e-6)


def test_false_positive_negative_definitions():
    m = evaluate([0, 0, 1, 1], [.2, .8, .4, .9])
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["false_positive_rate"] == m["false_negative_rate"] == .5


def test_temperature_reduces_overconfidence_without_changing_labels():
    logits, y = np.array([-4., -3., 3., 4.]), np.array([0, 1, 0, 1])
    t = fit_temperature(logits, y)
    assert t > 1
    assert np.array_equal(logits >= 0, logits / t >= 0)


def test_laplace_gradients_match_autograd_for_general_psd_metric():
    rng = np.random.default_rng(3)
    centers, x = rng.normal(size=(9, 5)), rng.normal(size=(4, 5))
    a = rng.normal(size=(5, 5))
    metric, alpha, bw = a.T @ a, rng.normal(size=9), 2.7
    analytic = prediction_gradients(x, centers, metric, alpha, bw)
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    diff = xt[:, None, :] - torch.tensor(centers)[None, :, :]
    distance = torch.sqrt(torch.einsum("bni,ij,bnj->bn", diff, torch.tensor(metric), diff))
    f = torch.exp(-distance / bw) @ torch.tensor(alpha)
    gradient, = torch.autograd.grad(f.sum(), xt)
    np.testing.assert_allclose(analytic, gradient.detach().numpy(), rtol=1e-8, atol=1e-8)
    agop = agop_from_gradients(analytic)
    assert np.linalg.eigvalsh(agop).min() >= -1e-12
    np.testing.assert_allclose(agop, analytic.T @ analytic / len(x))


def test_rfm_learns_direction_and_orients_toward_positive():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(180, 6))
    y = (x[:, 0] + .15 * x[:, 1] > 0).astype(float)
    rfm = LaplacianRFM(bandwidth=3, iterations=3).fit(x[:120], y[:120], x[120:], y[120:])
    assert rfm.validation_auc > .9
    assert rfm.direction[0] > .8
    assert rfm.positive_orientation_covariance > 0
    assert len(rfm.history) == 3


def test_bert_hook_zero_identity_cleanup_and_nonzero_intervention():
    from transformers import BertConfig, BertForSequenceClassification
    torch.manual_seed(4)
    model = BertForSequenceClassification(BertConfig(vocab_size=40, hidden_size=16, num_hidden_layers=2,
                  num_attention_heads=2, intermediate_size=24, hidden_dropout_prob=0, attention_probs_dropout_prob=0)).eval()
    inputs = {"input_ids": torch.tensor([[2, 5, 8, 3]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    original = model(**inputs).logits.detach()
    direction = np.random.default_rng(4).normal(size=16)
    with ActivationSteering(model, 1, direction, 0):
        assert torch.equal(original, model(**inputs).logits.detach())
    with ActivationSteering(model, 1, direction, 2):
        assert not torch.allclose(original, model(**inputs).logits.detach(), atol=1e-8)
    with pytest.raises(RuntimeError):
        with ActivationSteering(model, 1, direction, 1):
            raise RuntimeError("Deliberate exception")
    assert not model.bert.encoder.layer[0]._forward_hooks
    assert torch.equal(original, model(**inputs).logits.detach())


def test_cached_bert_suffix_matches_full_forward_with_padding_and_steering():
    from transformers import BertConfig, BertForSequenceClassification
    torch.manual_seed(9)
    model = BertForSequenceClassification(BertConfig(vocab_size=40, hidden_size=16, num_hidden_layers=3,
            num_attention_heads=2, intermediate_size=24, hidden_dropout_prob=0, attention_probs_dropout_prob=0)).eval()
    inputs = {"input_ids": torch.tensor([[2, 5, 8, 3, 0], [2, 6, 9, 7, 3]]),
              "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])}
    with torch.inference_mode():
        full = model(**inputs, output_hidden_states=True)
        hidden = full.hidden_states[1].clone()
        assert torch.allclose(full.logits, bert_tail_logits(model, hidden, inputs["attention_mask"], 1), atol=1e-7)
        direction = np.random.default_rng(9).normal(size=16)
        with ActivationSteering(model, 1, direction, 1.5, scale=.7):
            hooked = model(**inputs).logits
        hidden[:, 0] += torch.as_tensor(1.5 * .7 * direction, dtype=hidden.dtype)
        cached = bert_tail_logits(model, hidden, inputs["attention_mask"], 1)
        assert torch.allclose(hooked, cached, atol=1e-7)
