import numpy as np
import pytest
import torch

from imdb_lab.factorized_rfm import (FactorizedLaplacianRFM, factor_distances,
                                     factor_gradients)
from imdb_lab.rfm import LaplacianRFM, distances, prediction_gradients
from imdb_lab.qwen_core import CausalSteering, capture_prefix, cached_labels, encode_batch, generation_inputs


def test_factorized_metric_and_gradients_equal_full_agop_algebra():
    rng = np.random.default_rng(3)
    x, z, factor = rng.normal(size=(7, 19)), rng.normal(size=(11, 19)), rng.normal(size=(5, 19))
    alpha = rng.normal(size=11)
    for f in (None, factor):
        metric = np.eye(19) if f is None else f.T @ f
        np.testing.assert_allclose(factor_distances(x, z, f), distances(x, z, metric), atol=1e-10)
        np.testing.assert_allclose(factor_gradients(x, z, f, alpha, 4), prediction_gradients(x, z, metric, alpha, 4), atol=1e-9)


def test_factorized_rfm_matches_dense_without_dropping_validation_components():
    rng = np.random.default_rng(8)
    x, val = rng.normal(size=(24, 37)), rng.normal(size=(20, 37))
    y, yv = (x[:, 0] > 0).astype(float), (val[:, 0] > 0).astype(float)
    full = LaplacianRFM(bandwidth=12, iterations=3).fit(x, y, val, yv)
    fact = FactorizedLaplacianRFM(bandwidth=12, iterations=3).fit(x, y, val, yv)
    np.testing.assert_allclose(fact.predict(val), full.predict(val), atol=1e-6)
    np.testing.assert_allclose(fact.agop_factor.T @ fact.agop_factor, full.agop, atol=1e-8)
    assert abs(fact.direction @ full.direction) > .99999
    for a, b in zip(fact.history, full.history):
        assert abs(a['validation_auc'] - b['validation_auc']) < 1e-8
        assert abs(a['validation_mse'] - b['validation_mse']) < 1e-6


class TinyTokenizer:
    def encode(self, word, add_special_tokens=False):
        return [10 if word == 'negative' else 11]

    def __call__(self, sequences, **kwargs):
        from transformers import BatchEncoding
        n = max(map(len, sequences))
        return BatchEncoding({'input_ids': torch.tensor([[0] * (n-len(s)) + list(s) for s in sequences]),
                              'attention_mask': torch.tensor([[0] * (n-len(s)) + [1] * len(s) for s in sequences])})


def tiny_qwen():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(9)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        attention_dropout=0, pad_token_id=0, eos_token_id=2)).eval()
    return model


def test_qwen_cached_suffix_matches_hook_with_left_padding():
    tokenizer, model = TinyTokenizer(), tiny_qwen()
    prompts = [[1, 4, 5, 9], [1, 5, 6, 8, 7, 3]]
    cached, original = capture_prefix(model, tokenizer, prompts, 1, batch_size=2)
    v = np.random.default_rng(7).normal(size=32)
    v /= np.linalg.norm(v)
    np.testing.assert_allclose(cached_labels(model, tokenizer, cached, 1, v, 0, 1), original, atol=1e-7)
    with torch.inference_mode():
        with CausalSteering(model, 1, v, 2, .7):
            hooked = model(**encode_batch(tokenizer, prompts, 'cpu'), use_cache=False, logits_to_keep=1).logits[:, -1, [10, 11]].numpy()
    np.testing.assert_allclose(cached_labels(model, tokenizer, cached, 1, v, 2, .7), hooked, atol=1e-7)
    assert not model.model.layers[0]._forward_hooks and not model._forward_pre_hooks


def test_qwen_zero_generation_and_hook_cleanup_with_exception():
    model, tokenizer = tiny_qwen(), TinyTokenizer()
    inputs = encode_batch(tokenizer, [[1, 7, 3], [1, 3, 6, 4, 8]], 'cpu')
    v = np.ones(32) / np.sqrt(32)
    with torch.inference_mode():
        baseline = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        with CausalSteering(model, 2, v, 0, 1):
            zero = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        assert torch.equal(baseline, zero)
    with pytest.raises(RuntimeError):
        with CausalSteering(model, 2, v, 1, 1):
            raise RuntimeError('cleanup test')
    assert not model._forward_pre_hooks and not model.model.layers[1]._forward_hooks


def test_qwen_generation_cache_and_recomputed_positions_agree_with_full_decode():
    model, tokenizer = tiny_qwen(), TinyTokenizer()
    inputs = generation_inputs(tokenizer, [[1, 7, 3], [1, 3, 6, 4, 8]], 'cpu')
    assert 'position_ids' not in inputs
    v = np.random.default_rng(9).normal(size=32)
    v /= np.linalg.norm(v)
    with torch.inference_mode():
        with CausalSteering(model, 1, v, 1, .3) as hook:
            cached = model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=True)
        with CausalSteering(model, 1, v, 1, .3):
            full = model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=False)
    assert torch.equal(cached, full)
    assert hook.prefill_calls > 0 and hook.decode_calls > 0
