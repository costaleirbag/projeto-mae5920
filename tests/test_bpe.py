import numpy as np
import torch
from imdb_lab.bpe_study import canonicalize, fit_bpe, token_stats, metrics
from imdb_lab.models import TextCNN


def test_bpe_preserves_original_lexical_normalization():
    text = "The WELL-made movie wasn't good! 10/10"
    assert canonicalize(text) == "the well made movie wasn't good 10 10"
    tokenizer = fit_bpe([text] * 4 + ["the movie was unwatchable"] * 4, vocab_size=100)
    assert tokenizer.decode(tokenizer.encode(canonicalize(text)).ids).strip() == canonicalize(text)
    assert tokenizer.token_to_id("<pad>") == 0 and tokenizer.token_to_id("<unk>") == 1


def test_bpe_can_spell_an_unseen_word_without_learning_it():
    tokenizer = fit_bpe(["low lower lowest new newer newest"] * 4, vocab_size=35)
    tokenized = tokenizer.encode("newestlower")
    assert 1 not in tokenized.ids
    assert tokenizer.decode(tokenized.ids).strip() == "newestlower"
    assert "newestlower</w>" not in tokenizer.get_vocab()
    # Character BPE has no byte fallback: a genuinely unseen alphabet can use UNK.
    assert 1 in tokenizer.encode("Ж").ids


def test_coverage_counts_unknown_tokens_before_truncation():
    stats = token_stats([[2, 1, 3], [], [1]], cap=2)
    assert stats["unknown_tokens"] == 2 and stats["unknown_rate"] == .5
    assert stats["truncated_documents"] == 1 and stats["empty_documents"] == 1


def test_metrics_depend_on_logits_not_coverage():
    logits = np.array([[2., 0.], [0., 2.]])
    assert metrics(logits, [0, 1])["accuracy"] == 1.
    assert metrics(logits[::-1], [0, 1])["accuracy"] == 0.


def test_cnn_padding_does_not_change_bpe_prediction():
    torch.manual_seed(19)
    model = TextCNN(40, embedding_dim=8, filters=3, kernels=[3, 4, 5], dropout=0).eval()
    ids = torch.tensor([[4, 8, 2, 19, 7, 6]])
    lengths = torch.tensor([6])
    with torch.no_grad():
        a = model(ids, lengths)
        b = model(torch.nn.functional.pad(ids, (0, 12)), lengths)
    torch.testing.assert_close(a, b)
