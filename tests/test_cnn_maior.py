import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seminario"))

from treinar_cnn_maior import ResidualCNN, training_batches


def test_residual_pooling_uses_the_same_valid_tokens_with_extra_right_padding():
    torch.manual_seed(17)
    model = ResidualCNN(24, embedding_dim=8, channels=12, blocks=6).eval()
    tokens = torch.tensor([[3, 4, 5, 6, 7, 8, 9], [4, 3, 0, 0, 0, 0, 0]])
    lengths = torch.tensor([7, 2])
    padded = torch.nn.functional.pad(tokens, (0, 73))
    with torch.inference_mode():
        torch.testing.assert_close(model(tokens, lengths), model(padded, lengths), atol=1e-6, rtol=1e-5)


def test_length_grouped_batches_replay_and_cover_uneven_dataset():
    lengths = np.tile([1, 400, 5, 11, 3, 145, 56], 163)
    batches = training_batches(lengths, seed=17, epoch=1)
    order = np.concatenate(batches)
    assert len(lengths) % len(batches[0]) != 0
    np.testing.assert_array_equal(np.sort(order), np.arange(len(lengths)))
    for original, replay in zip(batches, training_batches(lengths, seed=17, epoch=1), strict=True):
        np.testing.assert_array_equal(original, replay)
