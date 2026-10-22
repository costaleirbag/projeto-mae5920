from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

import torch
from torch import nn
from torch.utils.data import Dataset

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+", re.I)


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Vocabulary:
    token_to_id: dict

    @classmethod
    def fit(cls, texts, max_tokens=20000):
        counts = Counter()
        for text in texts:
            counts.update(tokenize(text))
        # Sorting ties makes the vocabulary independent of Counter internals.
        words = sorted(counts, key=lambda w: (-counts[w], w))[:max_tokens - 2]
        return cls({"<pad>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate(words)}})

    def encode(self, text, max_length):
        return [self.token_to_id.get(t, 1) for t in tokenize(text)[:max_length]] or [1]


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, filters=100, kernels=(3, 4, 5), dropout=.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, filters, k) for k in kernels)
        self.kernels = tuple(kernels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters * len(kernels), 2)

    def forward(self, input_ids, lengths):
        x = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for conv, k in zip(self.convs, self.kernels):
            features = torch.relu(conv(x))
            # Padded convolution windows must never win max pooling.
            n_valid = (lengths - k + 1).clamp_min(1)
            keep = torch.arange(features.shape[-1], device=features.device)[None, :] < n_valid[:, None]
            features = features.masked_fill(~keep[:, None, :], torch.finfo(features.dtype).min)
            pooled.append(features.max(dim=2).values)
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class EncodedReviews(Dataset):
    def __init__(self, sequences, labels):
        self.sequences, self.labels = sequences, list(map(int, labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.sequences[i], self.labels[i]


class PadBatch:
    def __init__(self, model_kind, pad_id=0, min_length=5):
        self.model_kind, self.pad_id, self.min_length = model_kind, pad_id, min_length

    def __call__(self, batch):
        sequences, labels = zip(*batch)
        lengths = torch.tensor([len(x) for x in sequences])
        size = max(self.min_length, int(lengths.max()))
        tokens = torch.full((len(batch), size), self.pad_id, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            tokens[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        inputs = {"input_ids": tokens}
        if self.model_kind == "cnn":
            inputs["lengths"] = lengths
        else:
            inputs["attention_mask"] = (tokens != self.pad_id).long()
        return inputs, torch.tensor(labels, dtype=torch.long)
