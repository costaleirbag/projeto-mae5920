"""Laplacian Recursive Feature Machine, equations (1)-(2), Beaglehole et al.

AGOP is an average of gradient outer products, not the exact Hessian.
No coordinatewise standardization, PCA replacement, or classifier weight is
silently substituted for the RFM. Inputs are centered and scaled by ONE scalar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score


def distances(x, z, metric):
    xm, zm = x @ metric, z @ metric
    squared = (xm * x).sum(axis=1)[:, None] + (zm * z).sum(axis=1)[None, :] - 2 * xm @ z.T
    return np.sqrt(np.maximum(squared, 0))


def laplace_kernel(x, z, metric, bandwidth):
    return np.exp(-distances(x, z, metric) / bandwidth)


def prediction_gradients(x, centers, metric, alpha, bandwidth, chunk=128):
    """Analytic gradients w.r.t. x, checked against PyTorch autograd in tests.

    The Laplace kernel is nondifferentiable at distance zero. We choose the zero
    subgradient there (including each training point's own kernel term).
    """
    output = []
    cm = centers @ metric
    center_sq = np.sum(centers * centers, axis=1)
    for start in range(0, len(x), chunk):
        block = x[start:start + chunk]
        distance = distances(block, centers, metric)
        # Numerical cancellation of identical points can leave tiny distances.
        euclidean_sq = np.sum(block * block, axis=1)[:, None] + center_sq[None, :] - 2 * block @ centers.T
        identical = np.abs(euclidean_sq) < 1e-10
        ok = (distance > 1e-10) & ~identical
        weight = np.divide(np.exp(-distance / bandwidth), distance, out=np.zeros_like(distance), where=ok)
        weight *= alpha[None, :]
        gradient = -(weight.sum(axis=1)[:, None] * (block @ metric) - weight @ cm) / bandwidth
        output.append(gradient)
    return np.concatenate(output)


def agop_from_gradients(gradients):
    agop = gradients.T @ gradients / len(gradients)
    return (agop + agop.T) / 2


def oriented_top_vector(agop, x, y):
    eigenvalues, eigenvectors = np.linalg.eigh(agop)
    vector = eigenvectors[:, -1]
    covariance = np.mean(((x @ vector) - (x @ vector).mean()) * (y - y.mean()))
    if covariance < 0:
        vector = -vector
    return vector / np.linalg.norm(vector), eigenvalues


@dataclass
class LaplacianRFM:
    bandwidth: float = 30.
    regularization: float = 1e-3
    iterations: int = 4

    def fit(self, x, y, x_val, y_val):
        x, x_val = np.asarray(x, dtype=np.float64), np.asarray(x_val, dtype=np.float64)
        y, y_val = np.asarray(y, dtype=np.float64), np.asarray(y_val, dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.scale = max(float(np.sqrt(np.mean((x - self.mean) ** 2))), 1e-8)
        self.centers = (x - self.mean) / self.scale
        val = (x_val - self.mean) / self.scale
        metric = np.eye(x.shape[1])
        self.intercept = float(y.mean())
        best_key = (-np.inf, -np.inf)
        self.history = []
        for iteration in range(self.iterations):
            kernel = laplace_kernel(self.centers, self.centers, metric, self.bandwidth)
            alpha = cho_solve(cho_factor(kernel + self.regularization * np.eye(len(x)), lower=True), y - self.intercept)
            preds = laplace_kernel(val, self.centers, metric, self.bandwidth) @ alpha + self.intercept
            auc = float(roc_auc_score(y_val, preds))
            mse = float(np.mean((preds - y_val) ** 2))
            gradients = prediction_gradients(self.centers, self.centers, metric, alpha, self.bandwidth)
            agop = agop_from_gradients(gradients)
            self.history.append({"iteration": iteration + 1, "validation_auc": auc, "validation_mse": mse,
                                 "agop_trace": float(np.trace(agop))})
            if (auc, -mse) > best_key:
                best_key = (auc, -mse)
                self.metric, self.alpha, self.agop = metric.copy(), alpha.copy(), agop.copy()
                self.selected_iteration = iteration + 1
                self.validation_auc = auc
            # The raw AGOP update from the paper, without trace renormalization.
            metric = agop
        self.direction, self.eigenvalues = oriented_top_vector(self.agop, self.centers, y)
        self.projection_std = float(np.std(x @ self.direction, ddof=1))
        self.positive_orientation_covariance = float(np.cov(x @ self.direction, y)[0, 1])
        return self

    def predict(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return laplace_kernel(z, self.centers, self.metric, self.bandwidth) @ self.alpha + self.intercept

    def save(self, path):
        np.savez_compressed(path, centers=self.centers, mean=self.mean, scale=self.scale,
                            metric=self.metric, alpha=self.alpha, agop=self.agop,
                            direction=self.direction, eigenvalues=self.eigenvalues,
                            projection_std=self.projection_std, bandwidth=self.bandwidth,
                            regularization=self.regularization, intercept=self.intercept)


class ActivationSteering:
    """Add alpha * scale * direction after a 1-indexed BERT encoder block.

    Scope 'cls' controls the classifier representation; scope 'all' changes all
    non-padding positions and is used in the separate masked-LM exercise.
    Hooks are always removed on exiting the context, including exceptions.
    """
    def __init__(self, model, layer, direction, alpha, scale=1., scope="cls", attention_mask=None):
        import torch
        if scope not in {"cls", "all"}:
            raise ValueError(scope)
        self.block = model.bert.encoder.layer[layer - 1]
        self.direction = torch.as_tensor(direction).detach().clone()
        self.alpha, self.scale, self.scope = float(alpha), float(scale), scope
        self.attention_mask, self.handle = attention_mask, None

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def _hook(self, module, args, result):
        if self.alpha == 0:
            return result
        hidden = result[0] if isinstance(result, tuple) else result
        direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        if direction.numel() != hidden.shape[-1]:
            raise ValueError("Steering direction has wrong hidden dimension")
        modified = hidden.clone()
        delta = self.alpha * self.scale * direction
        if self.scope == "cls":
            modified[:, 0, :] += delta
        else:
            if self.attention_mask is None:
                raise ValueError("All-token steering requires an attention mask")
            modified += self.attention_mask.to(hidden.device, dtype=hidden.dtype)[:, :, None] * delta
        return (modified,) + result[1:] if isinstance(result, tuple) else modified

    def __exit__(self, *exception):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


def bert_tail_logits(model, hidden, attention_mask, after_layer):
    """Continue BERT from a cached block output; tested against a full forward.

    after_layer uses the same 1-indexed convention as ActivationSteering.
    This avoids repeating the unchanged prefix for every intervention strength.
    """
    extended = model.bert.get_extended_attention_mask(attention_mask, attention_mask.shape)
    for block in model.bert.encoder.layer[after_layer:]:
        hidden = block(hidden, attention_mask=extended)[0]
    return model.classifier(model.dropout(model.bert.pooler(hidden)))
