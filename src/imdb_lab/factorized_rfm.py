"""Exact low-rank algebra for the same Laplace RFM, without a dense d x d metric.

The initial metric is the FULL identity, including directions outside the
training sample span. Later metrics are stored as B.T @ B with B=G/sqrt(n).
This is an exact representation of the raw AGOP update, not PCA or a sketch.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score


def factor_distances(x, z, factor):
    xx, zz = (x, z) if factor is None else (x @ factor.T, z @ factor.T)
    squared = (xx * xx).sum(1)[:, None] + (zz * zz).sum(1)[None, :] - 2 * xx @ zz.T
    return np.sqrt(np.maximum(squared, 0))


def factor_gradients(x, centers, factor, alpha, bandwidth, chunk=128):
    cm = centers if factor is None else (centers @ factor.T) @ factor
    center_sq = (centers * centers).sum(1)
    output = []
    for start in range(0, len(x), chunk):
        block = x[start:start + chunk]
        distance = factor_distances(block, centers, factor)
        euclid_sq = (block * block).sum(1)[:, None] + center_sq[None, :] - 2 * block @ centers.T
        ok = (distance > 1e-10) & (np.abs(euclid_sq) >= 1e-10)
        weight = np.divide(np.exp(-distance / bandwidth), distance,
                           out=np.zeros_like(distance), where=ok) * alpha[None, :]
        bm = block if factor is None else (block @ factor.T) @ factor
        output.append(-(weight.sum(1)[:, None] * bm - weight @ cm) / bandwidth)
    return np.concatenate(output)


@dataclass
class FactorizedLaplacianRFM:
    bandwidth: float = 30.
    regularization: float = .001
    iterations: int = 4

    def fit(self, x, y, x_val, y_val):
        x, x_val = np.asarray(x, dtype=np.float64), np.asarray(x_val, dtype=np.float64)
        y, y_val = np.asarray(y, dtype=np.float64), np.asarray(y_val, dtype=np.float64)
        self.mean = x.mean(0)
        self.scale = max(float(np.sqrt(np.mean((x - self.mean) ** 2))), 1e-8)
        self.centers = (x - self.mean) / self.scale
        val = (x_val - self.mean) / self.scale
        self.intercept = float(y.mean())
        factor, best = None, (-np.inf, -np.inf)
        self.history = []
        for iteration in range(self.iterations):
            kernel = np.exp(-factor_distances(self.centers, self.centers, factor) / self.bandwidth)
            alpha = cho_solve(cho_factor(kernel + self.regularization * np.eye(len(x)), lower=True), y - self.intercept)
            pred = np.exp(-factor_distances(val, self.centers, factor) / self.bandwidth) @ alpha + self.intercept
            auc, mse = float(roc_auc_score(y_val, pred)), float(np.mean((pred - y_val) ** 2))
            gradients = factor_gradients(self.centers, self.centers, factor, alpha, self.bandwidth)
            agop_factor = gradients / np.sqrt(len(x))
            self.history.append({"iteration": iteration + 1, "validation_auc": auc,
                                 "validation_mse": mse, "agop_trace": float(np.sum(agop_factor ** 2))})
            if (auc, -mse) > best:
                best = (auc, -mse)
                self.metric_factor = None if factor is None else factor.copy()
                self.alpha, self.agop_factor = alpha.copy(), agop_factor.copy()
                self.selected_iteration, self.validation_auc = iteration + 1, auc
            factor = agop_factor
        # Nonzero eigenvalues of B.T B equal those of B B.T. Lift its top vector.
        self.eigenvalues, u = np.linalg.eigh(self.agop_factor @ self.agop_factor.T)
        vector = self.agop_factor.T @ u[:, -1]
        vector /= np.linalg.norm(vector)
        if np.cov(x @ vector, y)[0, 1] < 0:
            vector = -vector
        self.direction = vector
        self.projection_std = float(np.std(x @ vector, ddof=1))
        return self

    def predict(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return np.exp(-factor_distances(z, self.centers, self.metric_factor) / self.bandwidth) @ self.alpha + self.intercept

    def save(self, path):
        np.savez_compressed(path, centers=self.centers, mean=self.mean, scale=self.scale,
                            metric_factor=np.empty((0, self.centers.shape[1])) if self.metric_factor is None else self.metric_factor,
                            metric_is_identity=self.metric_factor is None, alpha=self.alpha,
                            agop_factor=self.agop_factor, direction=self.direction,
                            eigenvalues=self.eigenvalues, projection_std=self.projection_std,
                            bandwidth=self.bandwidth, regularization=self.regularization,
                            intercept=self.intercept, selected_iteration=self.selected_iteration)
