"""Synthetic mechanism demonstrations, explicitly separate from IMDb evidence."""
from __future__ import annotations

import math
import time
import numpy as np
import pandas as pd
import torch
from scipy.linalg import svd

from .common import save_json
from .didactic import DROOT, settings, array_hash


def quadratic():
    """Full-batch geometry example; no minibatch noise in this panel."""
    rows = []
    for name, lr, momentum in [("gd_stable", .08, 0.), ("gd_unstable", .102, 0.),
                               ("momentum", .03, .9), ("adam", .1, 0.)]:
        x = torch.tensor([4., 3.], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([x], lr=lr) if name == "adam" else torch.optim.SGD([x], lr=lr, momentum=momentum)
        for step in range(121):
            loss = (.5 * x[0].square() + 20 * x[1].square()) / 2
            rows.append({"condition": name, "lr": lr, "step": step, "x": float(x[0].detach()), "y": float(x[1].detach()), "loss": float(loss.detach())})
            if step < 120:
                opt.zero_grad(); loss.backward(); opt.step()
    pd.DataFrame(rows).to_csv(DROOT / "quadratic.csv", index=False)
    save_json(DROOT / "quadratic_metadata.json", {"hessian_eigenvalues": [.5, 20.], "gd_stability_upper_bound": .1,
              "formula": "L(x,y)=(0.5*x^2+20*y^2)/2", "initial": [4., 3.],
              "caveat": "Deterministic full-batch toy; chosen rates illustrate geometry, not a fair optimizer ranking. Real minibatch comparisons are separate."})


def initialization():
    spec = settings()["initialization"]; rows = []
    torch.set_num_threads(settings()["threads"])
    for seed in settings()["seeds"]:
        gen = torch.Generator().manual_seed(seed)
        source = torch.randn(spec["n"], spec["width"], generator=gen, dtype=torch.float64)
        weights = [torch.randn(spec["width"], spec["width"], generator=gen, dtype=torch.float64) for _ in range(spec["depth"])]
        readout = torch.randn(spec["width"], generator=gen, dtype=torch.float64)
        readout /= readout.norm()
        for gain in spec["gains"]:
            x = source.clone().requires_grad_(True); values = [x]
            for z in weights:
                x = torch.relu(x @ (gain * math.sqrt(2 / spec["width"]) * z))
                x.retain_grad(); values.append(x)
            # A fixed linear probe supplies the same output gradient at every gain.
            (x @ readout).sum().backward()
            for layer, value in enumerate(values):
                rows.append({"seed": seed, "gain": gain, "layer": layer,
                             "activation_rms": float(value.detach().square().mean().sqrt()),
                             "activation_variance": float(value.detach().var(unbiased=False)),
                             "gradient_rms": float(value.grad.square().mean().sqrt()),
                             "zero_fraction": float((value.detach() == 0).double().mean())})
    pd.DataFrame(rows).to_csv(DROOT / "initialization.csv", index=False)
    save_json(DROOT / "initialization_metadata.json", {"data": "synthetic independent standard Gaussian inputs", "dtype": "float64",
              "variance": "weight variance = gain^2 * 2 / width; gain=1 is He for ReLU",
              "paired_controls": "same input, base Gaussian weights and linear readout for all gains of a seed",
              "no_training": True, "caveat": "Finite-width diagnostic. He is a scale-preservation approximation, not a guarantee."})


def ridge_weights(phi, y, lam, rcond=1e-12):
    u, s, vt = svd(phi, full_matrices=False, check_finite=True, lapack_driver="gesdd")
    active = s > rcond * s[0]
    if lam == 0:
        factors = np.zeros_like(s); factors[active] = 1 / s[active]
    else:
        factors = s / (s * s + len(phi) * lam)
    weights = vt.T @ (factors * (u.T @ y))
    effective_df = float(active.sum()) if lam == 0 else float(np.sum(s * s / (s * s + len(phi) * lam)))
    return weights, {"rank": int(active.sum()), "effective_df": effective_df,
                     "s_max": float(s[0]), "s_min": float(s[-1]), "condition_number": float(s[0] / s[-1])}


def synthetic_case(seed, spec):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(spec["train_n"], spec["input_dim"]))
    xt = rng.normal(size=(spec["test_n"], spec["input_dim"]))
    f = x[:, 0] + .5 * np.sin(x[:, 1]); ft = xt[:, 0] + .5 * np.sin(xt[:, 1])
    y = f + spec["noise_sd"] * rng.normal(size=len(x))
    yt = ft + spec["noise_sd"] * rng.normal(size=len(xt))
    w = rng.normal(size=(spec["input_dim"], max(spec["widths"]))) / math.sqrt(spec["input_dim"])
    return x, xt, y, yt, f, ft, w


def double_descent():
    spec = settings()["double_descent"]; rows = []; weights = {}
    started = time.perf_counter()
    for seed in spec["seeds"]:
        x, xt, y, yt, f, ft, hidden = synthetic_case(seed, spec)
        all_train, all_test = np.maximum(x @ hidden, 0), np.maximum(xt @ hidden, 0)
        for width in spec["widths"]:
            phi = all_train[:, :width] / math.sqrt(width)
            test_phi = all_test[:, :width] / math.sqrt(width)
            for lam in spec["lambdas"]:
                beta, diagnostics = ridge_weights(phi, y, lam, spec["rcond"])
                pred, pred_t = phi @ beta, test_phi @ beta
                key = f"seed{seed}_p{width}_lambda{lam:g}"; weights[key] = beta
                rows.append({"seed": seed, "p": width, "p_over_n": width / len(x), "lambda": lam,
                             "train_mse": float(np.mean((pred - y) ** 2)), "test_mse": float(np.mean((pred_t - yt) ** 2)),
                             "test_signal_mse": float(np.mean((pred_t - ft) ** 2)), "coefficient_norm": float(np.linalg.norm(beta)),
                             "train_data_sha256": array_hash(np.column_stack([x, y])), "weights_key": key, **diagnostics})
        print("Synthetic capacity sweep completed", seed, flush=True)
    pd.DataFrame(rows).to_csv(DROOT / "double_descent.csv", index=False)
    np.savez_compressed(DROOT / "double_descent_weights.npz", **weights)
    save_json(DROOT / "double_descent_metadata.json", {"seconds": time.perf_counter() - started, "cases": len(rows),
              "spec": spec, "all_prescribed_seeds_retained": True, "trained_parameters": "p readout coefficients only; hidden Gaussian weights are frozen",
              "nested_features": True, "test_selection": "No settings or seeds selected on synthetic test metrics.",
              "caveat": "Synthetic regression with squared loss and a frozen random ReLU layer. Not end-to-end CNN training or evidence that IMDb exhibits double descent."})


def run():
    DROOT.mkdir(parents=True, exist_ok=True)
    quadratic(); initialization(); double_descent()
