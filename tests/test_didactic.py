import numpy as np
import pandas as pd
import torch

from imdb_lab.didactic import (TeachingMLP, balanced_partition, early_stop_trace,
                               epoch_orders, explicit_l2, state_hash, train_one)
from imdb_lab.didactic_toys import ridge_weights


def test_balanced_roles_are_disjoint_and_reproducible():
    frame = pd.DataFrame({"id": [f"row{i}" for i in range(40)], "text_sha256": [str(i) for i in range(40)],
                          "label": [i % 2 for i in range(40)]})
    roles = balanced_partition(frame, {"train": 16, "validation": 8, "audit": 8}, 7)
    for name, part in roles.items():
        assert part.label.value_counts().nunique() == 1
        for other in roles:
            if name != other: assert set(part.id).isdisjoint(roles[other].id)
    assert roles["train"].equals(balanced_partition(frame, {"train": 16, "validation": 8, "audit": 8}, 7)["train"])


def test_residual_and_plain_have_equal_parameters_and_zero_branch_is_identity():
    torch.manual_seed(4); plain = TeachingMLP(7, 8, 4)
    torch.manual_seed(4); residual = TeachingMLP(7, 8, 4, residual=True)
    assert state_hash(plain) == state_hash(residual)
    for layer in residual.blocks:
        torch.nn.init.zeros_(layer.weight); torch.nn.init.zeros_(layer.bias)
    x = torch.randn(3, 7)
    assert torch.allclose(residual(x), residual.head(torch.relu(residual.stem(x))))


def test_explicit_l2_excludes_bias_and_matches_gradient():
    model = TeachingMLP(3, 4, 2).double()
    value = explicit_l2(model); value.backward()
    for name, p in model.named_parameters():
        if name.endswith("weight"): torch.testing.assert_close(p.grad, 2 * p)
        else: assert p.grad is None


def test_early_stop_decision_never_uses_future_loss():
    prefix = [1.0, .8, .81, .82, .83]
    assert early_stop_trace(prefix, 3, .001) == (2, 5)
    assert early_stop_trace(prefix + [.01, .001], 3, .001) == (2, 5)


def test_ridge_svd_matches_primal_and_minimum_norm_interpolant():
    rng = np.random.default_rng(5)
    phi, y = rng.normal(size=(7, 12)), rng.normal(size=7)
    beta, info = ridge_weights(phi, y, 0)
    np.testing.assert_allclose(beta, np.linalg.lstsq(phi, y, rcond=1e-12)[0], atol=1e-10)
    np.testing.assert_allclose(phi @ beta, y, atol=1e-10)
    lam = .1
    beta, info = ridge_weights(phi, y, lam)
    expected = np.linalg.solve(phi.T @ phi + len(phi) * lam * np.eye(phi.shape[1]), phi.T @ y)
    np.testing.assert_allclose(beta, expected, atol=1e-10)
    assert 0 < info["effective_df"] < 7


def test_minibatch_sequences_are_paired_without_replacement():
    orders = epoch_orders(31, 4, 8)
    assert all(np.array_equal(np.sort(order), np.arange(31)) for order in orders)
    assert np.array_equal(orders[0], epoch_orders(31, 4, 8)[0])
    assert not np.array_equal(orders[0], orders[1])


def test_real_training_uses_paired_start_and_records_complete_epochs():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(32, 6)).astype(np.float32); y = (x[:, 0] > 0).astype(np.int64)
    data = {"x": x, "y": y}; spec = {"input_dim": 6, "width": 8, "depth": 2}
    a = train_one(data, data, spec, "sgd", .03, 7, 3, 16, torch.device("cpu"))
    b = train_one(data, data, spec, "adam", .003, 7, 3, 16, torch.device("cpu"))
    assert a["initial_state_sha256"] == b["initial_state_sha256"]
    assert a["minibatch_order_sha256"] == b["minibatch_order_sha256"]
    assert [r["updates"] for r in a["history"]] == [0, 2, 4, 6]
    assert checkpoint_bytes(a["final_state"]) != checkpoint_bytes(b["final_state"])


def checkpoint_bytes(state):
    return np.concatenate([v.numpy().ravel() for v in state.values()]).tobytes()
