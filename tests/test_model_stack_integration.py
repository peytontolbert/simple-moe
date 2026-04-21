from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import _model_stack as model_stack_mod
from src.models.moe import MixtureOfExperts
from src.models.router import Router


def _bind_method(instance, fn):
    return fn.__get__(instance, type(instance))


def test_router_uses_runtime_topk_helper(monkeypatch):
    captured = {}
    expected_weights = torch.tensor(
        [[0.75, 0.25], [0.10, 0.90], [0.60, 0.40]],
        dtype=torch.float32,
    )
    expected_indices = torch.tensor(
        [[1, 0], [2, 1], [0, 2]],
        dtype=torch.long,
    )

    def fake_topk_route(scores, *, k, capacity_factor, drop_policy):
        captured["shape"] = tuple(scores.shape)
        captured["k"] = k
        captured["capacity_factor"] = capacity_factor
        captured["drop_policy"] = drop_policy
        return expected_weights.to(device=scores.device, dtype=scores.dtype), expected_indices.to(device=scores.device)

    monkeypatch.setattr(model_stack_mod, "topk_route", fake_topk_route)

    router = Router(input_dim=4, num_experts=3, k=2, capacity_factor=1.25)
    router.train()
    x = torch.randn(3, 4)

    weights, indices, aux_loss = router(x)

    assert captured == {
        "shape": (3, 3),
        "k": 2,
        "capacity_factor": 1.25,
        "drop_policy": "dropless",
    }
    assert torch.allclose(weights, expected_weights)
    assert torch.equal(indices, expected_indices)
    assert aux_loss is not None


def test_moe_forward_matches_naive_reference(monkeypatch):
    torch.manual_seed(0)
    model = MixtureOfExperts(
        input_dim=4,
        output_dim=3,
        num_experts=3,
        k=2,
        expert_kwargs={"hidden_dim": 8, "dropout": 0.0},
    )
    model.eval()

    routing_weights = torch.tensor(
        [
            [0.9, 0.1],
            [0.3, 0.7],
            [0.5, 0.5],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    routing_indices = torch.tensor(
        [
            [0, 2],
            [1, 0],
            [2, 1],
            [1, 2],
        ],
        dtype=torch.long,
    )

    def fake_router_forward(_self, x):
        return routing_weights.to(device=x.device, dtype=x.dtype), routing_indices.to(device=x.device), None

    monkeypatch.setattr(model.router, "forward", _bind_method(model.router, fake_router_forward))

    x = torch.randn(4, 4)
    actual, aux_loss = model(x)

    expected = torch.zeros(4, 3, dtype=x.dtype)
    for i in range(model.k):
        expert_indices = routing_indices[:, i]
        expert_weights = routing_weights[:, i].unsqueeze(1)
        expert_outputs = torch.stack(
            [
                model.experts[int(expert_idx)](x[row_idx: row_idx + 1])
                for row_idx, expert_idx in enumerate(expert_indices.tolist())
            ],
            dim=0,
        ).squeeze(1)
        expected += expert_outputs * expert_weights

    assert aux_loss is None
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_moe_uses_runtime_linear_for_router_and_experts(monkeypatch):
    torch.manual_seed(1)
    model = MixtureOfExperts(
        input_dim=6,
        output_dim=2,
        num_experts=2,
        k=2,
        expert_kwargs={"hidden_dim": 10, "num_layers": 3, "dropout": 0.0},
    )
    model.eval()

    calls = []

    def fake_runtime_linear(module, x):
        calls.append(module)
        return module(x)

    def fake_topk_route(scores, *, k, capacity_factor, drop_policy):
        weights = torch.tensor(
            [[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.2, 0.8]],
            device=scores.device,
            dtype=scores.dtype,
        )
        indices = torch.tensor(
            [[0, 1], [1, 0], [0, 1], [1, 0]],
            device=scores.device,
            dtype=torch.long,
        )
        return weights, indices

    monkeypatch.setattr(model_stack_mod, "runtime_linear", fake_runtime_linear)
    monkeypatch.setattr(model_stack_mod, "topk_route", fake_topk_route)

    x = torch.randn(4, 6)
    y, _ = model(x)

    expert_linear_count = sum(
        1
        for expert in model.experts
        for layer in expert.network
        if isinstance(layer, torch.nn.Linear)
    )
    assert y.shape == (4, 2)
    assert len(calls) == 1 + expert_linear_count
    assert calls[0] is model.router.router
