from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from runtime.moe import topk_router as _runtime_topk_router
except Exception:
    _runtime_topk_router = None

try:
    from runtime.ops import linear_module as _runtime_linear_module
except Exception:
    _runtime_linear_module = None


def runtime_linear(module, x: torch.Tensor) -> torch.Tensor:
    if _runtime_linear_module is not None:
        return _runtime_linear_module(x, module)
    return module(x)


def topk_route(
    scores: torch.Tensor,
    *,
    k: int,
    capacity_factor: float = 1.0,
    drop_policy: str = "dropless",
) -> tuple[torch.Tensor, torch.Tensor]:
    if _runtime_topk_router is not None:
        flat_scores = scores.reshape(-1, 1, scores.shape[-1])
        assignments, weights = _runtime_topk_router(
            flat_scores,
            k=int(k),
            capacity_factor=float(capacity_factor),
            drop_policy=str(drop_policy),
        )
        weights = weights.reshape(*scores.shape[:-1], int(k))
        assignments = assignments.reshape(*scores.shape[:-1], int(k))
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        return weights / denom, assignments

    weights, assignments = torch.topk(scores, int(k), dim=-1)
    return F.softmax(weights, dim=-1), assignments
