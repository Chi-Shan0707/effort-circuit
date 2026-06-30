from __future__ import annotations

from contextlib import contextmanager

import torch

from .hooks import get_layers, make_residual_steering_hook


@contextmanager
def residual_steering(model, layer: int, vector: torch.Tensor, alpha: float, positions=None):
    handle = get_layers(model)[layer].register_forward_hook(make_residual_steering_hook(vector, alpha, positions))
    try:
        yield
    finally:
        handle.remove()


def load_vector(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "vector" in payload:
        return payload["vector"].float(), int(payload.get("layer", 0))
    return payload.float(), 0
