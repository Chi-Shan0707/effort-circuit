from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class ActivationRecord:
    name: str
    tensor: torch.Tensor


@dataclass
class ActivationCache:
    records: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def add(self, name: str, tensor: torch.Tensor) -> None:
        self.records.setdefault(name, []).append(tensor.detach().float().cpu())

    def clear(self) -> None:
        self.records.clear()

    def latest(self, name: str) -> torch.Tensor:
        return self.records[name][-1]


def get_layers(model) -> list[object]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise AttributeError("Could not locate transformer layers on model")


def parse_layers(layers: str, model=None) -> list[int]:
    if layers == "all":
        if model is None:
            raise ValueError("model is required when layers='all'")
        return list(range(len(get_layers(model))))
    selected: list[int] = []
    for item in layers.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            selected.extend(range(int(start), int(end) + 1))
        else:
            selected.append(int(item))
    return selected


def site_module(layer, site: str):
    if site == "resid_post":
        return layer
    if site == "mlp_act":
        mlp = getattr(layer, "mlp", None)
        for name in ("act_fn", "activation_fn", "act"):
            if mlp is not None and hasattr(mlp, name):
                return getattr(mlp, name)
        return mlp
    if site == "mlp_out":
        mlp = getattr(layer, "mlp", None)
        for name in ("down_proj", "c_proj", "dense_4h_to_h"):
            if mlp is not None and hasattr(mlp, name):
                return getattr(mlp, name)
        return mlp
    raise ValueError(f"Unsupported activation site {site!r}")


def _output_tensor(output) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def register_activation_hooks(model, layers: list[int], sites: list[str], cache: ActivationCache):
    handles = []
    all_layers = get_layers(model)
    for layer_idx in layers:
        layer = all_layers[layer_idx]
        for site in sites:
            module = site_module(layer, site)
            if module is None:
                continue
            name = f"layer{layer_idx}.{site}"

            def hook(_module, _inputs, output, hook_name=name):
                cache.add(hook_name, _output_tensor(output))

            handles.append(module.register_forward_hook(hook))
    return handles


def make_residual_steering_hook(vector: torch.Tensor, alpha: float, positions: slice | int | None = None) -> Callable:
    direction = vector.detach()

    def hook(_module, _inputs, output):
        tensor = _output_tensor(output)
        steered = tensor.clone()
        pos = slice(None) if positions is None else positions
        steered[:, pos, :] = steered[:, pos, :] + alpha * direction.to(device=steered.device, dtype=steered.dtype)
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    return hook
