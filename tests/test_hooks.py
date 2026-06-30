import torch
from torch import nn

from src.hooks import ActivationCache, get_layers, parse_layers, register_activation_hooks


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))

    def forward(self, x):
        return x + self.mlp(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock(), TinyBlock()])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_parse_layers_all_and_ranges():
    model = TinyModel()
    assert len(get_layers(model)) == 2
    assert parse_layers("all", model) == [0, 1]
    assert parse_layers("0-1") == [0, 1]


def test_register_resid_hook_captures_output():
    model = TinyModel()
    cache = ActivationCache()
    handles = register_activation_hooks(model, [0], ["resid_post"], cache)
    try:
        _ = model(torch.randn(1, 3, 4))
        assert "layer0.resid_post" in cache.records
        assert cache.latest("layer0.resid_post").shape == (1, 3, 4)
    finally:
        for handle in handles:
            handle.remove()
