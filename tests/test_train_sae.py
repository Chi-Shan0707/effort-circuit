import torch

from src.train_sae import SparseAutoencoder


def test_sparse_autoencoder_forward_shapes():
    model = SparseAutoencoder(input_dim=4, hidden_dim=8)
    reconstruction, features = model(torch.ones(3, 4))
    assert reconstruction.shape == (3, 4)
    assert features.shape == (3, 8)
    assert torch.all(features >= 0)
