import pytest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.network import HybridNetwork

def test_network_initialization():
    model = HybridNetwork(grid_size=32, latent_dim=64)
    assert isinstance(model, HybridNetwork)

def test_forward_autoencoder():
    model = HybridNetwork(grid_size=32, latent_dim=64)
    dummy_input = torch.randn(1, 1, 32, 32, 32)
    output, latent = model.forward_autoencoder(dummy_input)
    assert output.shape == (1, 1, 32, 32, 32)
    assert latent.shape == (1, 64)

def test_forward_params():
    model = HybridNetwork(grid_size=32, latent_dim=64)
    dummy_params = torch.randn(1, 4)  # center (3) + radius (1)
    output = model.forward_params(dummy_params)
    assert output.shape == (1, 1, 32, 32, 32)
