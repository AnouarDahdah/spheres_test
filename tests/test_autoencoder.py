import torch
import pytest
from src.autoencoder import SDF_Autoencoder

def test_autoencoder_initialization():
    model = SDF_Autoencoder(grid_res=30, latent_dim=64)
    
    assert model is not None
    assert len(list(model.parameters())) > 0

def test_autoencoder_forward_pass():
    model = SDF_Autoencoder(grid_res=30, latent_dim=64)
    
    # Create a sample input tensor
    x = torch.randn(4, 1, 30, 30, 30)  # [batch, channels, depth, height, width]
    
    # Run forward pass
    output = model(x)
    
    # Check output shape matches input
    assert output.shape == x.shape
