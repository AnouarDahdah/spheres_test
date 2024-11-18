
import pytest
import torch
import yaml
from src.data_generator import SDFGenerator

def test_config_loading():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    assert 'model' in config
    assert 'data' in config

def test_sphere_sdf_generation():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    generator = SDFGenerator(config)
    
    # Test batch generation
    sdf_batch, params_batch = generator.generate_sphere_sdf_batch(batch_size=16)
    
    assert sdf_batch.shape[0] == 16
    assert params_batch.shape[0] == 16
    
    assert sdf_batch.ndim == 4  # [batch, channels, height, width, depth]
    assert params_batch.ndim == 2  # [batch, params]
    
    assert sdf_batch.shape[1] == 1  # Single channel
    assert params_batch.shape[1] == 4  # x, y, z, radius

