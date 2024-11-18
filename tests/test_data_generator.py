import pytest
import yaml
from src.data_generator import SDFGenerator

def test_config_loading():
    """Test loading configuration from the YAML file."""
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    assert config is not None
    assert "grid_size" in config

def test_sphere_sdf_generation():
    """Test SDF batch generation."""
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    generator = SDFGenerator(config)

    # Test batch generation
    sdf_batch, params_batch = generator.generate_sphere_sdf_batch(batch_size=16)

    assert sdf_batch.shape[0] == 16  # Batch size
    assert params_batch.shape[0] == 16
    assert sdf_batch.ndim == 5  # Ensure correct dimensions: [batch, channels, depth, height, width]
    assert params_batch.ndim == 2  # [batch, params]
