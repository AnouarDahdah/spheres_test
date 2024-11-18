import yaml
from src.data_generator import SDFGenerator

def test_config_loading():
    """Test loading configuration from the YAML file."""
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    assert config is not None
    assert "model" in config  # Ensure "model" section exists
    assert "grid_size" in config["model"]  # Check "grid_size" exists under "model"

def test_sphere_sdf_generation():
    """Test generating a batch of SDFs and parameters."""
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    generator = SDFGenerator(config)
    
    # Test batch generation
    sdf_batch, params_batch = generator.generate_sphere_sdf_batch(batch_size=16)
    
    assert sdf_batch.shape[0] == 16
    assert params_batch.shape[0] == 16
    assert sdf_batch.ndim == 5  # [batch, channels, height, width, depth]
