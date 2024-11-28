import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data_generator import SDFGenerator, SDFDataset

def test_sdf_generator_initialization():
    generator = SDFGenerator(grid_size=32)
    assert generator.grid_size == 32
    assert isinstance(generator.grid, torch.Tensor)
    assert generator.grid.shape == (32, 32, 32, 3)

def test_grid_creation():
    generator = SDFGenerator(grid_size=16)
    grid = generator._create_grid()
    assert grid.shape == (16, 16, 16, 3)
    assert torch.allclose(grid.min(), torch.tensor(-1.0))
    assert torch.allclose(grid.max(), torch.tensor(1.0))

def test_sphere_sdf_generation():
    generator = SDFGenerator(grid_size=32)
    center = [0.0, 0.0, 0.0]
    radius = 0.3
    sdf = generator.generate_sphere_sdf(center, radius)
    
    assert sdf.shape == (32, 32, 32)
    assert (sdf[16, 16, 16] < 0).item()  # Center should be inside (negative)
    assert (sdf[0, 0, 0] > 0).item()     # Corners should be outside (positive)

def test_dataset_creation():
    dataset = SDFDataset(size=10, grid_size=16, train=True)
    assert len(dataset) == 8  # 80% of 10 for training
    
    params, sdf = dataset[0]
    assert params.shape == (4,)           # 3 for center, 1 for radius
    assert sdf.shape == (1, 16, 16, 16)   # Grid shape with batch dimension

def test_dataset_split():
    dataset_train = SDFDataset(size=100, grid_size=32, train=True, split=0.8)
    dataset_test = SDFDataset(size=100, grid_size=32, train=False, split=0.8)
    
    assert len(dataset_train) == 80
    assert len(dataset_test) == 20

def test_parameter_ranges():
    dataset = SDFDataset(size=50, grid_size=32)
    params, _ = zip(*[dataset[i] for i in range(len(dataset))])
    params = torch.stack(params)
    
    # Check center coordinates are within [-0.5, 0.5]
    assert (params[:, :3] >= -0.5).all() and (params[:, :3] <= 0.5).all()
    
    # Check radius is within [0.2, 0.3]
    assert (params[:, 3] >= 0.2).all() and (params[:, 3] <= 0.3).all()
