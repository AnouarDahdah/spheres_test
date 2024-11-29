import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class SDFGenerator:
    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size
        self.grid = self._create_grid()

    def _create_grid(self):
        x = np.linspace(-1, 1, self.grid_size)
        y = np.linspace(-1, 1, self.grid_size)
        z = np.linspace(-1, 1, self.grid_size)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return np.stack([X, Y, Z], axis=-1)

    def generate_sphere_sdf(self, center, radius):
        center = np.array(center)
        points = self.grid.reshape(-1, 3)
        distances = np.linalg.norm(points - center, axis=1)
        sdf = distances - radius
        return torch.FloatTensor(sdf.reshape(self.grid_size, self.grid_size, self.grid_size))

class SDFDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.generator = SDFGenerator(config.grid_size)
        
        # Generate dataset
        center_range = 1.0 - config.boundary_margin - config.max_radius
        self.centers = []
        self.radii = []
        self.sdfs = []
        
        dataset_size = config.dataset_size
        if split == 'val':
            dataset_size = dataset_size // 10
            
        for _ in tqdm(range(dataset_size), desc=f'Generating {split} dataset'):
            # Random center within bounds
            center = np.random.uniform(-center_range, center_range, 3)
            radius = np.random.uniform(config.min_radius, config.max_radius)
            
            # Generate SDF
            sdf = self.generator.generate_sphere_sdf(center, radius)
            
            self.centers.append(torch.FloatTensor(center))
            self.radii.append(torch.FloatTensor([radius]))
            self.sdfs.append(sdf)

    def __len__(self):
        return len(self.sdfs)

    def __getitem__(self, idx):
        center = self.centers[idx]
        radius = self.radii[idx]
        sdf = self.sdfs[idx]
        
        params = torch.cat([center, radius])
        sdf = sdf.unsqueeze(0)  # Add channel dimension
        
        # Data augmentation for training
        if self.split == 'train':
            # Random rotations could be added here
            pass
            
        return params, sdf
