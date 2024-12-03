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
        grid = np.stack([X, Y, Z], axis=-1)
        return grid

    def generate_sphere_sdf(self, center, radius):
        center = np.array(center)
        squared_distances = np.sum((self.grid - center.reshape(1, 1, 1, 3)) ** 2, axis=-1)
        distances = np.sqrt(squared_distances)
        sdf = distances - radius
        return torch.FloatTensor(sdf).unsqueeze(0)

class SDFDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.generator = SDFGenerator(config.grid_size)
        
        center_range = 1.0 - config.boundary_margin - config.max_radius
        self.centers = []
        self.radii = []
        self.sdfs = []
        
        dataset_size = config.dataset_size
        if split == 'val':
            dataset_size = dataset_size // 10
            
        for _ in tqdm(range(dataset_size), desc=f'Generating {split} dataset'):
            center = np.random.uniform(-center_range, center_range, 3)
            radius = np.random.uniform(config.min_radius, config.max_radius)
            sdf = self.generator.generate_sphere_sdf(center, radius)
            
            self.centers.append(torch.FloatTensor(center))
            self.radii.append(torch.FloatTensor([radius]))
            self.sdfs.append(sdf)

    def __len__(self):
        return len(self.sdfs)

    def __getitem__(self, idx):
        params = torch.cat([self.centers[idx], self.radii[idx]])
        sdf = self.sdfs[idx]
        return params, sdf