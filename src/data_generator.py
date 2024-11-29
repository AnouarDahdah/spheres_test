import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class SDFGenerator:
    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size
        self.grid = self._create_grid()
        
    def _create_grid(self):
        x = torch.linspace(-1, 1, self.grid_size)
        coords = torch.meshgrid(x, x, x, indexing='ij')
        return torch.stack(coords, dim=-1)

    def generate_sphere_sdf(self, center, radius):
        center = torch.tensor(center, dtype=torch.float32)
        distances = torch.norm(self.grid - center.view(1, 1, 1, 3), dim=-1)
        return distances - radius

class SDFDataset(Dataset):
    def __init__(self, size: int, grid_size: int, train: bool = True, split: float = 0.8):
        self.generator = SDFGenerator(grid_size=grid_size)
        params_list = []
        sdf_list = []
        
        for _ in tqdm(range(size), desc='Generating dataset'):
            center = np.random.uniform(-0.5, 0.5, size=3)
            radius = np.random.uniform(0.2, 0.3)
            
            sdf = self.generator.generate_sphere_sdf(center, radius)
            params_list.append(torch.tensor([*center, radius], dtype=torch.float32))
            sdf_list.append(sdf.unsqueeze(0))
        
        data = torch.stack(params_list), torch.stack(sdf_list)
        split_idx = int(split * size)
        self.params, self.sdfs = [x[:(split_idx if train else -split_idx)] for x in data]
    
    def __len__(self):
        return len(self.params)
    
    def __getitem__(self, idx):
        return self.params[idx], self.sdfs[idx]
