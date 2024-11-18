
import torch
import numpy as np

class SDFGenerator:
    def __init__(self, config):
        self.config = config
        
    def generate_sphere_sdf_batch(self, batch_size=32):
        """
        Generate sphere SDFs with configurable parameters
        """
        grid_res = self.config['model']['grid_res']
        grid_min = self.config['data']['grid_min']
        grid_max = self.config['data']['grid_max']
        
        sdf_batch = []
        params_batch = []
        for _ in range(batch_size):
            x = torch.linspace(grid_min, grid_max, grid_res)
            y = torch.linspace(grid_min, grid_max, grid_res)
            z = torch.linspace(grid_min, grid_max, grid_res)
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            grid_points = torch.stack([xx.ravel(), yy.ravel(), zz.ravel()], dim=1)

            # Ensure sphere is centered very close to origin
            center = np.random.uniform(
                self.config['data']['sphere']['center_range'][0],
                self.config['data']['sphere']['center_range'][1], 
                3
            )
            radius = np.random.uniform(
                *self.config['data']['sphere']['radius_range']
            )

            sphere_center = torch.tensor(center, dtype=torch.float32)
            grid_points_tensor = grid_points.clone().detach()
            distances = torch.norm(grid_points_tensor - sphere_center, dim=1)
            sdf_values = distances - radius

            sdf_values = (sdf_values - sdf_values.mean()) / sdf_values.std()
            sdf_values_3d = sdf_values.view(grid_res, grid_res, grid_res)

            sdf_batch.append(sdf_values_3d)
            params_batch.append(torch.tensor([*center, radius], dtype=torch.float32))

        return torch.stack(sdf_batch).unsqueeze(1), torch.stack(params_batch)

