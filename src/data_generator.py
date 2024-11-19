import torch
import numpy as np

class SDFGenerator:
    def __init__(self, config):
        self.grid_size = config['model']['grid_size']
        # Add grid parameters
        self.grid_min = -3
        self.grid_max = 3
        
    def generate_sphere_sdf_batch(self, batch_size):
        """Generate a batch of sphere SDFs and corresponding parameters.
        
        Args:
            batch_size (int): Number of samples to generate
            
        Returns:
            tuple: (sdf_batch, params_batch)
                - sdf_batch: torch.Tensor of shape [batch, 1, depth, height, width]
                - params_batch: torch.Tensor of shape [batch, 4] containing [center_x, center_y, center_z, radius]
        """
        # Create coordinate grids
        x = torch.linspace(self.grid_min, self.grid_max, self.grid_size[0])
        y = torch.linspace(self.grid_min, self.grid_max, self.grid_size[1])
        z = torch.linspace(self.grid_min, self.grid_max, self.grid_size[2])
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.stack([xx.ravel(), yy.ravel(), zz.ravel()], dim=1)
        
        sdf_batch = []
        params_batch = []
        
        for _ in range(batch_size):
            # Generate random center and radius
            center = np.random.uniform(self.grid_min + 1, self.grid_max - 1, 3)
            max_radius = np.min([np.abs(center - self.grid_min), np.abs(center - self.grid_max)])
            radius = np.random.uniform(0.5, max_radius)
            
            # Convert to torch tensors
            sphere_center = torch.tensor(center, dtype=torch.float32)
            
            # Calculate distances and SDF values
            grid_points_tensor = grid_points.clone().detach()
            distances = torch.norm(grid_points_tensor - sphere_center, dim=1)
            sdf_values = distances - radius
            
            # Normalize SDF values
            sdf_values = (sdf_values - sdf_values.mean()) / sdf_values.std()
            
            # Reshape to 3D grid
            sdf_values_3d = sdf_values.view(*self.grid_size)
            
            sdf_batch.append(sdf_values_3d)
            params_batch.append(torch.tensor([*center, radius], dtype=torch.float32))
        
        # Stack batches
        sdf_batch = torch.stack(sdf_batch)
        params_batch = torch.stack(params_batch)
        
        return sdf_batch.unsqueeze(1), params_batch
