import torch
import numpy as np

class SDFGenerator:
    def __init__(self, config):
        """Initialize SDF Generator
        
        Args:
            config: dict containing model configuration
                - Should include grid_size under config['model']
        """
        self.grid_size = config['model']['grid_size']
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
        sdf_batch = []
        params_batch = []
        
        # Create coordinate grids
        x = torch.linspace(self.grid_min, self.grid_max, self.grid_size[0])
        y = torch.linspace(self.grid_min, self.grid_max, self.grid_size[1])
        z = torch.linspace(self.grid_min, self.grid_max, self.grid_size[2])
        
        # Create meshgrid
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        
        # Create points tensor of shape (depth, height, width, 3)
        points = torch.stack([xx, yy, zz], dim=-1)
        
        for _ in range(batch_size):
            # Generate random center and radius
            # Keep spheres away from boundaries
            center = np.random.uniform(-1.5, 1.5, 3)  # Reduced range for better visibility
            radius = np.random.uniform(0.3, 0.7)  # More consistent sphere sizes
            
            # Convert to torch tensors
            sphere_center = torch.tensor(center, dtype=torch.float32)
            
            # Calculate SDF values for sphere
            # Broadcast sphere_center to match points shape
            center_broadcast = sphere_center.view(1, 1, 1, 3)
            
            # Calculate distances from each point to sphere center
            distances = torch.norm(points - center_broadcast, dim=-1)
            
            # Calculate SDF values (negative inside sphere, positive outside)
            sdf_values = distances - radius
            
            # No normalization to preserve true SDF values
            sdf_batch.append(sdf_values)
            params_batch.append(torch.tensor([*center, radius], dtype=torch.float32))
        
        # Stack batches
        sdf_batch = torch.stack(sdf_batch)
        params_batch = torch.stack(params_batch)
        
        return sdf_batch.unsqueeze(1), params_batch

    def generate_test_sphere(self):
        """Generate a single centered test sphere for visualization verification."""
        # Create coordinate grids
        x = torch.linspace(self.grid_min, self.grid_max, self.grid_size[0])
        y = torch.linspace(self.grid_min, self.grid_max, self.grid_size[1])
        z = torch.linspace(self.grid_min, self.grid_max, self.grid_size[2])
        
        # Create meshgrid
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([xx, yy, zz], dim=-1)
        
        # Create centered sphere
        center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        radius = 1.0
        
        # Calculate SDF values
        center_broadcast = center.view(1, 1, 1, 3)
        distances = torch.norm(points - center_broadcast, dim=-1)
        sdf_values = distances - radius
        
        return sdf_values.unsqueeze(0).unsqueeze(0), torch.tensor([[0.0, 0.0, 0.0, radius]])
