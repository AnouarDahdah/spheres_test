import torch
import numpy as np

class SDFGenerator:
    def __init__(self, config):
        self.grid_size = config["grid_size"]

    def generate_sphere_sdf_batch(self, batch_size):
        """Generate a batch of sphere SDFs and corresponding parameters."""
        sdf_batch = torch.zeros((batch_size, 1, *self.grid_size))  # [batch, channels, depth, height, width]
        params_batch = torch.zeros((batch_size, 4))  # [batch, center_x, center_y, center_z, radius]
        
        for i in range(batch_size):
            center = np.random.uniform(-0.5, 0.5, size=(3,))
            radius = np.random.uniform(0.1, 0.4)
            params_batch[i] = torch.tensor([*center, radius])
            
            # Generate SDF for the sphere
            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    for z in range(self.grid_size[2]):
                        point = (x / self.grid_size[0] - 0.5, y / self.grid_size[1] - 0.5, z / self.grid_size[2] - 0.5)
                        sdf_batch[i, 0, x, y, z] = np.linalg.norm(point - center) - radius
        
        return sdf_batch, params_batch
