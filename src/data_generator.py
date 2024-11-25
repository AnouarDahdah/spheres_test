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
        
        # Precompute grid points
        x = torch.linspace(self.grid_min, self.grid_max, self.grid_size[0])
        y = torch.linspace(self.grid_min, self.grid_max, self.grid_size[1])
        z = torch.linspace(self.grid_min, self.grid_max, self.grid_size[2])
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        self.grid_points = torch.stack([xx, yy, zz], dim=-1)  # Shape: (D, H, W, 3)

    def generate_sphere_sdf_batch(self, batch_size, normalize=False):
        """Generate a batch of sphere SDFs and corresponding parameters.
        
        Args:
            batch_size (int): Number of samples to generate
            normalize (bool): If True, normalize SDF values to have zero mean and unit variance
            
        Returns:
            tuple: (sdf_batch, params_batch)
                - sdf_batch: torch.Tensor of shape [batch, 1, depth, height, width]
                - params_batch: torch.Tensor of shape [batch, 4] containing [center_x, center_y, center_z, radius]
        """
        sdf_batch = []
        params_batch = []
        
        for _ in range(batch_size):
            # Generate random center within safe boundaries
            center = np.random.uniform(self.grid_min + 1, self.grid_max - 1, 3)
            
            # Determine maximum possible radius to ensure sphere fits within the grid
            max_radius = min(self.grid_max - max(center), min(center) - self.grid_min)
            radius = np.random.uniform(0.3, max_radius)
            
            # Convert to torch tensors
            sphere_center = torch.tensor(center, dtype=torch.float32)
            radius = torch.tensor(radius, dtype=torch.float32)
            
            # Calculate SDF
            center_broadcast = sphere_center.view(1, 1, 1, 3)  # Match grid_points shape
            distances = torch.norm(self.grid_points - center_broadcast, dim=-1)
            sdf_values = distances - radius
            
            # Normalize SDF values if requested
            if normalize:
                sdf_values = (sdf_values - sdf_values.mean()) / sdf_values.std()
            
            sdf_batch.append(sdf_values)
            params_batch.append(torch.tensor([*center, radius.item()], dtype=torch.float32))
        
        # Stack batches
        sdf_batch = torch.stack(sdf_batch)  # Shape: (batch, depth, height, width)
        params_batch = torch.stack(params_batch)  # Shape: (batch, 4)
        
        return sdf_batch.unsqueeze(1), params_batch  # Add channel dimension to SDF batch

    def generate_test_sphere(self, center=[0.0, 0.0, 0.0], radius=1.0):
        """Generate a single test sphere for visualization or verification.
        
        Args:
            center (list): Sphere center [x, y, z]
            radius (float): Sphere radius
        
        Returns:
            tuple: (sdf, params)
        """
        center = torch.tensor(center, dtype=torch.float32)
        radius = torch.tensor(radius, dtype=torch.float32)
        
        # Calculate SDF
        center_broadcast = center.view(1, 1, 1, 3)  # Match grid_points shape
        distances = torch.norm(self.grid_points - center_broadcast, dim=-1)
        sdf_values = distances - radius
        
        return sdf_values.unsqueeze(0).unsqueeze(0), torch.tensor([[*center, radius]])

    def generate_train_test_split(self, batch_size, test_size=0.2, normalize=False):
        """Generate train and test batches for SDFs and parameters.
        
        Args:
            batch_size (int): Number of samples to generate for each batch
            test_size (float): Fraction of data to use for testing (default 20%)
            normalize (bool): Whether to normalize the SDF values
            
        Returns:
            tuple: (train_sdf_batch, test_sdf_batch, train_params_batch, test_params_batch)
        """
        # Generate full data batch
        full_sdf_batch, full_params_batch = self.generate_sphere_sdf_batch(batch_size, normalize)
        
        # Split the data into train and test sets
        split_idx = int((1 - test_size) * batch_size)
        train_sdf_batch = full_sdf_batch[:split_idx]
        test_sdf_batch = full_sdf_batch[split_idx:]
        train_params_batch = full_params_batch[:split_idx]
        test_params_batch = full_params_batch[split_idx:]
        
        return train_sdf_batch, test_sdf_batch, train_params_batch, test_params_batch

