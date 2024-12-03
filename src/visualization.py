import numpy as np
from skimage import measure
import torch

def visualize_sdf(sdf, ax, title):
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.detach().cpu().numpy()
    
    # Handle different input shapes
    if sdf.ndim == 4:  # (1, 1, H, W)
        sdf = sdf.squeeze()
    elif sdf.ndim == 1:  # Flattened
        sdf = sdf.reshape(32, 32, 32)
    
    # Calculate data range for proper level setting
    data_min, data_max = np.min(sdf), np.max(sdf)
    level = 0 if data_min <= 0 <= data_max else (data_min + data_max) / 2
    
    # Generate mesh
    vertices, faces, _, _ = measure.marching_cubes(sdf, level=level)
    
    # Scale vertices to [-1, 1] range
    vertices = (vertices / sdf.shape[0]) * 2 - 1
    
    # Plot
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color='skyblue',
        alpha=0.8
    )
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=20, azim=45)