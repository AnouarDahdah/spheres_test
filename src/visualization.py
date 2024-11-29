import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

def visualize_sdf(sdf, ax, title):
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.cpu().squeeze().numpy()
    
    x, y, z = np.mgrid[-1:1:32j, -1:1:32j, -1:1:32j]
    pts = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    
    vertices, faces, _, _ = measure.marching_cubes(sdf.reshape(32, 32, 32))
    
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
