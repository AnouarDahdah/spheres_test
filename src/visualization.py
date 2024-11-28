import torch
import matplotlib.pyplot as plt
from skimage import measure

def visualize_sdf(sdf, ax, title):
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.cpu().squeeze().numpy()

    print(f"SDF range: min={sdf.min():.4f}, max={sdf.max():.4f}")

    if sdf.min() >= 0 or sdf.max() <= 0:
        print("Warning: SDF values do not cross zero. Adjusting values...")
        sdf = sdf - sdf.mean()

    try:
        verts, faces, _, _ = measure.marching_cubes(
            volume=sdf,
            level=0,
            allow_degenerate=False,
            method='lewiner'
        )

        verts = verts / sdf.shape[0] * 2 - 1

        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            color='skyblue',
            alpha=0.8,
            edgecolor='white',
            linewidth=0.1
        )

        ax.view_init(elev=30, azim=45)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect((1,1,1))
        ax.grid(True)

    except ValueError as e:
        print(f"Visualization error: {str(e)}")
