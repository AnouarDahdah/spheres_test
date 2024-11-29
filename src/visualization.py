import torch
import matplotlib.pyplot as plt
from skimage import measure
import os

def visualize_sdf(sdf, ax, title, save_path=None):
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
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
    except ValueError as e:
        print(f"Visualization error: {str(e)}")

def generate_and_save_sphere(model, center, radius, save_dir='outputs/visualizations'):
    model.eval()
    with torch.no_grad():
        params = torch.tensor([*center, radius], dtype=torch.float32)
        sdf = model.forward_params(params.unsqueeze(0))
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        filename = f"sphere_c{center[0]}_{center[1]}_{center[2]}_r{radius}.png"
        save_path = os.path.join(save_dir, filename)
        
        visualize_sdf(sdf[0], ax, f"Generated Sphere (r={radius:.2f})", save_path)
        plt.close()
        
        return sdf
