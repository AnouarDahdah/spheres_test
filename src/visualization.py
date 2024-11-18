```python
import torch
import plotly.graph_objects as go
from skimage import measure
import plotly.io as pio

class SDFVisualizer:
    def __init__(self, grid_res=20):
        self.grid_res = grid_res
        pio.renderers.default = "notebook"

    def save_sdf_as_3d_isosurface(self, sdf, filename, isovalue=0.0):
        """
        Save SDF as 3D isosurface with improved visualization
        """
        verts, faces, _, _ = measure.marching_cubes(sdf, level=isovalue)
        verts = verts / self.grid_res * 3 - 1.5

        fig = go.Figure(data=[go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.8,
            color='cyan',
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=0.2,
                roughness=0.5
            )
        )])

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-2, 2], showbackground=True, 
                           gridcolor="rgb(200, 200, 200)",
                           zeroline=True, zerolinecolor="rgb(128, 128, 128)"),
                yaxis=dict(range=[-2, 2], showbackground=True, 
                           gridcolor="rgb(200, 200, 200)",
                           zeroline=True, zerolinecolor="rgb(128, 128, 128)"),
                zaxis=dict(range=[-2, 2], showbackground=True, 
                           gridcolor="rgb(200, 200, 200)",
                           zeroline=True, zerolinecolor="rgb(128, 128, 128)"),
                aspectmode='cube',
                camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.8, y=1.8, z=1.8))
            ),
            title=dict(text="3D SDF Isosurface - Centered Sphere", y=0.95),
            width=1000,
            height=1000,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False
        )

        fig.write_html(filename)
```
