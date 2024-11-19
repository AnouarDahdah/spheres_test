import torch
import numpy as np
import plotly.graph_objects as go
from skimage import measure
import plotly.io as pio

class SDFVisualizer:
    def __init__(self, grid_res=32):  # Updated to match your model's grid_res
        self.grid_res = grid_res
        pio.renderers.default = "notebook"
    
    def save_sdf_as_3d_isosurface(self, sdf, filename, isovalue=0.0):
        """
        Save SDF as 3D isosurface with improved visualization and error handling
        
        Args:
            sdf: torch.Tensor or numpy.ndarray - The signed distance field
            filename: str - Output filename for the HTML visualization
            isovalue: float - The isosurface level value (default: 0.0)
        """
        # Convert to numpy if tensor
        if isinstance(sdf, torch.Tensor):
            sdf = sdf.detach().cpu().numpy()
        
        # If the input is 4D or 5D (has batch/channel dimensions), take the first element
        if sdf.ndim > 3:
            sdf = sdf[0]
        if sdf.ndim > 3:
            sdf = sdf[0]
            
        # Ensure we have a 3D array
        assert sdf.ndim == 3, f"Expected 3D array, got shape {sdf.shape}"
        
        # Print value range for debugging
        print(f"SDF value range: [{sdf.min():.4f}, {sdf.max():.4f}]")
        
        # Normalize SDF values if they're too large/small
        if sdf.max() > 1.0 or sdf.min() < -1.0:
            sdf = np.clip(sdf, -1.0, 1.0)
            print("SDF values clipped to [-1, 1] range")
        
        try:
            # Try to generate the isosurface
            verts, faces, normals, values = measure.marching_cubes(
                sdf,
                level=isovalue,
                allow_degenerate=False,
                method='lewiner'
            )
            
            # Scale vertices to desired range
            verts = verts / self.grid_res * 3 - 1.5
            
            # Create the 3D visualization
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
                ),
                flatshading=True
            )])
            
            # Update layout with improved settings
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
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.8, y=1.8, z=1.8)
                    )
                ),
                title=dict(
                    text=f"3D SDF Isosurface (isovalue={isovalue:.3f})",
                    y=0.95
                ),
                width=1000,
                height=1000,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False
            )
            
            # Save the visualization
            fig.write_html(filename)
            print(f"Successfully saved visualization to {filename}")
            
        except ValueError as e:
            print(f"Error generating isosurface: {str(e)}")
            print("Try adjusting the isovalue or checking the SDF value range")
            
        except Exception as e:
            print(f"Unexpected error during visualization: {str(e)}")
