import torch
import numpy as np
import plotly.graph_objects as go
from skimage import measure
import plotly.io as pio
import os

class SDFVisualizer:
    def __init__(self, grid_res=None, config=None):
        """
        Initialize the SDF Visualizer
        
        Args:
            grid_res: int or tuple - Grid resolution (for backward compatibility)
            config: dict - Configuration dictionary containing model parameters
        """
        if config is not None:
            # If config is provided, use grid_size from config
            self.grid_size = config['model']['grid_size']
        elif grid_res is not None:
            # If grid_res is provided directly
            if isinstance(grid_res, int):
                self.grid_size = (grid_res, grid_res, grid_res)
            else:
                self.grid_size = grid_res
        else:
            # Default value
            self.grid_size = (32, 32, 32)

        self.grid_min = -3
        self.grid_max = 3
        pio.renderers.default = "notebook"

    def save_sdf_as_3d_isosurface(self, sdf, filename, isovalue=0.0):
        """
        Save SDF as 3D isosurface with improved visualization
        
        Args:
            sdf: torch.Tensor - The signed distance field [batch, channel, depth, height, width]
            filename: str - Output filename for the HTML visualization
            isovalue: float - The isosurface level value (default: 0.0)
        """
        # Convert to numpy if tensor
        if isinstance(sdf, torch.Tensor):
            sdf = sdf.detach().cpu().numpy()

        # Handle batch and channel dimensions
        if sdf.ndim > 3:
            sdf = sdf[0]  # Remove batch dimension if present
        if sdf.ndim > 3:
            sdf = sdf[0]  # Remove channel dimension if present

        # Ensure we have a 3D array
        assert sdf.ndim == 3, f"Expected 3D array, got shape {sdf.shape}"

        # Print value range for debugging
        print(f"SDF value range: [{sdf.min():.4f}, {sdf.max():.4f}]")

        try:
            # Generate the isosurface
            verts, faces, normals, values = measure.marching_cubes(
                sdf,
                level=isovalue,
                allow_degenerate=False,
                method='lewiner'
            )

            # Scale vertices to match the grid range used in generation
            verts = verts / np.array(self.grid_size) * (self.grid_max - self.grid_min) + self.grid_min

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

            # Update layout for better visualization
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[self.grid_min, self.grid_max],
                              showbackground=True,
                              gridcolor="rgb(200, 200, 200)",
                              zeroline=True,
                              zerolinecolor="rgb(128, 128, 128)"),
                    yaxis=dict(range=[self.grid_min, self.grid_max],
                              showbackground=True,
                              gridcolor="rgb(200, 200, 200)",
                              zeroline=True,
                              zerolinecolor="rgb(128, 128, 128)"),
                    zaxis=dict(range=[self.grid_min, self.grid_max],
                              showbackground=True,
                              gridcolor="rgb(200, 200, 200)",
                              zeroline=True,
                              zerolinecolor="rgb(128, 128, 128)"),
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

            # Save the visualization (the filename can include the full path)
            fig.write_html(filename)
            print(f"Successfully saved visualization to {filename}")

            return fig  # Return figure for optional display in notebook

        except ValueError as e:
            print(f"Error generating isosurface: {str(e)}")
            print("Try adjusting the isovalue or checking the SDF value range")

        except Exception as e:
            print(f"Unexpected error during visualization: {str(e)}")

