import numpy as np
import plotly.graph_objects as go
<<<<<<< HEAD
from skimage import measure
import plotly.io as pio
import os
=======
from plotly.subplots import make_subplots
>>>>>>> 0d1ca93e5ffc931f7379d01b92a83994adcd913e

class SDFVisualizer:
    def __init__(self, grid_res=32):
        self.grid_res = grid_res

    def save_sdf_as_html(self, reconstructed_sdf, original_sdf, filename="sdf_comparison.html", title="SDF Comparison"):
        """
        Save both the original and reconstructed SDFs as 3D isosurfaces in an HTML file.
        """
<<<<<<< HEAD
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
=======
        assert reconstructed_sdf.ndim == 3, f"Expected 3D array, got shape {reconstructed_sdf.shape}"
        assert original_sdf.ndim == 3, f"Expected 3D array, got shape {original_sdf.shape}"

        # Create the figure with two subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Original SDF", "Reconstructed SDF"],
            specs=[[{"type": "surface"}, {"type": "surface"}]]
        )

        # Create isosurfaces for both SDFs
        original_surface = self.create_isosurface(original_sdf, title="Original SDF")
        reconstructed_surface = self.create_isosurface(reconstructed_sdf, title="Reconstructed SDF")

        # Add isosurfaces to subplots
        fig.add_trace(original_surface, row=1, col=1)
        fig.add_trace(reconstructed_surface, row=1, col=2)

        # Update layout for better visualization
        fig.update_layout(
            title=title,
            height=600,
            width=1200,
            showlegend=False,
            scene=dict(aspectmode='cube'),
            scene2=dict(aspectmode='cube')
        )

        # Save the figure as an HTML file
        fig.write_html(filename)
        print(f"Visualization saved to {filename}")

    def create_isosurface(self, sdf_data, title="SDF"):
>>>>>>> 0d1ca93e5ffc931f7379d01b92a83994adcd913e
        """
        Create a 3D isosurface from the SDF data using Plotly.
        This will show the level set where the SDF value is close to zero.
        """
<<<<<<< HEAD
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

=======
        # Generate grid coordinates for the 3D space
        x = np.linspace(-1, 1, self.grid_res)
        y = np.linspace(-1, 1, self.grid_res)
        z = np.linspace(-1, 1, self.grid_res)

        # Create meshgrid for visualization (flattening it for isosurface plot)
        X, Y, Z = np.meshgrid(x, y, z)
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        # Flatten SDF data for use in isosurface plot
        sdf_values = sdf_data.flatten()

        # Create the isosurface plot using the SDF data
        isosurface = go.Isosurface(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            value=sdf_values,
            isomin=-0.01,  # Display SDF values close to zero
            isomax=0.01,
            surface_count=1,  # Show only one surface (SDF = 0)
            colorscale='Viridis',
            caps=dict(x_show=False, y_show=False, z_show=False),
            name=title
        )
        return isosurface
>>>>>>> 0d1ca93e5ffc931f7379d01b92a83994adcd913e
