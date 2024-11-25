import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

class SDFVisualizer:
    def __init__(self, grid_res=32):
        self.grid_res = grid_res

    def save_sdf_as_html(self, reconstructed_sdf, original_sdf, filename="/scratch/adahdah/spheres_test/sdf_comparison.html", title="SDF Comparison"):
        """
        Save both the original and reconstructed SDFs as 3D isosurfaces in an HTML file.
        Also calculates and prints the difference error between the original and reconstructed SDFs.
        """
        assert reconstructed_sdf.ndim == 3, f"Expected 3D array, got shape {reconstructed_sdf.shape}"
        assert original_sdf.ndim == 3, f"Expected 3D array, got shape {original_sdf.shape}"

        # Calculate the difference error between original and reconstructed SDFs
        error = self.calculate_error(original_sdf, reconstructed_sdf)
        print(f"Mean Squared Error (MSE) between original and reconstructed SDF: {error:.4f}")

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
        """
        Create a 3D isosurface from the SDF data using Plotly.
        This will show the level set where the SDF value is close to zero.
        """
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

    def calculate_error(self, original_sdf, reconstructed_sdf):
        """
        Calculate the Mean Squared Error (MSE) between the original and reconstructed SDF.
        
        Args:
            original_sdf (np.array): The original SDF array.
            reconstructed_sdf (np.array): The reconstructed SDF array.
        
        Returns:
            float: The Mean Squared Error between the two SDFs.
        """
        # Flatten both SDFs and calculate MSE
        original_flat = original_sdf.flatten()
        reconstructed_flat = reconstructed_sdf.flatten()
        
        # MSE calculation
        mse = np.mean((original_flat - reconstructed_flat) ** 2)
        return mse

