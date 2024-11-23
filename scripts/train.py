import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure  # For generating isosurfaces
from src.data_generator import SDFGenerator
from src.autoencoder import SDF_Autoencoder
from src.latent_predictor import LatentPredictor

def load_config(config_path='config/config.yaml'):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_autoencoder(model, train_loader, config, device="cuda"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(config['training']['autoencoder_epochs']):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config['training']['autoencoder_epochs']}], Loss: {avg_loss:.6f}")
    return model

def train_latent_predictor(predictor, params_loader, autoencoder, config, device="cuda"):
    predictor = predictor.to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()

    autoencoder.eval()
    predictor.train()

    for epoch in range(config['training']['predictor_epochs']):
        total_loss = 0
        for batch_idx, (sdf, params) in enumerate(params_loader):
            params = params.to(device)
            sdf = sdf.to(device)
            
            with torch.no_grad():
                latent_true = autoencoder.encode(sdf)
                
            latent_pred = predictor(params)
            loss = criterion(latent_pred, latent_true)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(params_loader)
        print(f"Epoch [{epoch + 1}/{config['training']['predictor_epochs']}], Loss: {avg_loss:.6f}")
    
    return predictor

def plot_isosurface(original_sdf, reconstructed_sdf, iso_value=0.0):
    """
    Generate isosurface plots for the original and reconstructed SDFs.

    Args:
        original_sdf (numpy.ndarray): Original SDF values (3D array).
        reconstructed_sdf (numpy.ndarray): Reconstructed SDF values (3D array).
        iso_value (float): Isosurface value to extract (default is 0.0 for the surface).
    """
    fig = plt.figure(figsize=(16, 8))
    
    # Plot original SDF isosurface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Original SDF Isosurface", fontsize=14)

    # Generate isosurface
    verts, faces, _, _ = measure.marching_cubes(original_sdf, level=iso_value)
    mesh = ax1.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        cmap="viridis", lw=1, alpha=0.7
    )
    fig.colorbar(mesh, ax=ax1, pad=0.1)
    
    # Plot reconstructed SDF isosurface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Reconstructed SDF Isosurface", fontsize=14)

    # Generate isosurface
    verts, faces, _, _ = measure.marching_cubes(reconstructed_sdf, level=iso_value)
    mesh = ax2.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        cmap="viridis", lw=1, alpha=0.7
    )
    fig.colorbar(mesh, ax=ax2, pad=0.1)
    
    plt.tight_layout()
    plt.show()

def main():
    # Load configuration
    config = load_config()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate data
    data_generator = SDFGenerator(config)
    sdf_data, params_data = data_generator.generate_sphere_sdf_batch(
        batch_size=config['model']['num_samples']
    )
    
    # Create data loaders
    dataset = torch.utils.data.TensorDataset(sdf_data, params_data)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config['model']['batch_size'], 
        shuffle=True,
        drop_last=True
    )
    
    # Initialize models
    autoencoder = SDF_Autoencoder(
        grid_res=config['model']['grid_res'], 
        latent_dim=config['model']['latent_dim']
    )
    latent_predictor = LatentPredictor(
        input_dim=4, 
        latent_dim=config['model']['latent_dim']
    )
    
    # Train models
    autoencoder = train_autoencoder(autoencoder, dataloader, config, device)
    latent_predictor = train_latent_predictor(
        latent_predictor, dataloader, autoencoder, config, device
    )
    
    # Generate and visualize test results
    print("Testing and visualizing results...")
    test_sdf, test_params = data_generator.generate_sphere_sdf_batch(batch_size=1)
    
    with torch.no_grad():
        predicted_latent = latent_predictor(test_params[0].unsqueeze(0).to(device))
        reconstructed_sdf = autoencoder.decode(predicted_latent).cpu().numpy().squeeze()

    # Remove batch dimension by squeezing
    original_sdf = test_sdf[0].cpu().numpy().squeeze()  # Ensuring it's a 3D array
    reconstructed_sdf = reconstructed_sdf.squeeze()  # Ensuring it's a 3D array

    # Visualize isosurfaces
    plot_isosurface(original_sdf, reconstructed_sdf)

if __name__ == "__main__":
    main()
