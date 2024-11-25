import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from src.data_generator import SDFGenerator
from src.autoencoder import SDF_Autoencoder
from src.latent_predictor import LatentPredictor
from src.visualization import SDFVisualizer

def load_config(config_path='config/config.yaml'):
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
        print(f"[Autoencoder Training] Epoch [{epoch + 1}/{config['training']['autoencoder_epochs']}], Loss: {avg_loss:.6f}")
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
        print(f"[Latent Predictor Training] Epoch [{epoch + 1}/{config['training']['predictor_epochs']}], Loss: {avg_loss:.6f}")
    
    return predictor

def calculate_final_error(original_sdf, reconstructed_sdf):
    error = ((original_sdf - reconstructed_sdf) ** 2).mean()
    print(f"Final Reconstruction Error (MSE): {error:.6f}")
    return error

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

    original_sdf = test_sdf[0].cpu().numpy().squeeze()  # Ensure it's a 3D array
    reconstructed_sdf = reconstructed_sdf.squeeze()    # Ensure it's a 3D array

    # Calculate final error
    calculate_final_error(original_sdf, reconstructed_sdf)

    # Save visualization
    visualizer = SDFVisualizer(grid_res=config['model']['grid_res'])
    visualizer.save_sdf_as_html(
        reconstructed_sdf=reconstructed_sdf,
        original_sdf=original_sdf,
        filename="/scratch/adahdah/spheres_test/sdf_comparison.html",
        title="SDF Comparison"
    )

if __name__ == "__main__":
    main()

