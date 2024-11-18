
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from src.data_generator import SDFGenerator
from src.autoencoder import SDF_Autoencoder
from src.latent_predictor import LatentPredictor
from src.visualization import SDFVisualizer

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_autoencoder(model, train_loader, config, device="cuda"):
    """
    Train the autoencoder model
    """
    print(f"Training on device: {device}")
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
    """
    Train the latent predictor model
    """
    print(f"Training Latent Predictor on device: {device}")
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
    
    # Generate and save test results
    print("Testing and saving results...")
    test_sdf, test_params = data_generator.generate_sphere_sdf_batch(batch_size=1)
    with torch.no_grad():
        predicted_latent = latent_predictor(test_params[0].unsqueeze(0).to(device))
        reconstructed_sdf = autoencoder.decode(predicted_latent).cpu().numpy().squeeze()

    # Visualize results
    visualizer = SDFVisualizer(grid_res=config['model']['grid_res'])
    visualizer.save_sdf_as_3d_isosurface(reconstructed_sdf, filename="final_sphere.html")
    print("Visualization saved as 'final_sphere.html'")

if __name__ == "__main__":
    main()

