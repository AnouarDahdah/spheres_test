import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
from dataclasses import dataclass
from src.network import CNNHybridNetwork
from src.data_generator import SDFGenerator, SDFDataset
from src.visualization import visualize_sdf
from tqdm import tqdm

@dataclass
class Config:
    grid_size: int
    dataset_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    latent_dim: int
    min_radius: float
    max_radius: float
    boundary_margin: float

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    train_loader = DataLoader(
        SDFDataset(config, split='train'),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    model = CNNHybridNetwork(config.grid_size, config.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0

        for params, sdfs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            params = params.to(device)
            sdfs = sdfs.to(device).float()
            
            ae_output, latent_ae = model.forward_autoencoder(sdfs)
            loss_ae = criterion(ae_output, sdfs)

            latent_params = model.param_to_latent(params)
            loss_latent = criterion(latent_params, latent_ae.detach())

            output_params = model.forward_params(params)
            loss_params = criterion(output_params, sdfs)

            loss = loss_ae + loss_latent + loss_params

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")

    return model

def generate_sphere(model, center, radius, device, save_path=None):
    model.eval()
    with torch.no_grad():
        params = torch.tensor([*center, radius], dtype=torch.float32).to(device)
        sdf = model.forward_params(params.unsqueeze(0))
        sdf = sdf.cpu()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        visualize_sdf(sdf[0], ax, f"Generated Sphere (r={radius:.2f})")
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved to {save_path}")
            plt.close()
        return sdf

def test_basic_sphere(save_path=None):
    generator = SDFGenerator(32)
    sdf = generator.generate_sphere_sdf([0, 0, 0], 0.3)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    visualize_sdf(sdf, ax, "Test Sphere")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
        plt.close()

def main():
    try:
        with open("config/config.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
            config = Config(**config_dict)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")
        os.makedirs('output/images', exist_ok=True)
        
        test_basic_sphere('output/images/test_sphere.png')

        print("Starting training...")
        model = train_model(config)
        print("Training completed")

        torch.save(model.state_dict(), 'output/sphere_model.pth')
        print("Model saved")

        centers = [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [-0.2, -0.2, -0.2]]
        radii = [0.3, 0.25, 0.2]

        for i, (center, radius) in enumerate(zip(centers, radii)):
            save_path = f'output/images/generated_sphere_{i}.png'
            generate_sphere(model, center, radius, device, save_path)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()