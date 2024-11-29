import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.network import CNNHybridNetwork
from src.data_generator import SDFDataset

def train_model(config):
    print("Training on CPU...")
    train_loader = DataLoader(
        SDFDataset(config['dataset_size'], config['grid_size'], train=True),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )

    model = CNNHybridNetwork(config['grid_size'], config['latent_dim'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0

        for params, sdfs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            sdfs = sdfs.view(-1, 1, config['grid_size'], config['grid_size'], config['grid_size'])
            
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
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")

    return model

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = train_model(config)
    os.makedirs('output', exist_ok=True)
    torch.save(model.state_dict(), 'output/sphere_model_cnn.pth')
