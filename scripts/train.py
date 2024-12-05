import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import sys
import os
from dataclasses import dataclass

# Fix for Colab path
current_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.network import HybridNetwork
from src.data_generator import SDFDataset

@dataclass
class DataConfig:
    grid_size: int
    dataset_size: int
    min_radius: float
    max_radius: float
    boundary_margin: float

def train_model(config):
    print("Training model...")

    # Create DataConfig object from config dict
    data_config = DataConfig(
        grid_size=config['training']['grid_size'],
        dataset_size=config['training']['dataset_size'],
        min_radius=config['generation']['min_radius'],
        max_radius=config['generation']['max_radius'],
        boundary_margin=config['generation']['boundary_margin']
    )

    train_loader = DataLoader(
        SDFDataset(data_config, split='train'),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )

    model = HybridNetwork(
        grid_size=config['training']['grid_size'],
        latent_dim=config['training']['latent_dim']
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    criterion = nn.MSELoss()

    # Initialize metrics dictionary
    metrics = {
        'total_loss': [],
        'ae_loss': [],
        'latent_loss': [],
        'params_loss': []
    }

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_losses = {
            'total': 0.0,
            'ae': 0.0,
            'latent': 0.0,
            'params': 0.0
        }

        for params, sdfs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
            # Train autoencoder
            ae_output, latent_ae = model.forward_autoencoder(sdfs)
            loss_ae = criterion(ae_output, sdfs)

            # Train param_to_latent mapping
            latent_params = model.param_to_latent(params)
            loss_latent = criterion(latent_params, latent_ae.detach())

            # Train direct params to SDF
            output_params = model.forward_params(params)
            loss_params = criterion(output_params, sdfs)

            loss = loss_ae + loss_latent + loss_params

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record batch losses
            epoch_losses['total'] += loss.item()
            epoch_losses['ae'] += loss_ae.item()
            epoch_losses['latent'] += loss_latent.item()
            epoch_losses['params'] += loss_params.item()

        # Calculate average losses for the epoch
        num_batches = len(train_loader)
        metrics['total_loss'].append(epoch_losses['total'] / num_batches)
        metrics['ae_loss'].append(epoch_losses['ae'] / num_batches)
        metrics['latent_loss'].append(epoch_losses['latent'] / num_batches)
        metrics['params_loss'].append(epoch_losses['params'] / num_batches)

        # Print epoch metrics with percentage
        progress = (epoch + 1) / config['training']['epochs'] * 100
        print(f"Epoch {epoch+1}/{config['training']['epochs']} ({progress:.1f}% complete)")
        print(f"Total Loss: {metrics['total_loss'][-1]:.4f}")
        print(f"AE Loss: {metrics['ae_loss'][-1]:.4f}")
        print(f"Latent Loss: {metrics['latent_loss'][-1]:.4f}")
        print(f"Params Loss: {metrics['params_loss'][-1]:.4f}")
        print("-" * 50)

    print("Training completed!")
    return model, metrics

if __name__ == "__main__":
    try:
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)

        # Train model and get metrics
        model, metrics = train_model(config)

        # Save model
        model_path = os.path.join('output', 'sphere_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Save final metrics
        final_metrics = {
            'metrics': metrics,
            'config': config
        }
        metrics_path = os.path.join('output', 'training_metrics.pth')
        torch.save(final_metrics, metrics_path)
        print(f"Training metrics saved to {metrics_path}")

    except Exception as e:
        print(f"Training error: {str(e)}")
        raise
