import torch
import torch.nn as nn

class CNNHybridNetwork(nn.Module):
    def __init__(self, grid_size: int = 32, latent_dim: int = 64):
        super().__init__()
        self.grid_size = grid_size
        
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Flatten(),
            nn.Linear(64 * (grid_size // 8) ** 3, latent_dim)
        )
        
        self.param_to_latent = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        self.decoder_linear = nn.Linear(
            latent_dim, 
            64 * (grid_size // 8) ** 3
        )
        
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward_autoencoder(self, sdf):
        latent = self.encoder(sdf)
        return self.decode(latent), latent
    
    def forward_params(self, params):
        latent = self.param_to_latent(params)
        return self.decode(latent)
    
    def decode(self, latent):
        x = self.decoder_linear(latent)
        x = x.view(-1, 64, self.grid_size // 8, 
                   self.grid_size // 8, self.grid_size // 8)
        return self.decoder_cnn(x)
