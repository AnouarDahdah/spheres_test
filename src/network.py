import torch.nn as nn

class HybridNetwork(nn.Module):
    def __init__(self, grid_size: int = 32, latent_dim: int = 64):
        super().__init__()
        self.grid_size = grid_size
        input_size = grid_size ** 3

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.param_to_latent = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_size)
        )

    def forward_autoencoder(self, sdf):
        latent = self.encoder(sdf)
        output = self.decoder(latent)
        return output.view(-1, 1, self.grid_size, self.grid_size, self.grid_size), latent

    def forward_params(self, params):
        latent = self.param_to_latent(params)
        output = self.decoder(latent)
        return output.view(-1, 1, self.grid_size, self.grid_size, self.grid_size)
