import torch
import torch.nn as nn

class LatentPredictor(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LatentPredictor, self).__init__()
        
        # Define the architecture of the latent predictor
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, latent_dim)
        
        # Replace BatchNorm with LayerNorm
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(64)
        
        self.activation_fn = nn.ReLU()

    def forward(self, x):
        x = self.activation_fn(self.layer_norm1(self.fc1(x)))  # Using LayerNorm here
        x = self.activation_fn(self.layer_norm2(self.fc2(x)))  # Using LayerNorm here
        x = self.fc3(x)  # Final output without normalization
        return x
