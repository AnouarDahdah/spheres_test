import torch
import torch.nn as nn
import torch.nn.functional as F

class SDF_Autoencoder(nn.Module):
    def __init__(self, grid_res=30, latent_dim=64, activation_fn=nn.ReLU):
        """
        Autoencoder for Signed Distance Function (SDF) data.

        Args:
            grid_res (int): Resolution of the input 3D grid.
            latent_dim (int): Dimensionality of the latent space.
            activation_fn: Activation function class to use in layers (default: nn.ReLU).
        """
        super(SDF_Autoencoder, self).__init__()
        self.grid_res = grid_res
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn()

        # Encoder layers
        self.enc_conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm3d(16)
        self.enc_conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm3d(32)
        self.enc_conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.enc_bn3 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(2)

        # Calculate size after encodings
        self.encoded_size = grid_res // 8  # After 3 pooling layers

        # Fully connected layers
        self.fc_encoder = nn.Sequential(
            nn.Linear(64 * self.encoded_size**3, latent_dim),
            nn.Dropout(0.3)  # Regularization
        )
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * self.encoded_size**3),
            nn.Dropout(0.3)
        )

        # Decoder layers
        self.dec_conv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec_bn1 = nn.BatchNorm3d(32)
        self.dec_conv2 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec_bn2 = nn.BatchNorm3d(16)
        self.dec_conv3 = nn.ConvTranspose3d(16, 1, kernel_size=2, stride=2)

    def encode(self, x):
        """Encodes input to the latent representation."""
        x = self.activation_fn(self.enc_bn1(self.enc_conv1(x)))
        x = self.pool(x)
        x = self.activation_fn(self.enc_bn2(self.enc_conv2(x)))
        x = self.pool(x)
        x = self.activation_fn(self.enc_bn3(self.enc_conv3(x)))
        x = self.pool(x)

        # Flatten and encode to latent space
        x = x.view(x.size(0), -1)
        return self.fc_encoder(x)

    def decode(self, z):
        """Decodes latent representation to reconstruct the SDF."""
        # Decode from latent space
        x = self.fc_decoder(z)

        # Reshape to match decoder's expected input
        x = x.view(-1, 64, self.encoded_size, self.encoded_size, self.encoded_size)

        # Decoding path
        x = self.activation_fn(self.dec_bn1(self.dec_conv1(x)))
        x = self.activation_fn(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_conv3(x)

        return x

    def forward(self, x):
        """Forward pass through the autoencoder."""
        z = self.encode(x)
        return self.decode(z)
