import torch
import torch.nn as nn
import torch.nn.functional as F

class SDF_Autoencoder(nn.Module):
    def __init__(self, grid_res=32, latent_dim=32):
        super(SDF_Autoencoder, self).__init__()
        self.grid_res = grid_res
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.enc_conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        
        # Calculate the size after encodings
        self.encoded_size = grid_res // 8  # After 3 pooling layers
        
        # Fully connected layers
        self.fc_encoder = nn.Linear(64 * self.encoded_size**3, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, 64 * self.encoded_size**3)
        
        # Decoder layers
        self.dec_conv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec_conv3 = nn.ConvTranspose3d(16, 1, kernel_size=2, stride=2)

    def encode(self, x):
        # First ensure input is the correct size
        if x.size()[-1] != self.grid_res:
            x = F.interpolate(x, size=(self.grid_res, self.grid_res, self.grid_res),
                            mode='trilinear', align_corners=True)
        
        # Encoding path
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv3(x))
        x = self.pool(x)
        
        # Flatten and encode to latent space
        x = x.view(x.size(0), -1)
        x = self.fc_encoder(x)
        return x

    def decode(self, z):
        # Decode from latent space
        x = F.relu(self.fc_decoder(z))
        
        # Reshape to match decoder's expected input
        x = x.view(-1, 64, self.encoded_size, self.encoded_size, self.encoded_size)
        
        # Decoding path
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = self.dec_conv3(x)
        
        # Ensure output size matches grid_res
        x = F.interpolate(x, size=(self.grid_res, self.grid_res, self.grid_res),
                         mode='trilinear', align_corners=True)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
