
import torch
import torch.nn as nn
import torch.nn.functional as F

class SDF_Autoencoder(nn.Module): 
    def __init__(self, grid_res=20, latent_dim=32):
        super(SDF_Autoencoder, self).__init__()
        self.grid_res = grid_res

        # Encoder layers
        self.enc_conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)

        # Manually calculate flatten_dim based on grid_res
        self.flatten_dim = self._get_flatten_dim(grid_res)
        
        # Latent space
        self.fc_encoder = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder layers
        self.dec_conv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec_conv3 = nn.ConvTranspose3d(16, 1, kernel_size=2, stride=2)

    def _get_flatten_dim(self, grid_res):
        # Manually calculate the output size after convolutions and poolings
        dummy_input = torch.zeros(1, 1, grid_res, grid_res, grid_res)
        
        # Pass through the layers
        x = F.relu(self.enc_conv1(dummy_input))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv3(x))
        x = self.pool(x)
        
        # Return the number of elements in the output tensor
        return x.numel()

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc_encoder(x)     # Pass through fully connected encoder
        return x

    def decode(self, z):
        x = F.relu(self.fc_decoder(z))
        x = x.view(-1, 64, self.s3, self.s3, self.s3)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = self.dec_conv3(x)
        if x.size()[-1] != self.grid_res:
            x = F.interpolate(x, size=(self.grid_res, self.grid_res, self.grid_res), 
                            mode='trilinear', align_corners=True)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class LatentPredictor(nn.Module):
    def __init__(self, input_dim=4, latent_dim=32):
        super(LatentPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
