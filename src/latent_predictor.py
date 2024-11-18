
import torch
import torch.nn as nn
import torch.nn.functional as F

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

