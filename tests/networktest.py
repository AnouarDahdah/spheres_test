import unittest
import torch
from src.network import CNNHybridNetwork

class TestCNNHybridNetwork(unittest.TestCase):
    def setUp(self):
        self.grid_size = 32
        self.latent_dim = 64
        self.model = CNNHybridNetwork(self.grid_size, self.latent_dim)
        self.batch_size = 2

    def test_forward_autoencoder(self):
        sdf = torch.randn(self.batch_size, 1, self.grid_size, self.grid_size, self.grid_size)
        output, latent = self.model.forward_autoencoder(sdf)
        
        self.assertEqual(output.shape, sdf.shape)
        self.assertEqual(latent.shape, (self.batch_size, self.latent_dim))

    def test_forward_params(self):
        params = torch.randn(self.batch_size, 4)
        output = self.model.forward_params(params)
        
        self.assertEqual(output.shape, (self.batch_size, 1, self.grid_size, self.grid_size, self.grid_size))

if __name__ == '__main__':
    unittest.main()
