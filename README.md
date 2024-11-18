# spheres_test
 The pipeline maps sphere parameters (center coordinates and radius) to their 3D SDF representation by predicting a latent space encoding using the LatentPredictor and reconstructing the SDF through the autoencoder’s decoder. This allows efficient generation of SDFs directly from shape parameters.
```markdown

## Features
- 3D SDF Generation
- Autoencoder for SDF Compression
- Latent Space Prediction
- Visualization of Reconstructed Spheres

## Installation
```bash
git clone https://github.com/yourusername/sdf-autoencoder.git
cd sdf-autoencoder
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
# Train the model
python main.py

# Run tests
pytest tests/
```

## Configuration
Edit `config/config.yaml` to modify:
- Model architecture
- Training parameters
- Data generation settings

## Dependencies
- PyTorch
- NumPy
- Plotly
- scikit-image
- PyYAML


```
