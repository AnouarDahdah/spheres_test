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
git clone https://github.com/AnouarDahdah/spheres_test.git
cd spheres_test

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

```

## Usage
```bash
# Train the model
python main.py
```
To run the tests:

1-Export the project root directory to the PYTHONPATH:
```bash
export PYTHONPATH=$(pwd)
```
2-Run the tests with pytest:
```bash
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
