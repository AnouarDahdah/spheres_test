import os
import torch
import matplotlib.pyplot as plt
import yaml
from src.network import HybridNetwork
from src.data_generator import SDFGenerator
from src.visualization import visualize_sdf
from scripts.train import train_model

def generate_sphere(model, center, radius):
    model.eval()
    with torch.no_grad():
        params = torch.tensor([*center, radius], dtype=torch.float32)
        sdf = model.forward_params(params.unsqueeze(0))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        visualize_sdf(sdf[0], ax, f"Generated Sphere (r={radius:.2f})")
        plt.show()
        return sdf

def test_basic_sphere():
    print("\nTesting basic sphere generation...")
    generator = SDFGenerator(32)
    sdf = generator.generate_sphere_sdf([0, 0, 0], 0.3)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    visualize_sdf(sdf, ax, "Test Sphere")
    plt.show()

def main():
    try:
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        os.makedirs('output', exist_ok=True)
        test_basic_sphere()

        print("Starting pipeline...")
        model = train_model(config)
        print("Training completed")

        torch.save(model.state_dict(), 'output/sphere_model.pth')
        print("Model saved to output/sphere_model.pth")

        print("\nGenerating example spheres...")
        centers = [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [-0.2, -0.2, -0.2]]
        radii = [0.3, 0.25, 0.2]

        for center, radius in zip(centers, radii):
            generate_sphere(model, center, radius)

    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
