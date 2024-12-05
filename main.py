import os
import torch
import matplotlib.pyplot as plt
import yaml
import numpy as np
from src.network import HybridNetwork
from src.data_generator import SDFGenerator
from src.visualization import visualize_sdf
from scripts.train import train_model
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

def generate_sphere(model, center, radius, save_dir=None, filename=None):
    """Generate a sphere and optionally save the visualization"""
    model.eval()
    with torch.no_grad():
        params = torch.tensor([*center, radius], dtype=torch.float32)
        sdf = model.forward_params(params.unsqueeze(0))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        visualize_sdf(sdf[0], ax, f"Generated Sphere (r={radius:.2f})")
        
        if save_dir and filename:
            save_path = Path(save_dir) / filename
            plt.savefig(save_path)
            print(f"Saved figure to {save_path}")
            
        plt.show()
        plt.close()
        return sdf

def test_basic_sphere(save_dir=None):
    """Test basic sphere generation and save the visualization"""
    print("\nTesting basic sphere generation...")
    generator = SDFGenerator(32)
    sdf = generator.generate_sphere_sdf([0, 0, 0], 0.3)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    visualize_sdf(sdf, ax, "Test Sphere")
    
    if save_dir:
        save_path = Path(save_dir) / "test_sphere.png"
        plt.savefig(save_path)
        print(f"Saved test sphere to {save_path}")
        
    plt.show()
    plt.close()
    return sdf

def plot_training_losses(metrics, save_dir):
    """Plot detailed training losses"""
    plt.figure(figsize=(15, 10))
    
    # Plot total loss
    plt.subplot(2, 2, 1)
    plt.plot(metrics['total_loss'], 'b-', label='Total Loss')
    plt.title('Total Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot AE vs Parameters loss
    plt.subplot(2, 2, 2)
    plt.plot(metrics['ae_loss'], 'r-', label='Autoencoder Loss')
    plt.plot(metrics['params_loss'], 'g-', label='Parameters Loss')
    plt.title('AE vs Parameters Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot latent space loss
    plt.subplot(2, 2, 3)
    plt.plot(metrics['latent_loss'], 'm-', label='Latent Space Loss')
    plt.title('Latent Space Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot loss ratios
    plt.subplot(2, 2, 4)
    ae_to_params = np.array(metrics['ae_loss']) / np.array(metrics['params_loss'])
    plt.plot(ae_to_params, 'c-', label='AE/Parameters Ratio')
    plt.title('Loss Ratios')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / 'detailed_training_losses.png')
    plt.close()

def compare_spheres(original_sdf, generated_sdf, save_dir):
    """Compare original and generated spheres"""
    if isinstance(original_sdf, torch.Tensor):
        original_sdf = original_sdf.detach().cpu().numpy()
    if isinstance(generated_sdf, torch.Tensor):
        generated_sdf = generated_sdf.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(20, 5))

    # Original sphere
    plt.subplot(141)
    plt.imshow(original_sdf[:, :, original_sdf.shape[2]//2])
    plt.title('Original Sphere (Middle Slice)')
    plt.colorbar()

    # Generated sphere
    plt.subplot(142)
    plt.imshow(generated_sdf[:, :, generated_sdf.shape[2]//2])
    plt.title('Generated Sphere (Middle Slice)')
    plt.colorbar()

    # Difference
    diff = original_sdf - generated_sdf
    plt.subplot(143)
    plt.imshow(diff[:, :, diff.shape[2]//2])
    plt.title('Difference (Middle Slice)')
    plt.colorbar()

    # Histogram of differences
    plt.subplot(144)
    plt.hist(diff.flatten(), bins=50)
    plt.title('Difference Distribution')
    plt.xlabel('Difference')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(save_dir / 'sphere_comparison.png')
    plt.close()

    # Print statistics
    print("\nSphere Comparison Statistics:")
    print(f"Mean absolute difference: {np.abs(diff).mean():.6f}")
    print(f"Max absolute difference: {np.abs(diff).max():.6f}")
    print(f"Standard deviation of difference: {np.std(diff):.6f}")

def main():
    try:
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # Create output directories
        output_dir = Path("output")
        images_dir = output_dir / "images"
        analysis_dir = output_dir / "analysis"
        for dir_path in [output_dir, images_dir, analysis_dir]:
            dir_path.mkdir(exist_ok=True)

        # Test basic sphere generation and store the original SDF
        original_sdf = test_basic_sphere(save_dir=images_dir)

        print("Starting pipeline...")
        model, metrics = train_model(config)
        print("Training completed")

        # Save model and metrics
        torch.save(model.state_dict(), output_dir / 'sphere_model.pth')
        torch.save(metrics, output_dir / 'training_metrics.pth')
        print("Model and metrics saved to output/")

        # Plot detailed training losses
        plot_training_losses(metrics, analysis_dir)

        # Generate and compare spheres
        print("\nGenerating example spheres for comparison...")
        centers = [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [-0.2, -0.2, -0.2]]
        radii = [0.3, 0.25, 0.2]

        for i, (center, radius) in enumerate(zip(centers, radii)):
            filename = f"generated_sphere_{i+1}.png"
            generated_sdf = generate_sphere(model, center, radius, save_dir=images_dir, filename=filename)
            
            # Compare with original sphere for the first case (centered sphere)
            if i == 0:
                compare_spheres(original_sdf, generated_sdf, analysis_dir)

    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
