from setuptools import setup, find_packages

setup(
    name="sdf_generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'scikit-image>=0.19.0',
        'tqdm>=4.65.0',
        'pytest>=7.0.0',
        'pyyaml>=6.0.0',
    ],
    author="Anouar Dahdah",
    author_email="adahdah@sissa.it",
    description="A package for generating and learning 3D shapes using SDFs",
    keywords="deep learning, 3D, SDF, neural networks",
    python_requires=">=3.8",
)
