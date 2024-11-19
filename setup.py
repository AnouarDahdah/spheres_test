
from setuptools import setup, find_packages

setup(
    name='sdf-autoencoder',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'plotly',
        'scikit-image',
        'pyyaml',
    ],
    author='Anouar Dahdah',
    description='SDF Autoencoder for Sphere Reconstruction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
)

