#!/bin/bash

# Update package lists
apt-get update

# Install system dependencies
apt-get install -y libomp-dev

# Install PyTorch with CUDA (Google Colab has CUDA 11.8 by default)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and dependencies
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install additional dependencies
pip install networkx
pip install torch-geometric-temporal

# Install BioPython and GEMMI for .cif file parsing
pip install biopython gemmi

# Install scikit-learn and scipy for nearest neighbor computations
pip install scikit-learn scipy

# Verify installation
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), 'CUDA Version:', torch.version.cuda)"
python -c "import torch_geometric; print('Torch Geometric:', torch_geometric.__version__)"
