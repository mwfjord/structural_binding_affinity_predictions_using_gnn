#!/bin/bash


# Install PyTorch (CPU version)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install PyTorch Geometric (CPU version)
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

# Install PyTorch Geometric Dependencies
conda install -c conda-forge networkx -y
pip install torch-geometric-temporal

# Install BioPython and GEMMI for .cif file parsing
conda install -c conda-forge biopython gemmi -y

# Install torch-cluster dependencies for nearest neighbor graphs
conda install -c conda-forge scikit-learn scipy -y

# Verify Installation
python -c "import torch; import torch_geometric; print('Torch:', torch.__version__, 'Torch Geometric:', torch_geometric.__version__)"
