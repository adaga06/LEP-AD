#!/bin/bash

# Clone the ESM-2 repository
git clone https://github.com/facebookresearch/esm.git

# Navigate to the esm directory
cd esm

# Create a data directory
mkdir data

# Create the ESM environment
conda env create -f environment.yml
source activate esm2

# Assuming you have a way to automatically run the notebook, otherwise this step requires manual intervention
# Run the data_protein_esm.ipynb notebook

# Create the LEP-AD environment
conda env create -f environment_LEP_AD.yml
source activate LEP-AD

# Run the LEP-AD script
python LEP-AD.py --dataset Stitch --batch-size 128 --output_dim 64 --heads 4 --epochs 400 --n_layers 3 --step_size 250
