#!/bin/bash
# Kaggle setup script to install compatible packages

echo "Installing compatible packages for VietAI/vit5-base on Kaggle..."

# First, upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install main requirements with newer versions
pip install --upgrade \
  'transformers>=4.36.0' \
  'sentencepiece>=0.2.0' \
  'torch>=2.0.0' \
  'datasets>=2.14.0' \
  'protobuf>=3.20.0' \
  'py-fire>=0.0.3' \
  'spacy>=3.6.0' \
  'tqdm>=4.66.0' \
  'numpy>=1.24.0'

# Clear cache
pip cache purge

echo "Setup complete! Ready to use the repository."
