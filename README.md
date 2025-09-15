![Banner](assets/github_banner.png)

# Finetuning Toolkit

A finetuning toolkit for working with protein models like ESM2. Uses JAX/Flax for multi-label classification of protein functions. Currently uses the dataset + function inspiration from the book ![Deep Learning for Biology](https://deep-learning-for-biology.com/).

## Installation

1. Install dependencies:
```bash
uv sync
```

2. Install datasets:
```bash
cd example/ && mkdir -p data
uv run dlfb-provision --chapter proteins --destination ./data && cd ..
```

## GPU Setup

This project requires CUDA for optimal performance. Follow these steps to set up GPU support:

### Install CUDA Dependencies

The project uses JAX with CUDA 12 support. Install the required packages:

```bash
# Install JAX with CUDA 12 support
uv add "jax[cuda12_pip]" --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install CUDA plugins
uv add jax-cuda12-plugin --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Running Example Training 

### Training (GPU recommended)

```bash
# Run with GPU support (recommended)
XLA_FLAGS=--xla_gpu_cuda_data_dir=/tmp/cuda_data uv run python example/main.py

# Or use the command-line interface
XLA_FLAGS=--xla_gpu_cuda_data_dir=/tmp/cuda_data uv run finetune

# Run with CPU fallback
JAX_PLATFORMS=cpu uv run python example/main.py
```

### Model Size
In our specific example, we use `facebook/esm2_t30_150M_UR50D`, but it consumes about ~10GB GPU memory and ~4GB RAM. There are other model sizes if you have larger or smaller GPUs.


### Set up libdevice (if needed)

If you encounter libdevice errors, create the required directory structure:

```bash
# Find your libdevice.10.bc file
find /usr -name "libdevice.10.bc" 2>/dev/null

# Create symlink (adjust path as needed)
mkdir -p /tmp/cuda_data/nvvm
ln -sf /usr/lib/nvidia-cuda-toolkit/libdevice /tmp/cuda_data/nvvm/libdevice
```