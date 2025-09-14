![Banner](assets/github_banner.png)

# Fine Tuning ESM2

A protein function prediction system that fine-tunes ESM2 embeddings using JAX/Flax for multi-label classification of protein functions.


## Installation

1. Install dependencies:
```bash
uv sync
```

2. Install datasets:
```bash
cd example/ && mkdir -p data
uv run dlfb-provision --chapter proteins --destination ./data
```

## GPU Setup

This project requires CUDA for optimal performance. Follow these steps to set up GPU support:

### 1. Install CUDA Dependencies

The project uses JAX with CUDA 12 support. Install the required packages:

```bash
# Install JAX with CUDA 12 support
uv run python -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install CUDA plugins
uv run python -m pip install --upgrade jax-cuda12-plugin -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 2. Set up libdevice (if needed)

If you encounter libdevice errors, create the required directory structure:

```bash
# Find your libdevice.10.bc file
find /usr -name "libdevice.10.bc" 2>/dev/null

# Create symlink (adjust path as needed)
mkdir -p /tmp/cuda_data/nvvm
ln -sf /usr/lib/nvidia-cuda-toolkit/libdevice /tmp/cuda_data/nvvm/libdevice
```

## Usage

### Running the Training

```bash
# Run with GPU support (recommended)
XLA_FLAGS=--xla_gpu_cuda_data_dir=/tmp/cuda_data uv run python example/main.py

# Or use the command-line interface
XLA_FLAGS=--xla_gpu_cuda_data_dir=/tmp/cuda_data uv run finetune
```

### CPU-only Mode (if needed)

```bash
# Run with CPU fallback
JAX_PLATFORMS=cpu uv run python example/main.py
```