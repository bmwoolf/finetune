"""
Finetune: Fine-tune ESM2 to classify proteins

This package provides functionality for training protein function prediction models
using ESM2 embeddings and JAX/Flax.
"""

from .main import main, Model, get_mean_embeddings, store_sequence_embeddings

__version__ = "0.1.0"
__all__ = ["main", "Model", "get_mean_embeddings", "store_sequence_embeddings"]
