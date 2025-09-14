"""
Finetune: Fine-tune ESM2 to classify proteins

This package provides functionality for training protein function prediction models
using ESM2 embeddings and JAX/Flax.
"""

from .main import main
from .src.data import (
    get_go_term_descriptions,
    load_cafa3_data,
    create_train_valid_test_splits,
    generate_embeddings,
    store_sequence_embeddings,
    load_sequence_embeddings,
    get_mean_embeddings
)
from .src.model import Model

__version__ = "0.1.0"
__all__ = [
    "main", 
    "Model", 
    "get_go_term_descriptions",
    "load_cafa3_data",
    "create_train_valid_test_splits", 
    "generate_embeddings",
    "get_mean_embeddings", 
    "store_sequence_embeddings",
    "load_sequence_embeddings"
]
