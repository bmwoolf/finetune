#!/usr/bin/env python3

import os
import math
import obonet
import numpy as np
import pandas as pd
import torch
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn import metrics
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from flax.training.train_state import TrainState

# Set up paths for local environment
BASE_DIR = Path(__file__).parent.parent  # Go up one level from finetune/ to project root
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# DLFB UTILITIES (Recreated from original package)
# =============================================================================

def assets(subdir: str = None) -> str:
    """Get path to assets directory."""
    assets_dir = str(DATA_DIR)
    if subdir:
        assets_dir = os.path.join(assets_dir, subdir)
    return assets_dir

def get_device() -> torch.device:
    """Get available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_short_dict(d, max_items=10, width=80):
    """Print dictionary with truncation."""
    import textwrap
    from itertools import islice
    
    shown = list(islice(d.items(), max_items))
    remaining = len(d) - len(shown)
    preview = {k: v for k, v in shown}
    s = str(preview)
    wrapped_lines = textwrap.wrap(s, width=width)
    for line in wrapped_lines:
        print(line)
    if remaining > 0:
        print(f"…(+{remaining} more entries)")

# =============================================================================
# PROTEIN DATA HANDLING
# =============================================================================

def get_go_term_descriptions(store_path: str) -> pd.DataFrame:
    """Return GO term to description mapping, downloading if needed."""
    if not os.path.exists(store_path):
        url = "https://current.geneontology.org/ontology/go-basic.obo"
        graph = obonet.read_obo(url)
        
        # Extract GO term IDs and names from the graph nodes.
        id_to_name = {id: data.get("name") for id, data in graph.nodes(data=True)}
        go_term_descriptions = pd.DataFrame(
            zip(id_to_name.keys(), id_to_name.values()),
            columns=["term", "description"],
        )
        go_term_descriptions.to_csv(store_path, index=False)
    else:
        go_term_descriptions = pd.read_csv(store_path)
    return go_term_descriptions

def get_mean_embeddings(
    sequences: List[str],
    tokenizer,
    model,
    device: torch.device = None,
) -> np.ndarray:
    """Compute mean embedding for each sequence using a protein LM."""
    if not device:
        device = get_device()

    # Tokenize input sequences and pad them to equal length.
    model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

    # Move tokenized inputs to the target device (CPU or GPU).
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    # Move model to the target device and set it to evaluation mode.
    model = model.to(device)
    model.eval()

    # Forward pass without gradient tracking to obtain embeddings.
    with torch.no_grad():
        outputs = model(**model_inputs)
        mean_embeddings = outputs.last_hidden_state.mean(dim=1)

    return mean_embeddings.detach().cpu().numpy()

def store_sequence_embeddings(
    sequence_df: pd.DataFrame,
    store_prefix: str,
    tokenizer,
    model,
    batch_size: int = 64,
    force: bool = False,
) -> None:
    """Extract and store mean embeddings for each protein sequence."""
    model_name = str(model.name_or_path).replace("/", "_")
    store_file = f"{store_prefix}_{model_name}.feather"

    if not os.path.exists(store_file) or force:
        device = get_device()

        # Iterate through protein dataframe in batches, extracting embeddings.
        n_batches = math.ceil(sequence_df.shape[0] / batch_size)
        batches: List[np.ndarray] = []
        for i in range(n_batches):
            batch_seqs = list(
                sequence_df["Sequence"][i * batch_size : (i + 1) * batch_size]
            )
            batches.extend(get_mean_embeddings(batch_seqs, tokenizer, model, device))

        # Store each of the embedding values in a separate column in the dataframe.
        embeddings = pd.DataFrame(np.vstack(batches))
        embeddings.columns = [f"ME:{int(i)+1}" for i in range(embeddings.shape[1])]
        df = pd.concat([sequence_df.reset_index(drop=True), embeddings], axis=1)
        df.to_feather(store_file)

def load_sequence_embeddings(
    store_file_prefix: str, model_checkpoint: str
) -> pd.DataFrame:
    """Load stored embedding DataFrame from disk."""
    model_name = model_checkpoint.replace("/", "_")
    store_file = f"{store_file_prefix}_{model_name}.feather"
    return pd.read_feather(store_file)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class Model(nn.Module):
    """Simple MLP for protein function prediction."""
    
    num_targets: int
    dim: int = 256

    @nn.compact
    def __call__(self, x):
        """Apply MLP layers to input features."""
        x = nn.Sequential([
            nn.Dense(self.dim * 2),
            jax.nn.gelu,
            nn.Dense(self.dim),
            jax.nn.gelu,
            nn.Dense(self.num_targets),
        ])(x)
        return x

    def create_train_state(self, rng: jax.Array, dummy_input, tx) -> TrainState:
        """Initialize model parameters and return a training state."""
        variables = self.init(rng, dummy_input)
        return TrainState.create(
            apply_fn=self.apply, params=variables["params"], tx=tx
        )

# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def convert_to_tfds(
    df: pd.DataFrame,
    embeddings_prefix: str = "ME:",
    target_prefix: str = "GO:",
    is_training: bool = False,
    shuffle_buffer: int = 50,
) -> tf.data.Dataset:
    """Convert embedding DataFrame into a TensorFlow dataset."""
    dataset = tf.data.Dataset.from_tensor_slices({
        "embedding": df.filter(regex=f"^{embeddings_prefix}").to_numpy(),
        "target": df.filter(regex=f"^{target_prefix}").to_numpy(),
    })
    if is_training:
        dataset = dataset.shuffle(shuffle_buffer).repeat()
    return dataset

def compute_metrics(
    targets: np.ndarray, probs: np.ndarray, thresh=0.5
) -> Dict[str, float]:
    """Compute accuracy, recall, precision, auPRC, and auROC."""
    if np.sum(targets) == 0:
        return {
            m: 0.0 for m in ["accuracy", "recall", "precision", "auprc", "auroc"]
        }
    return {
        "accuracy": metrics.accuracy_score(targets, probs >= thresh),
        "recall": metrics.recall_score(targets, probs >= thresh).item(),
        "precision": metrics.precision_score(
            targets,
            probs >= thresh,
            zero_division=0.0,
        ).item(),
        "auprc": metrics.average_precision_score(targets, probs).item(),
        "auroc": metrics.roc_auc_score(targets, probs).item(),
    }

@jax.jit
def train_step(state, batch):
    """Run a single training step and update model parameters."""
    
    def calculate_loss(params):
        """Compute sigmoid cross-entropy loss from logits."""
        logits = state.apply_fn({"params": params}, x=batch["embedding"])
        loss = optax.sigmoid_binary_cross_entropy(logits, batch["target"]).mean()
        return loss

    grad_fn = jax.value_and_grad(calculate_loss, has_aux=False)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def eval_step(state, batch) -> Dict[str, float]:
    """Run evaluation step and return mean metrics over targets."""
    logits = state.apply_fn({"params": state.params}, x=batch["embedding"])
    loss = optax.sigmoid_binary_cross_entropy(logits, batch["target"]).mean()
    
    # Calculate per-target metrics
    probs = jax.nn.sigmoid(logits)
    target_metrics = []
    for target, prob in zip(batch["target"], probs):
        target_metrics.append(compute_metrics(target, prob))
    
    metrics_dict = {
        "loss": loss.item(),
        **pd.DataFrame(target_metrics).mean(axis=0).to_dict(),
    }
    return metrics_dict

def train(
    state: TrainState,
    dataset_splits: Dict[str, tf.data.Dataset],
    batch_size: int,
    num_steps: int = 300,
    eval_every: int = 30,
):
    """Train model using batched TF datasets and track performance metrics."""
    # Create containers to handle calculated during training and evaluation.
    train_metrics, valid_metrics = [], []

    # Create batched dataset to pluck batches from for each step.
    train_batches = (
        dataset_splits["train"]
        .batch(batch_size, drop_remainder=True)
        .as_numpy_iterator()
    )

    steps = tqdm(range(num_steps))  # Steps with progress bar.
    for step in steps:
        steps.set_description(f"Step {step + 1}")

        # Get batch of training data, convert into a JAX array, and train.
        state, loss = train_step(state, next(train_batches))
        train_metrics.append({"step": step, "loss": loss.item()})

        if step % eval_every == 0:
            # For all the evaluation batches, calculate metrics.
            eval_metrics = []
            for eval_batch in (
                dataset_splits["valid"].batch(batch_size=batch_size).as_numpy_iterator()
            ):
                eval_metrics.append(eval_step(state, eval_batch))
            valid_metrics.append(
                {"step": step, **pd.DataFrame(eval_metrics).mean(axis=0).to_dict()}
            )

    return state, {"train": train_metrics, "valid": valid_metrics}

# =============================================================================
# MAIN EXECUTION (Following Chapter 2 Flow)
# =============================================================================

def main():
    print("=== Chapter 2: Learning the Language of Proteins ===")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    jax.random.PRNGKey(42)
    
    # =============================================================================
    # 2.4. Preparing the Data
    # =============================================================================
    print("\n2.4. Preparing the Data...")
    
    # 2.4.1. Loading the CAFA3 Data
    print("2.4.1. Loading the CAFA3 Data...")
    
    # Note: In a real implementation, you would download these files
    # For now, we'll create dummy data to demonstrate the structure
    print("Creating dummy CAFA3-like data...")
    
    # Create dummy labels dataframe
    n_proteins = 1000
    n_go_terms = 50
    
    dummy_labels = []
    for protein_id in range(n_proteins):
        # Each protein gets 2-5 random GO terms
        n_terms = np.random.randint(2, 6)
        terms = np.random.choice(n_go_terms, n_terms, replace=False)
        for term in terms:
            dummy_labels.append({
                'EntryID': f'PROTEIN_{protein_id:06d}',
                'term': f'GO:{term:07d}',
                'aspect': 'MFO'
            })
    
    labels = pd.DataFrame(dummy_labels)
    print(f"Created {len(labels)} protein-function annotations")
    
    # Create dummy GO term descriptions
    go_descriptions = []
    for i in range(n_go_terms):
        go_descriptions.append({
            'term': f'GO:{i:07d}',
            'description': f'Molecular function {i}'
        })
    
    go_term_descriptions = pd.DataFrame(go_descriptions)
    go_term_descriptions.to_csv(assets("go_term_descriptions.csv"), index=False)
    
    # Merge labels with descriptions
    labels = labels.merge(go_term_descriptions, on="term")
    print(f"Labels with descriptions: {len(labels)} rows")
    
    # Create dummy protein sequences
    amino_acids = list("ARNDCQEGHILKMFPSTWYV")
    sequences = []
    for protein_id in range(n_proteins):
        # Random sequence length between 50-500
        seq_len = np.random.randint(50, 500)
        sequence = ''.join(np.random.choice(amino_acids, seq_len))
        sequences.append({
            'EntryID': f'PROTEIN_{protein_id:06d}',
            'Sequence': sequence,
            'Length': seq_len,
            'taxonomyID': 9606  # Human
        })
    
    sequence_df = pd.DataFrame(sequences)
    
    # Merge with taxonomy and labels
    sequence_df = sequence_df.merge(labels, on="EntryID")
    print(f"Final dataset: {sequence_df['EntryID'].nunique()} proteins with {sequence_df['term'].nunique()} functions")
    
    # Filter for common functions (appears in at least 10 proteins)
    common_functions = (
        sequence_df["term"]
        .value_counts()[sequence_df["term"].value_counts() >= 10]
        .index
    )
    sequence_df = sequence_df[sequence_df["term"].isin(common_functions)]
    print(f"After filtering: {sequence_df['EntryID'].nunique()} proteins with {sequence_df['term'].nunique()} functions")
    
    # Convert to multi-label format
    sequence_df = (
        sequence_df[["EntryID", "Sequence", "Length", "term"]]
        .assign(value=1)
        .pivot(
            index=["EntryID", "Sequence", "Length"], columns="term", values="value"
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    print(f"Multi-label format: {sequence_df.shape}")
    
    # Filter by length
    sequence_df = sequence_df[sequence_df["Length"] <= 500]
    print(f"After length filter: {sequence_df.shape}")
    
    # 2.4.2. Splitting the Dataset
    print("2.4.2. Splitting the Dataset...")
    
    train_sequence_ids, valid_test_sequence_ids = train_test_split(
        list(set(sequence_df["EntryID"])), test_size=0.40, random_state=42
    )
    valid_sequence_ids, test_sequence_ids = train_test_split(
        valid_test_sequence_ids, test_size=0.50, random_state=42
    )
    
    sequence_splits = {
        "train": sequence_df[sequence_df["EntryID"].isin(train_sequence_ids)],
        "valid": sequence_df[sequence_df["EntryID"].isin(valid_sequence_ids)],
        "test": sequence_df[sequence_df["EntryID"].isin(test_sequence_ids)],
    }
    
    for split, df in sequence_splits.items():
        print(f"{split} has {len(df)} entries.")
    
    # =============================================================================
    # 2.4.3. Converting Protein Sequences into Their Mean Embeddings
    # =============================================================================
    print("\n2.4.3. Converting Protein Sequences into Their Mean Embeddings...")
    
    model_checkpoint = "facebook/esm2_t30_150M_UR50D"
    print(f"Loading ESM2 model: {model_checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = EsmModel.from_pretrained(model_checkpoint)
    
    # Store embeddings for each split
    for split, df in sequence_splits.items():
        print(f"Processing {split} split...")
        store_sequence_embeddings(
            sequence_df=df,
            store_prefix=assets(f"protein_dataset_{split}"),
            tokenizer=tokenizer,
            model=model,
        )
    
    # Load the training data back
    train_df = load_sequence_embeddings(
        assets("protein_dataset_train"),
        model_checkpoint=model_checkpoint,
    )
    print(f"Loaded training data: {train_df.shape}")
    
    # =============================================================================
    # 2.5. Training the Model
    # =============================================================================
    print("\n2.5. Training the Model...")
    
    # Build dataset splits
    dataset_splits = {}
    for split in ["train", "valid", "test"]:
        dataset_splits[split] = convert_to_tfds(
            df=load_sequence_embeddings(
                store_file_prefix=f"{assets('protein_dataset')}_{split}",
                model_checkpoint=model_checkpoint,
            ),
            is_training=(split == "train"),
        )
    
    # Get a batch to initialize the model
    batch_size = 32
    batch = next(dataset_splits["train"].batch(batch_size).as_numpy_iterator())
    print(f"Batch shapes: embedding={batch['embedding'].shape}, target={batch['target'].shape}")
    
    # Initialize model
    targets = list(train_df.columns[train_df.columns.str.contains("GO:")])
    print(f"Number of target functions: {len(targets)}")
    
    mlp = Model(num_targets=len(targets))
    
    # Initialize training state
    rng = jax.random.PRNGKey(42)
    rng, rng_init = jax.random.split(key=rng, num=2)
    
    state = mlp.create_train_state(
        rng=rng_init, 
        dummy_input=batch["embedding"], 
        tx=optax.adam(0.001)
    )
    
    # Train the model
    print("Starting training...")
    state, metrics = train(
        state=state,
        dataset_splits=dataset_splits,
        batch_size=batch_size,
        num_steps=300,
        eval_every=30,
    )
    
    # =============================================================================
    # 2.5.2. Examining the Model Predictions
    # =============================================================================
    print("\n2.5.2. Examining the Model Predictions...")
    
    # Load validation data
    valid_df = load_sequence_embeddings(
        store_file_prefix=f"{assets('protein_dataset')}_valid",
        model_checkpoint=model_checkpoint,
    )
    
    # Generate predictions
    valid_probs = []
    for valid_batch in dataset_splits["valid"].batch(1).as_numpy_iterator():
        logits = state.apply_fn({"params": state.params}, x=valid_batch["embedding"])
        valid_probs.extend(jax.nn.sigmoid(logits))
    
    valid_true_df = valid_df[["EntryID"] + targets].set_index("EntryID")
    valid_prob_df = pd.DataFrame(
        np.stack(valid_probs), columns=targets, index=valid_true_df.index
    )
    
    print(f"Validation predictions shape: {valid_prob_df.shape}")
    
    # Calculate metrics by function
    metrics_by_function = {}
    for function in targets:
        metrics_by_function[function] = compute_metrics(
            valid_true_df[function].values, valid_prob_df[function].values
        )
    
    overview_valid = (
        pd.DataFrame(metrics_by_function)
        .T.merge(go_term_descriptions, left_index=True, right_on="term")
        .set_index("term")
        .sort_values("auprc", ascending=False)
    )
    
    print("\nTop 10 performing functions:")
    print(overview_valid.head(10)[["description", "auprc", "auroc"]])
    
    # =============================================================================
    # 2.5.4. Final Check on Test Set
    # =============================================================================
    print("\n2.5.4. Final Check on Test Set...")
    
    eval_metrics = []
    for split in ["valid", "test"]:
        split_metrics = []
        for eval_batch in dataset_splits[split].batch(32).as_numpy_iterator():
            split_metrics.append(eval_step(state, eval_batch))
        eval_metrics.append(
            {"split": split, **pd.DataFrame(split_metrics).mean(axis=0).to_dict()}
        )
    
    final_results = pd.DataFrame(eval_metrics)
    print("Final Results:")
    print(final_results)
    
    print("\n=== Chapter 2 Complete ===")
    print("Model trained successfully!")
    print(f"Final validation AUPRC: {final_results[final_results['split'] == 'valid']['auprc'].iloc[0]:.4f}")
    print(f"Final test AUPRC: {final_results[final_results['split'] == 'test']['auprc'].iloc[0]:.4f}")

if __name__ == "__main__":
    main()
