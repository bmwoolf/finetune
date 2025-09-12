#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, EsmModel
from .src.data import (
    get_go_term_descriptions,
    store_sequence_embeddings,
    load_sequence_embeddings
)
from .src.model import (
    Model,
    convert_to_tfds,
    compute_metrics,
    train_step,
    eval_step,
    train
)
from .src.utils import assets

# Set up paths for local environment
BASE_DIR = Path(__file__).parent.parent  # Go up one level from finetune/ to project root
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# execution pipeline
def main():
    # set random seeds for reproducibility
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
