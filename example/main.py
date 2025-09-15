#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import torch
import optax
from pathlib import Path

# add the parent directory to the path so we can import tools
# will need to clean this up
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.data import (
    get_go_term_descriptions,
    load_cafa3_data,
    create_train_valid_test_splits,
    generate_embeddings,
    store_sequence_embeddings,
    load_sequence_embeddings,
    get_mean_embeddings
)
from tools.model import (
    Model,
    convert_to_tfds,
    compute_metrics,
    train_step,
    eval_step,
    train
)
from tools.utils import assets, get_device, DATA_DIR
from tools.visualization import (
    plot_training_progress,
    plot_function_analysis,
    plot_predictions_analysis,
    plot_metrics_summary
)

# set up paths for local environment
BASE_DIR = Path(__file__).parent.parent
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BASE_DIR / "example/models", exist_ok=True)

def main():
    # set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    jax.random.PRNGKey(42)
    
    # data preparation
    data_dir = os.path.join(str(DATA_DIR), "proteins", "datasets")
    
    # load GO term descriptions
    go_term_descriptions = get_go_term_descriptions(
        os.path.join(str(DATA_DIR), "go_term_descriptions.csv")
    )
    print(f"Loaded {len(go_term_descriptions)} GO term descriptions")
    
    # check if we have pre-computed embeddings
    model_name = "facebook/esm2_t30_150M_UR50D"
    train_file = os.path.join(data_dir, f"protein_dataset_train_{model_name.replace('/', '_')}.feather")
    
    if os.path.exists(train_file):
        print("Loading pre-computed embeddings...")
        
        # load the datasets
        train_df = load_sequence_embeddings(
            os.path.join(data_dir, "protein_dataset_train"),
            model_name
        )
        valid_df = load_sequence_embeddings(
            os.path.join(data_dir, "protein_dataset_valid"),
            model_name
        )
        test_df = load_sequence_embeddings(
            os.path.join(data_dir, "protein_dataset_test"),
            model_name
        )
        
        print(f"Loaded pre-computed data: train={train_df.shape}, valid={valid_df.shape}, test={test_df.shape}")
        
    else:
        print("Pre-computed embeddings not found. Generating from raw data")
        
        # load raw CAFA3 data
        protein_df, go_columns = load_cafa3_data(data_dir)
        
        # create train/valid/test splits
        train_df, valid_df, test_df = create_train_valid_test_splits(protein_df)
        
        # generate ESM2 embeddings
        splits = generate_embeddings(train_df, valid_df, test_df, model_name)
        train_df, valid_df, test_df = splits['train'], splits['valid'], splits['test']
    
    print(f"Final data shapes: train={train_df.shape}, valid={valid_df.shape}, test={test_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    
    # get target columns (GO terms)
    targets = [col for col in train_df.columns if col.startswith("GO:")]
    print(f"Number of target functions: {len(targets)}")
    
    # build dataset splits
    dataset_splits = {}
    for split, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        dataset_splits[split] = convert_to_tfds(
            df=df,
            is_training=(split == "train"),
        )
    
    # get a batch to initialize the model
    batch_size = 32
    batch = next(dataset_splits["train"].batch(batch_size).as_numpy_iterator())
    print(f"Batch shapes: embedding={batch['embedding'].shape}, target={batch['target'].shape}")
    
    # initialize model
    mlp = Model(num_targets=len(targets))
    
    # initialize training state
    rng = jax.random.PRNGKey(42)
    rng, rng_init = jax.random.split(key=rng, num=2)
    
    state = mlp.create_train_state(
        rng=rng_init, 
        dummy_input=batch["embedding"], 
        tx=optax.adam(0.001)
    )
    
    # train the model
    print("Starting training...")
    state, metrics = train(
        state=state,
        dataset_splits=dataset_splits,
        batch_size=batch_size,
        num_steps=2000,
        eval_every=100,
    )
    
    # training progress
    print("Plotting training progress...")
    plot_training_progress(metrics)
    
    # evaluation
    print("\nExamining the Model Predictions")
    
    # generate predictions
    valid_probs = []
    for valid_batch in dataset_splits["valid"].batch(1).as_numpy_iterator():
        logits = state.apply_fn({"params": state.params}, x=valid_batch["embedding"])
        valid_probs.extend(jax.nn.sigmoid(logits))
    
    valid_true_df = valid_df[["EntryID"] + targets].set_index("EntryID")
    valid_prob_df = pd.DataFrame(
        np.stack(valid_probs), columns=targets, index=valid_true_df.index
    )
    
    print(f"Validation predictions shape: {valid_prob_df.shape}")
    
    # calculate metrics by function
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
    
    # function analysis
    print("Plotting function analysis...")
    plot_function_analysis(overview_valid, targets, valid_true_df, valid_prob_df)
    
    # predictions analysis
    print("Plotting predictions analysis...")
    plot_predictions_analysis(valid_true_df, valid_prob_df, targets)
    
    # final check on test set
    print("\nFinal Check on Test Set")
    
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
    
    # 4. final metrics summary
    print("Plotting final metrics summary...")
    plot_metrics_summary(final_results)
    
    print("Model trained successfully!")
    print(f"Final validation AUPRC: {final_results[final_results['split'] == 'valid']['auprc'].iloc[0]:.4f}")
    print(f"Final test AUPRC: {final_results[final_results['split'] == 'test']['auprc'].iloc[0]:.4f}")

if __name__ == "__main__":
    main()
