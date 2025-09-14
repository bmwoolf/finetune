import os
import math
import obonet
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, EsmModel
from .utils import get_device

# handling protein data
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

def load_cafa3_data(data_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load and process CAFA3 protein data from raw CSV files."""
    print("Loading CAFA3 data...")
    
    # Load the raw sequence data
    cco_df = pd.read_csv(os.path.join(data_dir, "sequence_df_cco.csv"))
    mfo_df = pd.read_csv(os.path.join(data_dir, "sequence_df_mfo.csv"))
    
    print(f"Loaded CCO data: {cco_df.shape}")
    print(f"Loaded MFO data: {mfo_df.shape}")
    
    # Combine both aspects
    combined_df = pd.concat([cco_df, mfo_df], ignore_index=True)
    print(f"Combined data: {combined_df.shape}")
    
    # Create a pivot table to get one row per protein with all GO terms as columns
    protein_df = combined_df.pivot_table(
        index=['EntryID', 'Sequence', 'taxonomyID', 'Length'], 
        columns='term', 
        values='aspect', 
        fill_value=0,
        aggfunc='count'
    ).reset_index()
    
    # Convert to binary (1 if protein has the term, 0 otherwise)
    go_columns = [col for col in protein_df.columns if col.startswith('GO:')]
    protein_df[go_columns] = (protein_df[go_columns] > 0).astype(int)
    
    print(f"Processed protein data: {protein_df.shape}")
    print(f"Number of GO terms: {len(go_columns)}")
    
    return protein_df, go_columns

def create_train_valid_test_splits(protein_df: pd.DataFrame, test_size: float = 0.2, valid_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/valid/test splits from protein data."""
    print("Creating train/valid/test splits...")
    
    # First split: separate test set
    train_valid_df, test_df = train_test_split(
        protein_df, 
        test_size=test_size, 
        random_state=42,
        stratify=protein_df['taxonomyID']  # Stratify by taxonomy for balanced splits
    )
    
    # Second split: separate train and validation
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_size/(1-test_size),  # Adjust for the remaining data
        random_state=42,
        stratify=train_valid_df['taxonomyID']
    )
    
    print(f"Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}")
    
    return train_df, valid_df, test_df

def generate_embeddings(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, model_name: str = "facebook/esm2_t30_150M_UR50D") -> dict:
    """Generate ESM2 embeddings for all protein sequences."""
    print(f"Loading ESM2 model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    device = get_device()
    
    print(f"Using device: {device}")
    
    # Generate embeddings for each split
    splits = {
        'train': train_df,
        'valid': valid_df, 
        'test': test_df
    }
    
    for split_name, df in splits.items():
        print(f"Generating embeddings for {split_name} set...")
        
        # Get unique sequences to avoid duplicate computation
        unique_sequences = df['Sequence'].unique()
        print(f"Unique sequences in {split_name}: {len(unique_sequences)}")
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        sequence_to_embedding = {}
        
        for i in range(0, len(unique_sequences), batch_size):
            batch_sequences = unique_sequences[i:i+batch_size]
            batch_embeddings = get_mean_embeddings(
                batch_sequences.tolist(), 
                tokenizer, 
                model, 
                device
            )
            embeddings.append(batch_embeddings)
            
            # Map sequences to their embeddings
            for seq, emb in zip(batch_sequences, batch_embeddings):
                sequence_to_embedding[seq] = emb
        
        # Add embeddings to dataframe
        embedding_columns = [f"ME:{i+1}" for i in range(embeddings[0].shape[1])]
        embedding_data = []
        
        for _, row in df.iterrows():
            embedding = sequence_to_embedding[row['Sequence']]
            embedding_data.append(embedding)
        
        embedding_df = pd.DataFrame(embedding_data, columns=embedding_columns)
        df_with_embeddings = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)
        
        # Save the processed data
        from .utils import DATA_DIR
        output_file = os.path.join(DATA_DIR, "proteins", "datasets", f"protein_dataset_{split_name}_{model_name.replace('/', '_')}.feather")
        df_with_embeddings.to_feather(output_file)
        print(f"Saved {split_name} data with embeddings to: {output_file}")
        
        splits[split_name] = df_with_embeddings
    
    return splits

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
