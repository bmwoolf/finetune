"""
ESM2 protein embeddings example.
"""

import torch
from transformers import EsmModel, EsmTokenizer


def get_protein_embeddings(protein_sequence: str, model, tokenizer):
    """Get protein embeddings from ESM2 model."""
    inputs = tokenizer(protein_sequence, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # mean pooling over sequence length
    attention_mask = inputs.attention_mask
    embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
    pooled_embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    
    return pooled_embeddings


if __name__ == "__main__":
    # load model
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    
    # example protein sequence
    protein_sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    
    # get embeddings
    embeddings = get_protein_embeddings(protein_sequence, model, tokenizer)
    print(f"Embeddings shape: {embeddings.shape}")
