"""
Shared model functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from typing import List, Tuple, Any
import jax.numpy as jnp
import haiku as hk
from nucleotide_transformer.pretrained import get_pretrained_model

# Available models constant
AVAILABLE_MODELS = [
    "500M_human_ref", "500M_1000G", "2B5_1000G", "2B5_multi_species",
    "50M_multi_species_v2", "100M_multi_species_v2",
    "250M_multi_species_v2", "500M_multi_species_v2", "1B_agro_nt"
]

def validate_model_name(model_name: str) -> None:
    """Validate model name is available."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available. Choose from: {AVAILABLE_MODELS}")

def tokenize_dna_sequences(sequences: List[str], tokenizer) -> Tuple[jnp.ndarray, List[List[str]]]:
    """Tokenize DNA sequences for the model."""
    batch_tokenized = tokenizer.batch_tokenize(sequences)
    tokens_ids = [b[1] for b in batch_tokenized]
    tokens_str = [b[0] for b in batch_tokenized]
    tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
    return tokens, tokens_str

def load_nucleotide_model_for_embedding(model_name: str, embeddings_layer: int, max_positions: int):
    """Load pretrained Nucleotide Transformer model for embedding extraction."""
    validate_model_name(model_name)

    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(embeddings_layer,),
        attention_maps_to_save=(),
        max_positions=max_positions,
    )
    forward_fn = hk.transform(forward_fn)

    return parameters, forward_fn, tokenizer, config

def load_nucleotide_model_for_attention(model_name: str, layer: int, head: int, max_positions: int):
    """Load pretrained Nucleotide Transformer model for attention extraction."""
    validate_model_name(model_name)

    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(),
        attention_maps_to_save=((layer, head),),
        max_positions=max_positions,
    )
    forward_fn = hk.transform(forward_fn)

    return parameters, forward_fn, tokenizer, config

def load_nucleotide_model_for_prediction(model_name: str, max_positions: int):
    """Load pretrained Nucleotide Transformer model for prediction."""
    validate_model_name(model_name)

    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(),
        attention_maps_to_save=(),
        max_positions=max_positions,
    )
    forward_fn = hk.transform(forward_fn)

    return parameters, forward_fn, tokenizer, config