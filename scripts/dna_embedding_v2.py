#!/usr/bin/env python3
"""
Script: dna_embedding_v2.py (using shared library)
Description: Extract DNA sequence embeddings using Nucleotide Transformer

This version uses the shared library to reduce code duplication.

Usage:
    python scripts/dna_embedding_v2.py --input FILE --output FILE

Example:
    python scripts/dna_embedding_v2.py --sequences "ATCG" --output results/embeddings.npz
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import sys
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

# Essential scientific packages
import numpy as np
import jax
import jax.numpy as jnp

# Shared library imports
sys.path.insert(0, str(Path(__file__).parent))
from lib.io import load_sequences_from_file, save_embeddings_npz, load_json_config
from lib.models import load_nucleotide_model_for_embedding, tokenize_dna_sequences, AVAILABLE_MODELS

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model": "50M_multi_species_v2",
    "layer": 12,
    "max_positions": 32,
    "default_sequences": [
        "ATTCCGAAATCGCTGACCGATCGTACGAAA",
        "ATTTCTCTCTCTCTCTGAGATCGATCGATCGATATCTCTCGAGCTAGC"
    ]
}

# ==============================================================================
# Core Functions (extracted and simplified from use case)
# ==============================================================================
def extract_sequence_embeddings(parameters, forward_fn, tokens, layer: int):
    """Extract embeddings from tokenized sequences."""
    # Initialize random key
    random_key = jax.random.PRNGKey(0)

    # Run inference
    outs = forward_fn.apply(parameters, random_key, tokens)

    # Get embeddings
    embeddings_key = f"embeddings_{layer}"
    if embeddings_key not in outs:
        available_keys = list(outs.keys())
        raise ValueError(f"Embeddings for layer {layer} not found. Available: {available_keys}")

    return outs[embeddings_key], outs

def process_raw_embeddings(embeddings, tokens, tokenizer):
    """Process embeddings to remove CLS token and handle padding."""
    # Remove CLS token (first position)
    embeddings = embeddings[:, 1:, :]

    # Create padding mask
    padding_mask = jnp.expand_dims(tokens[:, 1:] != tokenizer.pad_token_id, axis=-1)

    # Apply mask
    masked_embeddings = embeddings * padding_mask

    # Calculate sequence lengths
    sequences_lengths = jnp.sum(padding_mask, axis=1)

    # Calculate mean embeddings per sequence
    mean_embeddings = jnp.sum(masked_embeddings, axis=1) / sequences_lengths

    return mean_embeddings, sequences_lengths, masked_embeddings

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_dna_embedding(
    sequences: Optional[List[str]] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract embeddings from DNA sequences using Nucleotide Transformer.

    Args:
        sequences: List of DNA sequences to process
        input_file: Path to file containing sequences (one per line)
        output_file: Path to save embeddings (NPZ format)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - mean_embeddings: Mean embeddings per sequence
            - token_embeddings: Full token-level embeddings
            - sequences: Input sequences
            - sequence_lengths: Length of each sequence
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_dna_embedding(sequences=["ATCG"], output_file="embeddings.npz")
        >>> print(result['mean_embeddings'].shape)
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Load sequences
    if input_file:
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        sequences = load_sequences_from_file(input_file)
    elif not sequences:
        sequences = config["default_sequences"]

    if not sequences:
        raise ValueError("No sequences provided")

    # Load model using shared library
    parameters, forward_fn, tokenizer, model_config = load_nucleotide_model_for_embedding(
        config["model"], config["layer"], config["max_positions"]
    )

    # Tokenize sequences using shared library
    tokens, tokens_str = tokenize_dna_sequences(sequences, tokenizer)

    # Extract embeddings
    raw_embeddings, outputs = extract_sequence_embeddings(
        parameters, forward_fn, tokens, config["layer"]
    )

    # Process embeddings
    mean_embeddings, seq_lengths, token_embeddings = process_raw_embeddings(
        raw_embeddings, tokens, tokenizer
    )

    # Prepare results
    result = {
        "mean_embeddings": np.array(mean_embeddings),
        "token_embeddings": np.array(token_embeddings),
        "sequences": sequences,
        "sequence_lengths": np.array(seq_lengths.flatten()),
        "metadata": {
            "config": config,
            "model_config": {
                "num_layers": model_config.num_layers,
                "embed_dim": model_config.embed_dim,
                "model_name": config["model"]
            },
            "embedding_shape": mean_embeddings.shape,
            "num_sequences": len(sequences)
        },
        "output_file": None
    }

    # Save output if requested using shared library
    if output_file:
        output_path = Path(output_file)
        save_data = {
            "embeddings": result["mean_embeddings"],
            "sequences": sequences,
            "sequence_lengths": result["sequence_lengths"],
            "token_embeddings": result["token_embeddings"]
        }
        save_embeddings_npz(save_data, output_path)
        result["output_file"] = str(output_path)

    return result

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--sequences', nargs='+', help='DNA sequences to process')
    parser.add_argument('--input', '-i', help='Input file with sequences (one per line)')
    parser.add_argument('--output', '-o', help='Output file path (NPZ format)')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--model', choices=AVAILABLE_MODELS, help='Model to use')
    parser.add_argument('--layer', type=int, help='Layer to extract embeddings from')
    parser.add_argument('--max-positions', type=int, help='Maximum sequence positions')

    args = parser.parse_args()

    # Load config if provided using shared library
    config = None
    if args.config:
        config = load_json_config(args.config)

    # Override config with CLI args
    overrides = {}
    if args.model:
        overrides['model'] = args.model
    if args.layer:
        overrides['layer'] = args.layer
    if args.max_positions:
        overrides['max_positions'] = args.max_positions

    try:
        # Run embedding extraction
        result = run_dna_embedding(
            sequences=args.sequences,
            input_file=args.input,
            output_file=args.output,
            config=config,
            **overrides
        )

        print(f"✅ Success: Processed {result['metadata']['num_sequences']} sequences")
        print(f"   Embedding shape: {result['metadata']['embedding_shape']}")
        if result['output_file']:
            print(f"   Saved to: {result['output_file']}")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == '__main__':
    main()