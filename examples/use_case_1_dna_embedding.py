#!/usr/bin/env python3
"""
Use Case 1: DNA Sequence Embedding with Nucleotide Transformer

This script demonstrates how to extract embeddings from DNA sequences using
the Nucleotide Transformer model. These embeddings can be used for
downstream tasks like classification, similarity analysis, and functional prediction.

Environment: ./env (Python 3.10)
Complexity: medium
Priority: high
"""

import os
import sys
import argparse
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from nucleotide_transformer.pretrained import get_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from DNA sequences")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=[
            "ATTCCGAAATCGCTGACCGATCGTACGAAA",
            "ATTTCTCTCTCTCTCTGAGATCGATCGATCGATATCTCTCGAGCTAGC",
        ],
        help="DNA sequences to process"
    )
    parser.add_argument(
        "--model",
        default="50M_multi_species_v2",
        choices=[
            "500M_human_ref", "500M_1000G", "2B5_1000G", "2B5_multi_species",
            "50M_multi_species_v2", "100M_multi_species_v2",
            "250M_multi_species_v2", "500M_multi_species_v2", "1B_agro_nt"
        ],
        help="Model to use for embedding"
    )
    parser.add_argument(
        "--output",
        help="Output file to save embeddings (optional)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Layer to extract embeddings from"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=32,
        help="Maximum sequence positions"
    )
    return parser.parse_args()


def load_model(model_name, embeddings_layer, max_positions):
    """Load pretrained Nucleotide Transformer model."""
    print(f"Loading model: {model_name}")

    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(embeddings_layer,),
        attention_maps_to_save=(),
        max_positions=max_positions,
    )
    forward_fn = hk.transform(forward_fn)

    print(f"Model loaded with {config.num_layers} layers, {config.embed_dim} embedding dimensions")
    return parameters, forward_fn, tokenizer, config


def tokenize_sequences(sequences, tokenizer):
    """Tokenize DNA sequences."""
    tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
    tokens_str = [b[0] for b in tokenizer.batch_tokenize(sequences)]
    tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

    print(f"Tokenized {len(sequences)} sequences")
    for i, (seq, tokens_s) in enumerate(zip(sequences, tokens_str)):
        print(f"  Sequence {i+1}: {seq[:50]}{'...' if len(seq) > 50 else ''}")
        print(f"  Tokens: {tokens_s[:10]}")
        print()

    return tokens, tokens_str


def extract_embeddings(parameters, forward_fn, tokens, layer):
    """Extract embeddings from tokenized sequences."""
    print("Extracting embeddings...")

    # Initialize random key
    random_key = jax.random.PRNGKey(0)

    # Run inference
    outs = forward_fn.apply(parameters, random_key, tokens)

    # Get embeddings
    embeddings_key = f"embeddings_{layer}"
    if embeddings_key not in outs:
        available_keys = list(outs.keys())
        raise ValueError(f"Embeddings for layer {layer} not found. Available: {available_keys}")

    embeddings = outs[embeddings_key]
    print(f"Raw embeddings shape: {embeddings.shape}")

    return embeddings, outs


def process_embeddings(embeddings, tokens, tokenizer):
    """Process embeddings to remove CLS token and padding."""
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

    print(f"Processed embeddings shape: {mean_embeddings.shape}")
    print(f"Sequence lengths: {sequences_lengths.flatten()}")

    return mean_embeddings, sequences_lengths, masked_embeddings


def save_embeddings(embeddings, sequences, output_file):
    """Save embeddings to file."""
    data = {
        'embeddings': np.array(embeddings),
        'sequences': sequences
    }
    np.savez(output_file, **data)
    print(f"Embeddings saved to: {output_file}")


def main():
    args = parse_args()

    try:
        # Load model
        parameters, forward_fn, tokenizer, config = load_model(
            args.model, args.layer, args.max_positions
        )

        # Tokenize sequences
        tokens, tokens_str = tokenize_sequences(args.sequences, tokenizer)

        # Extract embeddings
        raw_embeddings, outputs = extract_embeddings(parameters, forward_fn, tokens, args.layer)

        # Process embeddings
        mean_embeddings, seq_lengths, token_embeddings = process_embeddings(
            raw_embeddings, tokens, tokenizer
        )

        # Print results
        print("\nResults Summary:")
        print(f"Number of sequences: {len(args.sequences)}")
        print(f"Mean embeddings shape: {mean_embeddings.shape}")
        print(f"Embedding dimension: {mean_embeddings.shape[-1]}")

        # Save if requested
        if args.output:
            save_embeddings(mean_embeddings, args.sequences, args.output)

        print("\n✓ DNA sequence embedding completed successfully!")

        return {
            'mean_embeddings': np.array(mean_embeddings),
            'token_embeddings': np.array(token_embeddings),
            'sequences': args.sequences,
            'sequence_lengths': np.array(seq_lengths),
            'config': config
        }

    except Exception as e:
        print(f"✗ Error during DNA sequence embedding: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()