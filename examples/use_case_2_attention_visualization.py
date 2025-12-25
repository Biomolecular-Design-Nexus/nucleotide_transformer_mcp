#!/usr/bin/env python3
"""
Use Case 2: DNA Sequence Attention Visualization

This script demonstrates how to extract and visualize attention maps from
DNA sequences using the Nucleotide Transformer model. Attention maps show
which parts of the sequence the model focuses on when making predictions.

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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nucleotide_transformer.pretrained import get_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize attention maps from DNA sequences")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=[
            "ATTCCGAAATCGCTGACCGATCGTACGAAA",
            "ATTTCTCTCTCTCTCTGAGATCGATCGATCGATATCTCTCGAGCTAGC",
        ],
        help="DNA sequences to analyze"
    )
    parser.add_argument(
        "--model",
        default="50M_multi_species_v2",
        choices=[
            "500M_human_ref", "500M_1000G", "2B5_1000G", "2B5_multi_species",
            "50M_multi_species_v2", "100M_multi_species_v2",
            "250M_multi_species_v2", "500M_multi_species_v2", "1B_agro_nt"
        ],
        help="Model to use"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=1,
        help="Layer to extract attention from"
    )
    parser.add_argument(
        "--head",
        type=int,
        default=4,
        help="Attention head to visualize"
    )
    parser.add_argument(
        "--output",
        help="Output file to save attention visualization"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=32,
        help="Maximum sequence positions"
    )
    return parser.parse_args()


def load_model(model_name, layer, head, max_positions):
    """Load pretrained Nucleotide Transformer model with attention extraction."""
    print(f"Loading model: {model_name}")
    print(f"Will extract attention from layer {layer}, head {head}")

    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(),
        attention_maps_to_save=((layer, head),),
        max_positions=max_positions,
    )
    forward_fn = hk.transform(forward_fn)

    print(f"Model loaded with {config.num_layers} layers, {config.attention_heads} attention heads")
    return parameters, forward_fn, tokenizer, config


def extract_attention(parameters, forward_fn, tokens, layer, head):
    """Extract attention maps from tokenized sequences."""
    print("Extracting attention maps...")

    # Initialize random key
    random_key = jax.random.PRNGKey(0)

    # Run inference
    outs = forward_fn.apply(parameters, random_key, tokens)

    # Get attention maps
    attention_key = f"attention_map_layer_{layer}_number_{head}"
    if attention_key not in outs:
        available_keys = list(outs.keys())
        raise ValueError(f"Attention map for layer {layer} head {head} not found. Available: {available_keys}")

    attention_maps = outs[attention_key]
    print(f"Attention maps shape: {attention_maps.shape}")

    return attention_maps, outs


def visualize_attention(attention_maps, tokens_str, sequences, output_file=None):
    """Visualize attention maps for each sequence."""
    batch_size = attention_maps.shape[0]

    # Calculate number of rows and columns for subplots
    if batch_size == 1:
        nrows, ncols = 1, 1
    elif batch_size == 2:
        nrows, ncols = 1, 2
    else:
        nrows = int(np.ceil(batch_size / 2))
        ncols = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows))

    if batch_size == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes if batch_size > 1 else [axes]
    else:
        axes = axes.flatten()

    for seq_id in range(batch_size):
        # Get sequence length (excluding CLS token and padding)
        tokens = tokens_str[seq_id]
        seq_length = len([t for t in tokens[1:] if t != '<PAD>'])

        # Extract attention map for this sequence (exclude CLS token)
        attention = attention_maps[seq_id, 1:(seq_length + 1), 1:(seq_length + 1)]

        # Create plot
        ax = axes[seq_id] if batch_size > 1 else axes[0]
        im = ax.imshow(attention, cmap='Blues')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        # Set labels
        sequence_tokens = tokens[1:(seq_length + 1)]
        ax.set_xticks(list(range(seq_length)))
        ax.set_yticks(list(range(seq_length)))
        ax.set_xticklabels(sequence_tokens, rotation=45, ha='right')
        ax.set_yticklabels(sequence_tokens)

        # Set title
        seq_preview = sequences[seq_id][:30] + "..." if len(sequences[seq_id]) > 30 else sequences[seq_id]
        ax.set_title(f"Sequence {seq_id + 1}: {seq_preview}")

        print(f"Sequence {seq_id + 1}:")
        print(f"  Length: {seq_length} tokens")
        print(f"  Sequence: {sequences[seq_id]}")
        print(f"  Tokens: {sequence_tokens}")
        print()

    # Hide extra subplots if batch_size is odd
    if batch_size < len(axes):
        for i in range(batch_size, len(axes)):
            axes[i].set_visible(False)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to: {output_file}")
    else:
        plt.show()

    return fig


def analyze_attention_patterns(attention_maps, tokens_str, sequences):
    """Analyze attention patterns and provide insights."""
    print("\n" + "="*50)
    print("ATTENTION ANALYSIS")
    print("="*50)

    for seq_id in range(attention_maps.shape[0]):
        tokens = tokens_str[seq_id]
        seq_length = len([t for t in tokens[1:] if t != '<PAD>'])

        # Get attention matrix for this sequence
        attention = attention_maps[seq_id, 1:(seq_length + 1), 1:(seq_length + 1)]

        # Calculate attention statistics
        max_attention = np.max(attention)
        mean_attention = np.mean(attention)

        # Find most attended positions
        max_pos = np.unravel_index(np.argmax(attention), attention.shape)

        print(f"\nSequence {seq_id + 1}:")
        print(f"  Sequence: {sequences[seq_id]}")
        print(f"  Max attention: {max_attention:.3f}")
        print(f"  Mean attention: {mean_attention:.3f}")
        print(f"  Most attended position: {tokens[max_pos[0] + 1]} -> {tokens[max_pos[1] + 1]} (attention: {attention[max_pos]:.3f})")

        # Analyze self-attention vs cross-attention
        diagonal_attention = np.mean(np.diag(attention))
        off_diagonal_attention = np.mean(attention - np.diag(np.diag(attention)))

        print(f"  Self-attention (diagonal): {diagonal_attention:.3f}")
        print(f"  Cross-attention (off-diagonal): {off_diagonal_attention:.3f}")


def main():
    args = parse_args()

    try:
        # Load model
        parameters, forward_fn, tokenizer, config = load_model(
            args.model, args.layer, args.head, args.max_positions
        )

        # Tokenize sequences
        print("Tokenizing sequences...")
        tokens_ids = [b[1] for b in tokenizer.batch_tokenize(args.sequences)]
        tokens_str = [b[0] for b in tokenizer.batch_tokenize(args.sequences)]
        tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

        # Extract attention maps
        attention_maps, outputs = extract_attention(
            parameters, forward_fn, tokens, args.layer, args.head
        )

        # Visualize attention
        fig = visualize_attention(attention_maps, tokens_str, args.sequences, args.output)

        # Analyze attention patterns
        analyze_attention_patterns(attention_maps, tokens_str, args.sequences)

        print(f"\n✓ Attention visualization completed successfully!")

        return {
            'attention_maps': np.array(attention_maps),
            'tokens': tokens_str,
            'sequences': args.sequences,
            'config': config
        }

    except Exception as e:
        print(f"✗ Error during attention visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()