#!/usr/bin/env python3
"""
Script: attention_visualization.py
Description: Extract and visualize attention maps from DNA sequences

Original Use Case: examples/use_case_2_attention_visualization.py
Dependencies Removed: matplotlib inline code simplified

Usage:
    python scripts/attention_visualization.py --input FILE --output FILE

Example:
    python scripts/attention_visualization.py --sequences "ATCG" --output attention.png
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json

# Essential scientific packages
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

# Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Model loading (cannot be inlined - complex neural network)
from nucleotide_transformer.pretrained import get_pretrained_model

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model": "50M_multi_species_v2",
    "layer": 1,
    "head": 4,
    "max_positions": 32,
    "dpi": 300,
    "default_sequences": [
        "ATTCCGAAATCGCTGACCGATCGTACGAAA",
        "ATTTCTCTCTCTCTCTGAGATCGATCGATCGATATCTCTCGAGCTAGC"
    ]
}

AVAILABLE_MODELS = [
    "500M_human_ref", "500M_1000G", "2B5_1000G", "2B5_multi_species",
    "50M_multi_species_v2", "100M_multi_species_v2",
    "250M_multi_species_v2", "500M_multi_species_v2", "1B_agro_nt"
]

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def load_sequences_from_file(file_path: Path) -> List[str]:
    """Load DNA sequences from text file. Inlined for simplicity."""
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                sequences.append(line)
    return sequences

def save_attention_plot(fig, file_path: Path, dpi: int = 300) -> None:
    """Save attention plot to file. Inlined for simplicity."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, dpi=dpi, bbox_inches='tight')

# ==============================================================================
# Core Functions (extracted and simplified from use case)
# ==============================================================================
def load_nucleotide_model_attention(model_name: str, layer: int, head: int, max_positions: int):
    """Load pretrained Nucleotide Transformer model for attention extraction."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available. Choose from: {AVAILABLE_MODELS}")

    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(),
        attention_maps_to_save=((layer, head),),
        max_positions=max_positions,
    )
    forward_fn = hk.transform(forward_fn)

    return parameters, forward_fn, tokenizer, config

def tokenize_dna_sequences(sequences: List[str], tokenizer):
    """Tokenize DNA sequences for the model."""
    batch_tokenized = tokenizer.batch_tokenize(sequences)
    tokens_ids = [b[1] for b in batch_tokenized]
    tokens_str = [b[0] for b in batch_tokenized]
    tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
    return tokens, tokens_str

def extract_attention_maps(parameters, forward_fn, tokens, layer: int, head: int):
    """Extract attention maps from tokenized sequences."""
    # Initialize random key
    random_key = jax.random.PRNGKey(0)

    # Run inference
    outs = forward_fn.apply(parameters, random_key, tokens)

    # Get attention maps
    attention_key = f"attention_map_layer_{layer}_number_{head}"
    if attention_key not in outs:
        available_keys = list(outs.keys())
        raise ValueError(f"Attention map for layer {layer} head {head} not found. Available: {available_keys}")

    return outs[attention_key], outs

def create_attention_visualization(attention_maps, tokens_str, sequences, config: Dict[str, Any]):
    """Create attention visualization plots."""
    batch_size = attention_maps.shape[0]

    # Calculate subplot layout
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

    attention_stats = []

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

        # Calculate attention statistics
        max_attention = np.max(attention)
        mean_attention = np.mean(attention)
        max_pos = np.unravel_index(np.argmax(attention), attention.shape)
        diagonal_attention = np.mean(np.diag(attention))

        attention_stats.append({
            'sequence_id': seq_id + 1,
            'sequence': sequences[seq_id],
            'max_attention': float(max_attention),
            'mean_attention': float(mean_attention),
            'diagonal_attention': float(diagonal_attention),
            'most_attended_from': tokens[max_pos[0] + 1],
            'most_attended_to': tokens[max_pos[1] + 1],
            'attention_shape': attention.shape
        })

    # Hide extra subplots if batch_size is odd
    if batch_size < len(axes):
        for i in range(batch_size, len(axes)):
            axes[i].set_visible(False)

    plt.tight_layout()

    return fig, attention_stats

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_attention_visualization(
    sequences: Optional[List[str]] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract and visualize attention maps from DNA sequences.

    Args:
        sequences: List of DNA sequences to process
        input_file: Path to file containing sequences (one per line)
        output_file: Path to save visualization (PNG format)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - attention_maps: Raw attention matrices
            - attention_stats: Statistics for each sequence
            - sequences: Input sequences
            - figure: Matplotlib figure object
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_attention_visualization(sequences=["ATCG"], output_file="attention.png")
        >>> print(result['attention_stats'][0]['max_attention'])
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

    # Load model
    parameters, forward_fn, tokenizer, model_config = load_nucleotide_model_attention(
        config["model"], config["layer"], config["head"], config["max_positions"]
    )

    # Tokenize sequences
    tokens, tokens_str = tokenize_dna_sequences(sequences, tokenizer)

    # Extract attention maps
    attention_maps, outputs = extract_attention_maps(
        parameters, forward_fn, tokens, config["layer"], config["head"]
    )

    # Create visualization
    fig, attention_stats = create_attention_visualization(
        attention_maps, tokens_str, sequences, config
    )

    # Prepare results
    result = {
        "attention_maps": np.array(attention_maps),
        "attention_stats": attention_stats,
        "sequences": sequences,
        "figure": fig,
        "metadata": {
            "config": config,
            "model_config": {
                "num_layers": model_config.num_layers,
                "attention_heads": model_config.attention_heads,
                "model_name": config["model"]
            },
            "attention_shape": attention_maps.shape,
            "num_sequences": len(sequences),
            "layer": config["layer"],
            "head": config["head"]
        },
        "output_file": None
    }

    # Save output if requested
    if output_file:
        output_path = Path(output_file)
        save_attention_plot(fig, output_path, config["dpi"])
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
    parser.add_argument('--output', '-o', help='Output file path (PNG format)')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--model', choices=AVAILABLE_MODELS, help='Model to use')
    parser.add_argument('--layer', type=int, help='Layer to extract attention from')
    parser.add_argument('--head', type=int, help='Attention head to visualize')
    parser.add_argument('--max-positions', type=int, help='Maximum sequence positions')
    parser.add_argument('--dpi', type=int, help='Output image DPI')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Fix config structure for nested model config
    if config and 'model' in config and isinstance(config['model'], dict):
        if 'name' in config['model']:
            config['model'] = config['model']['name']

    # Override config with CLI args
    overrides = {}
    for arg, value in [('model', args.model), ('layer', args.layer), ('head', args.head),
                       ('max_positions', args.max_positions), ('dpi', args.dpi)]:
        if value is not None:
            overrides[arg] = value

    try:
        # Run attention visualization
        result = run_attention_visualization(
            sequences=args.sequences,
            input_file=args.input,
            output_file=args.output,
            config=config,
            **overrides
        )

        print(f"✅ Success: Processed {result['metadata']['num_sequences']} sequences")
        print(f"   Attention shape: {result['metadata']['attention_shape']}")
        print(f"   Layer {result['metadata']['layer']}, Head {result['metadata']['head']}")

        for stat in result['attention_stats']:
            print(f"   Sequence {stat['sequence_id']}: max_attention={stat['max_attention']:.3f}")

        if result['output_file']:
            print(f"   Saved to: {result['output_file']}")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == '__main__':
    main()