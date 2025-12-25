"""
Shared visualization functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def save_attention_plot(fig, file_path: Path, dpi: int = 300) -> None:
    """Save attention plot to file. Inlined for simplicity."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, dpi=dpi, bbox_inches='tight')

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