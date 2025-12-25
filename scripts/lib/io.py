"""
Shared I/O functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Union, Any, List, Dict
import json
import csv
import numpy as np

def load_sequences_from_file(file_path: Path) -> List[str]:
    """Load DNA sequences from text file. Inlined for simplicity."""
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                sequences.append(line)
    return sequences

def save_embeddings_npz(data: Dict[str, Any], file_path: Path) -> None:
    """Save embeddings to NPZ file. Inlined for simplicity."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(file_path, **data)

def save_predictions_csv(results: List[Dict], metrics: Dict, file_path: Path) -> None:
    """Save predictions to CSV file. Inlined from original."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow([
            'sequence_id', 'sequence', 'position', 'actual_token',
            'predicted_token_1', 'probability_1',
            'predicted_token_2', 'probability_2',
            'predicted_token_3', 'probability_3',
            'predicted_token_4', 'probability_4',
            'predicted_token_5', 'probability_5',
            'is_correct'
        ])

        # Write data
        for seq_result in results:
            for pos_result in seq_result['positions']:
                row = [
                    seq_result['sequence_id'],
                    seq_result['sequence'],
                    pos_result['position'],
                    pos_result['actual_token']
                ]

                # Add top 5 predictions
                for i in range(5):
                    if i < len(pos_result['predictions']):
                        pred = pos_result['predictions'][i]
                        row.extend([pred['token'], pred['probability']])
                    else:
                        row.extend(['', ''])

                # Add correctness
                is_correct = pos_result['predictions'][0]['is_correct'] if pos_result['predictions'] else False
                row.append(is_correct)

                writer.writerow(row)

def load_json_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON configuration file with model structure fixing."""
    with open(file_path) as f:
        config = json.load(f)

    # Fix config structure for nested model config
    if 'model' in config and isinstance(config['model'], dict):
        if 'name' in config['model']:
            config['model'] = config['model']['name']

    return config