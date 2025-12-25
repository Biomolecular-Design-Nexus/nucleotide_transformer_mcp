#!/usr/bin/env python3
"""
Script: nucleotide_prediction.py
Description: Predict nucleotide probabilities at each position

Original Use Case: examples/use_case_3_nucleotide_prediction.py
Dependencies Removed: csv module inlined

Usage:
    python scripts/nucleotide_prediction.py --input FILE --output FILE

Example:
    python scripts/nucleotide_prediction.py --sequences "ATCG" --output predictions.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import csv

# Essential scientific packages
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

# Model loading (cannot be inlined - complex neural network)
from nucleotide_transformer.pretrained import get_pretrained_model

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model": "50M_multi_species_v2",
    "max_positions": 32,
    "top_k": 5,
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

# ==============================================================================
# Core Functions (extracted and simplified from use case)
# ==============================================================================
def load_nucleotide_model_prediction(model_name: str, max_positions: int):
    """Load pretrained Nucleotide Transformer model for prediction."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available. Choose from: {AVAILABLE_MODELS}")

    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(),
        attention_maps_to_save=(),
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

def predict_nucleotide_probabilities(parameters, forward_fn, tokens):
    """Predict nucleotide probabilities for each position."""
    # Initialize random key
    random_key = jax.random.PRNGKey(0)

    # Run inference
    outs = forward_fn.apply(parameters, random_key, tokens)

    # Get logits and convert to probabilities
    logits = outs["logits"]
    probabilities = jax.nn.softmax(logits, axis=-1)

    return probabilities, logits

def analyze_nucleotide_predictions(probabilities, tokens_str, sequences, tokenizer, top_k: int = 5):
    """Analyze and format prediction results."""
    results = []

    for seq_id in range(len(sequences)):
        tokens = tokens_str[seq_id]
        seq_length = len([t for t in tokens[1:] if t != '<PAD>'])

        # Get probabilities for this sequence (exclude CLS token and padding)
        seq_probs = probabilities[seq_id, 1:(seq_length + 1)]
        seq_tokens = tokens[1:(seq_length + 1)]

        seq_results = []

        for pos_id, (token, prob_dist) in enumerate(zip(seq_tokens, seq_probs)):
            # Get top-k predictions
            sorted_indices = jnp.argsort(-prob_dist)
            sorted_probs = prob_dist[sorted_indices]

            position_results = []

            for k in range(min(top_k, len(sorted_indices))):
                predicted_token = tokenizer.id_to_token(int(sorted_indices[k]))
                probability = float(sorted_probs[k])

                position_results.append({
                    'rank': k + 1,
                    'token': predicted_token,
                    'probability': probability,
                    'is_correct': predicted_token == token
                })

            seq_results.append({
                'position': pos_id + 1,
                'actual_token': token,
                'predictions': position_results
            })

        results.append({
            'sequence_id': seq_id + 1,
            'sequence': sequences[seq_id],
            'length': seq_length,
            'positions': seq_results
        })

    return results

def calculate_prediction_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate prediction accuracy and perplexity metrics."""
    total_positions = 0
    correct_predictions = 0
    total_perplexity = 0.0

    sequence_metrics = []

    for seq_result in results:
        seq_correct = 0
        seq_positions = len(seq_result['positions'])
        seq_perplexity = 0.0

        for pos_result in seq_result['positions']:
            total_positions += 1

            # Check if top prediction is correct
            top_prediction = pos_result['predictions'][0]
            if top_prediction['is_correct']:
                correct_predictions += 1
                seq_correct += 1

            # Calculate perplexity contribution
            actual_prob = 0.0
            for pred in pos_result['predictions']:
                if pred['is_correct']:
                    actual_prob = pred['probability']
                    break

            if actual_prob > 0:
                seq_perplexity -= np.log(actual_prob)

        seq_accuracy = seq_correct / seq_positions if seq_positions > 0 else 0.0
        seq_perplexity = np.exp(seq_perplexity / seq_positions) if seq_positions > 0 else float('inf')

        total_perplexity += seq_perplexity / len(results)

        sequence_metrics.append({
            'sequence_id': seq_result['sequence_id'],
            'accuracy': seq_accuracy,
            'correct': seq_correct,
            'total': seq_positions,
            'perplexity': seq_perplexity
        })

    overall_accuracy = correct_predictions / total_positions if total_positions > 0 else 0.0

    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': correct_predictions,
        'total_positions': total_positions,
        'average_perplexity': total_perplexity,
        'sequence_metrics': sequence_metrics
    }

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_nucleotide_prediction(
    sequences: Optional[List[str]] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict nucleotide probabilities at each position in DNA sequences.

    Args:
        sequences: List of DNA sequences to process
        input_file: Path to file containing sequences (one per line)
        output_file: Path to save predictions (CSV format)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - probabilities: Raw probability distributions
            - predictions: Detailed prediction results
            - metrics: Accuracy and perplexity metrics
            - sequences: Input sequences
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_nucleotide_prediction(sequences=["ATCG"], output_file="pred.csv")
        >>> print(f"Accuracy: {result['metrics']['overall_accuracy']:.3f}")
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
    parameters, forward_fn, tokenizer, model_config = load_nucleotide_model_prediction(
        config["model"], config["max_positions"]
    )

    # Tokenize sequences
    tokens, tokens_str = tokenize_dna_sequences(sequences, tokenizer)

    # Predict probabilities
    probabilities, logits = predict_nucleotide_probabilities(
        parameters, forward_fn, tokens
    )

    # Analyze predictions
    prediction_results = analyze_nucleotide_predictions(
        probabilities, tokens_str, sequences, tokenizer, config["top_k"]
    )

    # Calculate metrics
    metrics = calculate_prediction_metrics(prediction_results)

    # Prepare results
    result = {
        "probabilities": np.array(probabilities),
        "logits": np.array(logits),
        "predictions": prediction_results,
        "metrics": metrics,
        "sequences": sequences,
        "metadata": {
            "config": config,
            "model_config": {
                "alphabet_size": model_config.alphabet_size,
                "model_name": config["model"]
            },
            "probability_shape": probabilities.shape,
            "num_sequences": len(sequences),
            "top_k": config["top_k"]
        },
        "output_file": None
    }

    # Save output if requested
    if output_file:
        output_path = Path(output_file)
        save_predictions_csv(prediction_results, metrics, output_path)
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
    parser.add_argument('--output', '-o', help='Output file path (CSV format)')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--model', choices=AVAILABLE_MODELS, help='Model to use')
    parser.add_argument('--top-k', type=int, help='Number of top predictions to show')
    parser.add_argument('--max-positions', type=int, help='Maximum sequence positions')

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
    for arg, value in [('model', args.model), ('top_k', args.top_k),
                       ('max_positions', args.max_positions)]:
        if value is not None:
            overrides[arg] = value

    try:
        # Run nucleotide prediction
        result = run_nucleotide_prediction(
            sequences=args.sequences,
            input_file=args.input,
            output_file=args.output,
            config=config,
            **overrides
        )

        print(f"✅ Success: Processed {result['metadata']['num_sequences']} sequences")
        print(f"   Probability shape: {result['metadata']['probability_shape']}")
        print(f"   Overall accuracy: {result['metrics']['overall_accuracy']:.3f}")
        print(f"   Average perplexity: {result['metrics']['average_perplexity']:.3f}")

        for seq_metric in result['metrics']['sequence_metrics']:
            print(f"   Sequence {seq_metric['sequence_id']}: {seq_metric['accuracy']:.3f} accuracy")

        if result['output_file']:
            print(f"   Saved to: {result['output_file']}")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == '__main__':
    main()