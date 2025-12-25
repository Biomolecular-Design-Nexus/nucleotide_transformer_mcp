#!/usr/bin/env python3
"""
Use Case 3: Nucleotide Probability Prediction

This script demonstrates how to predict nucleotide probabilities at each position
using the Nucleotide Transformer model. These probabilities can be used for
sequence reconstruction, perplexity calculation, and mutation impact prediction.

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
    parser = argparse.ArgumentParser(description="Predict nucleotide probabilities from DNA sequences")
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
        help="Model to use for prediction"
    )
    parser.add_argument(
        "--output",
        help="Output file to save predictions (CSV format)"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=32,
        help="Maximum sequence positions"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show"
    )
    return parser.parse_args()


def load_model(model_name, max_positions):
    """Load pretrained Nucleotide Transformer model."""
    print(f"Loading model: {model_name}")

    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(),
        attention_maps_to_save=(),
        max_positions=max_positions,
    )
    forward_fn = hk.transform(forward_fn)

    print(f"Model loaded with vocabulary size: {config.alphabet_size}")
    return parameters, forward_fn, tokenizer, config


def predict_probabilities(parameters, forward_fn, tokens):
    """Predict nucleotide probabilities for each position."""
    print("Predicting nucleotide probabilities...")

    # Initialize random key
    random_key = jax.random.PRNGKey(0)

    # Run inference
    outs = forward_fn.apply(parameters, random_key, tokens)

    # Get logits and convert to probabilities
    logits = outs["logits"]
    probabilities = jax.nn.softmax(logits, axis=-1)

    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probabilities.shape}")

    return probabilities, logits


def analyze_predictions(probabilities, tokens_str, sequences, tokenizer, top_k=5):
    """Analyze and display prediction results."""
    results = []

    for seq_id in range(len(sequences)):
        tokens = tokens_str[seq_id]
        seq_length = len([t for t in tokens[1:] if t != '<PAD>'])

        print(f"\n{'='*60}")
        print(f"SEQUENCE {seq_id + 1}: {sequences[seq_id]}")
        print(f"{'='*60}")

        # Get probabilities for this sequence (exclude CLS token and padding)
        seq_probs = probabilities[seq_id, 1:(seq_length + 1)]
        seq_tokens = tokens[1:(seq_length + 1)]

        seq_results = []

        for pos_id, (token, prob_dist) in enumerate(zip(seq_tokens, seq_probs)):
            print(f"\nPosition {pos_id + 1} - Actual token: {token}")

            # Get top-k predictions
            sorted_indices = jnp.argsort(-prob_dist)
            sorted_probs = prob_dist[sorted_indices]

            print("Top predictions:")
            position_results = []

            for k in range(min(top_k, len(sorted_indices))):
                predicted_token = tokenizer.id_to_token(int(sorted_indices[k]))
                probability = float(sorted_probs[k])

                is_correct = "✓" if predicted_token == token else "✗"
                print(f"  {k+1}. {predicted_token} - {probability:.4f} ({probability*100:.2f}%) {is_correct}")

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


def calculate_metrics(results):
    """Calculate various prediction metrics."""
    print(f"\n{'='*60}")
    print("PREDICTION METRICS")
    print(f"{'='*60}")

    total_positions = 0
    correct_predictions = 0
    total_perplexity = 0.0

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
            actual_prob = top_prediction['probability'] if top_prediction['is_correct'] else 0.0
            for pred in pos_result['predictions']:
                if pred['is_correct']:
                    actual_prob = pred['probability']
                    break

            if actual_prob > 0:
                seq_perplexity -= np.log(actual_prob)

        seq_accuracy = seq_correct / seq_positions if seq_positions > 0 else 0.0
        seq_perplexity = np.exp(seq_perplexity / seq_positions) if seq_positions > 0 else float('inf')

        total_perplexity += seq_perplexity / len(results)

        print(f"Sequence {seq_result['sequence_id']}:")
        print(f"  Accuracy: {seq_accuracy:.4f} ({seq_correct}/{seq_positions})")
        print(f"  Perplexity: {seq_perplexity:.4f}")

    overall_accuracy = correct_predictions / total_positions if total_positions > 0 else 0.0

    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_positions})")
    print(f"  Average Perplexity: {total_perplexity:.4f}")

    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': correct_predictions,
        'total_positions': total_positions,
        'average_perplexity': total_perplexity
    }


def save_results(results, metrics, output_file):
    """Save results to CSV file."""
    import csv

    with open(output_file, 'w', newline='') as csvfile:
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

    print(f"Results saved to: {output_file}")


def main():
    args = parse_args()

    try:
        # Load model
        parameters, forward_fn, tokenizer, config = load_model(args.model, args.max_positions)

        # Tokenize sequences
        print("Tokenizing sequences...")
        tokens_ids = [b[1] for b in tokenizer.batch_tokenize(args.sequences)]
        tokens_str = [b[0] for b in tokenizer.batch_tokenize(args.sequences)]
        tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

        # Predict probabilities
        probabilities, logits = predict_probabilities(parameters, forward_fn, tokens)

        # Analyze predictions
        results = analyze_predictions(
            probabilities, tokens_str, args.sequences, tokenizer, args.top_k
        )

        # Calculate metrics
        metrics = calculate_metrics(results)

        # Save results if requested
        if args.output:
            save_results(results, metrics, args.output)

        print(f"\n✓ Nucleotide prediction analysis completed successfully!")

        return {
            'probabilities': np.array(probabilities),
            'results': results,
            'metrics': metrics,
            'sequences': args.sequences,
            'config': config
        }

    except Exception as e:
        print(f"✗ Error during nucleotide prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()