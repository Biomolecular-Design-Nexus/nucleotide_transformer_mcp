# MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported
2. **Self-Contained**: Functions inlined where possible
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts

| Script | Description | Repo Dependent | Config |
|--------|-------------|----------------|--------|
| `dna_embedding.py` | Extract DNA sequence embeddings | No* | `configs/dna_embedding_config.json` |
| `attention_visualization.py` | Visualize attention maps | No* | `configs/attention_visualization_config.json` |
| `nucleotide_prediction.py` | Predict nucleotide probabilities | No* | `configs/nucleotide_prediction_config.json` |
| `dna_embedding_v2.py` | Embeddings with shared library | No* | `configs/dna_embedding_config.json` |

*All scripts require the `nucleotide-transformer` package, which is the core ML functionality that cannot be inlined.

## Usage

### Environment Setup
```bash
# Activate environment (prefer mamba over conda)
mamba activate ./env  # or: conda activate ./env
```

### Basic Usage
```bash
# Run with direct sequences
python scripts/dna_embedding.py --sequences "ATCGATCGATCG" --output results/embeddings.npz

# Run with input file
python scripts/dna_embedding.py --input examples/data/sample_sequences.txt --output results/embeddings.npz

# Run with config file
python scripts/dna_embedding.py --config configs/dna_embedding_config.json --output results/embeddings.npz
```

### Available Commands

#### DNA Embedding Extraction
```bash
python scripts/dna_embedding.py \
    --sequences "ATTCCGAAATCGCTGACCGATCGTACGAAA" "ATTTCTCTCTCTCTCT" \
    --model 100M_multi_species_v2 \
    --layer 15 \
    --output results/embeddings.npz
```

#### Attention Visualization
```bash
python scripts/attention_visualization.py \
    --sequences "ATTCCGAAATCGCTGACCGATCGTACGAAA" \
    --layer 1 \
    --head 4 \
    --output results/attention.png
```

#### Nucleotide Prediction
```bash
python scripts/nucleotide_prediction.py \
    --sequences "ATTCCGAAATCGCTGACCGATCGTACGAAA" \
    --top-k 5 \
    --output results/predictions.csv
```

### Model Options

Available models (from small to large):
- `50M_multi_species_v2` (default, fastest)
- `100M_multi_species_v2`
- `250M_multi_species_v2`
- `500M_multi_species_v2`
- `2B5_multi_species` (largest, slowest)

### Input Formats

#### Direct Sequences
```bash
--sequences "ATCGATCG" "GGCCTTAA" "AAATTTGGG"
```

#### Input Files
Text files with one sequence per line (comments with # are ignored):
```
# Sample sequences for testing
ATTCCGAAATCGCTGACCGATCGTACGAAA
ATTTCTCTCTCTCTCTGAGATCGATCGATCGATATCTCTCGAGCTAGC
ATGAAACGCTACGGTCGCTACGGCAAACGCTAG
```

#### Configuration Files
JSON files with parameters:
```json
{
  "model": {"name": "50M_multi_species_v2"},
  "embedding": {"layer": 12, "max_positions": 32},
  "sequences": {
    "default": ["ATTCCGAAATCGCTGACCGATCGTACGAAA"]
  }
}
```

### Output Formats

#### NPZ Files (Embeddings)
```python
import numpy as np
data = np.load("results/embeddings.npz")
embeddings = data['embeddings']  # Shape: (N, 512)
sequences = data['sequences']    # Original sequences
```

#### PNG Files (Attention Maps)
Heatmap visualizations showing attention patterns between sequence positions.

#### CSV Files (Predictions)
```csv
sequence_id,sequence,position,actual_token,predicted_token_1,probability_1,predicted_token_2,probability_2,predicted_token_3,probability_3,predicted_token_4,probability_4,predicted_token_5,probability_5,is_correct
1,ATCGATCG,1,ATTCCA,ATTCCA,0.8234,ATTCCG,0.0923,ATTACT,0.0234,ATTAGG,0.0156,ATTCAG,0.0098,True
```

## Shared Library

Common functions are in `scripts/lib/`:

### lib/io.py
- `load_sequences_from_file()`: Load sequences from text files
- `save_embeddings_npz()`: Save embeddings to NPZ format
- `save_predictions_csv()`: Save predictions to CSV
- `load_json_config()`: Load and parse configuration files

### lib/models.py
- `validate_model_name()`: Check model availability
- `tokenize_dna_sequences()`: Convert sequences to tokens
- `load_nucleotide_model_for_*()`: Load models for different tasks

### lib/visualization.py
- `save_attention_plot()`: Save matplotlib figures
- `create_attention_visualization()`: Create attention heatmaps

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped:

```python
from scripts.dna_embedding import run_dna_embedding

@mcp.tool()
def extract_dna_embeddings(
    sequences: List[str],
    output_file: str = None,
    model: str = "50M_multi_species_v2"
):
    """Extract DNA sequence embeddings using Nucleotide Transformer."""
    return run_dna_embedding(
        sequences=sequences,
        output_file=output_file,
        model=model
    )
```

### Function Signatures

```python
# DNA Embedding
run_dna_embedding(
    sequences: Optional[List[str]] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]

# Attention Visualization
run_attention_visualization(
    sequences: Optional[List[str]] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]

# Nucleotide Prediction
run_nucleotide_prediction(
    sequences: Optional[List[str]] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]
```

### Return Format

All functions return a consistent dictionary:
```python
{
    "result": np.ndarray,           # Main computation result
    "sequences": List[str],         # Input sequences
    "metadata": {                   # Execution metadata
        "config": dict,
        "model_config": dict,
        "num_sequences": int
    },
    "output_file": Optional[str]    # Saved file path
}
```

## Example Usage

### Command Line
```bash
# Extract embeddings from sample data
python scripts/dna_embedding.py \
    --input examples/data/sample_sequences.txt \
    --config configs/dna_embedding_config.json \
    --output results/sample_embeddings.npz

# Visualize attention with custom parameters
python scripts/attention_visualization.py \
    --input examples/data/sample_sequences.txt \
    --layer 2 --head 1 --dpi 600 \
    --output results/attention_l2h1.png

# Predict nucleotides with different model
python scripts/nucleotide_prediction.py \
    --sequences "ATGAAACGCTACGGTCGC" \
    --model 100M_multi_species_v2 \
    --top-k 3 \
    --output results/predictions.csv
```

### Python API
```python
from scripts.dna_embedding import run_dna_embedding
from scripts.lib.io import load_json_config

# Load configuration
config = load_json_config("configs/dna_embedding_config.json")

# Extract embeddings
result = run_dna_embedding(
    sequences=["ATCGATCGATCG", "GGCCTTAA"],
    output_file="results/my_embeddings.npz",
    config=config
)

print(f"Processed {result['metadata']['num_sequences']} sequences")
print(f"Embedding shape: {result['mean_embeddings'].shape}")
```

## Troubleshooting

### Model Download Issues
Models are downloaded automatically on first use. Ensure internet connectivity and sufficient disk space (~1GB per model).

### Memory Issues
Use smaller models (50M) for large sequences or reduce `max_positions` parameter.

### CUDA Warnings
JAX will use CPU if CUDA is not available. This is normal and doesn't affect functionality, only performance.

### File Path Issues
Use absolute paths or ensure working directory is correct. Config files use relative paths from the script location.