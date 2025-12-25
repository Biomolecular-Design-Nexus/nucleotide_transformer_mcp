# Step 6: MCP Tools Documentation

## Server Information
- **Server Name**: nucleotide-transformer
- **Version**: 1.0.0
- **Created Date**: 2025-12-24
- **Server Path**: `src/server.py`
- **Total Tools**: 14

## Job Management Tools

| Tool | Description | Return Type |
|------|-------------|-------------|
| `get_job_status` | Check job progress and status | Dict with status, timestamps |
| `get_job_result` | Get completed job results | Dict with results or error |
| `get_job_log` | View job execution logs | Dict with log lines |
| `cancel_job` | Cancel running job | Success/error message |
| `list_jobs` | List all jobs (optionally filtered) | List of job summaries |

## Sync Tools (Fast Operations < 10 min)

| Tool | Description | Source Script | Est. Runtime | Max Sequences |
|------|-------------|---------------|--------------|---------------|
| `extract_dna_embeddings` | Extract DNA sequence embeddings | `scripts/dna_embedding.py` | ~30 sec - 2 min | ~1000 |
| `extract_dna_embeddings_v2` | Enhanced embeddings with shared library | `scripts/dna_embedding_v2.py` | ~30 sec - 2 min | ~1000 |
| `visualize_attention_patterns` | Create attention heatmaps | `scripts/attention_visualization.py` | ~1-3 min | ~50 |
| `predict_nucleotide_probabilities` | Predict nucleotide probabilities | `scripts/nucleotide_prediction.py` | ~30 sec - 2 min | ~100 |

### Tool Details

#### extract_dna_embeddings
- **Description**: Extract DNA sequence embeddings using Nucleotide Transformer
- **Source Script**: `scripts/dna_embedding.py`
- **Estimated Runtime**: ~30 seconds to 2 minutes
- **Suitable For**: Standard embedding extraction, small to medium datasets

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sequences | List[str] | No* | None | List of DNA sequences to process |
| input_file | str | No* | None | Path to file with sequences (one per line) |
| output_file | str | No | None | Path to save embeddings as NPZ |
| model | str | No | "50M_multi_species_v2" | Model name to use |
| layer | int | No | 12 | Layer to extract embeddings from |
| max_positions | int | No | 32 | Maximum sequence length |

*Either `sequences` or `input_file` must be provided.

**Example:**
```
Use extract_dna_embeddings with sequences ["ATCGATCGATCG", "GGCCTTAA"] and output_file "embeddings.npz"
```

**Returns:**
```json
{
  "status": "success",
  "embeddings_shape": [2, 512],
  "num_sequences": 2,
  "output_file": "embeddings.npz",
  "metadata": {
    "config": {...},
    "model_config": {...},
    "num_sequences": 2
  }
}
```

#### extract_dna_embeddings_v2
- **Description**: Enhanced version with shared library and better memory management
- **Source Script**: `scripts/dna_embedding_v2.py`
- **Estimated Runtime**: ~30 seconds to 2 minutes
- **Suitable For**: Improved performance and memory usage

**Parameters:** Same as `extract_dna_embeddings`

#### visualize_attention_patterns
- **Description**: Create attention pattern visualizations for DNA sequences
- **Source Script**: `scripts/attention_visualization.py`
- **Estimated Runtime**: ~1-3 minutes
- **Suitable For**: Understanding model attention patterns, research visualization

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sequences | List[str] | No* | None | List of DNA sequences to visualize |
| input_file | str | No* | None | Path to file with sequences |
| output_file | str | No | None | Path to save PNG visualization |
| model | str | No | "50M_multi_species_v2" | Model name to use |
| layer | int | No | 1 | Attention layer to visualize |
| head | int | No | 4 | Attention head to visualize |
| dpi | int | No | 300 | Image resolution |

**Example:**
```
Use visualize_attention_patterns with sequences ["ATCGATCGATCG"] and output_file "attention.png"
```

#### predict_nucleotide_probabilities
- **Description**: Predict nucleotide probabilities at each sequence position
- **Source Script**: `scripts/nucleotide_prediction.py`
- **Estimated Runtime**: ~30 seconds to 2 minutes
- **Suitable For**: Sequence analysis, mutation effect prediction

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sequences | List[str] | No* | None | List of DNA sequences to analyze |
| input_file | str | No* | None | Path to file with sequences |
| output_file | str | No | None | Path to save CSV predictions |
| model | str | No | "50M_multi_species_v2" | Model name to use |
| top_k | int | No | 5 | Number of top predictions per position |
| max_positions | int | No | 32 | Maximum sequence length |

**Example:**
```
Use predict_nucleotide_probabilities with sequences ["ATCGATCGATCG"] and top_k 3
```

---

## Submit Tools (Long Operations > 10 min)

| Tool | Description | Source Script | Est. Runtime | Batch Support |
|------|-------------|---------------|--------------|---------------|
| `submit_dna_embeddings` | Async DNA embeddings extraction | `scripts/dna_embedding.py` | >10 min | ✅ Yes |
| `submit_dna_embeddings_v2` | Async enhanced embeddings | `scripts/dna_embedding_v2.py` | >10 min | ✅ Yes |
| `submit_attention_visualization` | Async attention visualization | `scripts/attention_visualization.py` | >10 min | ✅ Yes |
| `submit_nucleotide_prediction` | Async nucleotide prediction | `scripts/nucleotide_prediction.py` | >10 min | ✅ Yes |
| `submit_batch_dna_analysis` | Batch processing multiple files | Multiple scripts | >10 min | ✅ Yes |

### Tool Details

#### submit_dna_embeddings
- **Description**: Submit DNA embedding extraction for background processing
- **Source Script**: `scripts/dna_embedding.py`
- **Estimated Runtime**: >10 minutes for large datasets
- **Suitable For**: Thousands of sequences, large files

**Parameters:** Same as sync version plus:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| output_dir | str | No | None | Directory to save outputs |
| job_name | str | No | auto | Custom job name for tracking |

**Example:**
```
Submit DNA embeddings for examples/data/sample_sequences.txt with job_name "large_dataset_embeddings"
```

**Returns:**
```json
{
  "status": "submitted",
  "job_id": "abc123de",
  "message": "Job submitted. Use get_job_status('abc123de') to check progress."
}
```

#### submit_batch_dna_analysis
- **Description**: Submit batch processing for multiple input files
- **Source Scripts**: Multiple (based on analysis_type)
- **Estimated Runtime**: >10 minutes for batch processing
- **Suitable For**: Processing many files with same parameters

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| input_files | List[str] | Yes | - | List of input file paths |
| analysis_type | str | No | "embeddings" | "embeddings", "attention", or "prediction" |
| model | str | No | "50M_multi_species_v2" | Model name for all files |
| output_dir | str | No | None | Directory to save all outputs |
| job_name | str | No | auto | Custom job name |

**Example:**
```
Submit batch analysis for ["file1.txt", "file2.txt"] with analysis_type "embeddings"
```

---

## Workflow Examples

### Quick Analysis (Sync)
```
1. Use extract_dna_embeddings with sequences ["ATCGATCGATCG", "GGCCTTAA"]
   → Returns results immediately with embeddings shape [2, 512]

2. Use visualize_attention_patterns with sequences ["ATCGATCGATCG"]
   → Returns PNG file path with attention heatmap

3. Use predict_nucleotide_probabilities with sequences ["ATCGATCGATCG"]
   → Returns CSV with top-k predictions per position
```

### Long-Running Task (Submit API)
```
1. Submit: submit_dna_embeddings with input_file "large_dataset.txt" and job_name "analysis_dec24"
   → Returns: {"job_id": "abc123de", "status": "submitted"}

2. Check: get_job_status with job_id "abc123de"
   → Returns: {"status": "running", "started_at": "2025-12-24T10:30:00", ...}

3. Monitor: get_job_log with job_id "abc123de" and tail 20
   → Returns: {"log_lines": ["Processing sequence 1500/3000...", ...]}

4. Result: get_job_result with job_id "abc123de"
   → Returns: {"status": "success", "result": {...}}
```

### Batch Processing
```
1. Prepare multiple files: ["promoters.txt", "genes.txt", "intergenic.txt"]

2. Submit: submit_batch_dna_analysis with:
   - input_files: ["promoters.txt", "genes.txt", "intergenic.txt"]
   - analysis_type: "embeddings"
   - output_dir: "batch_results/"

3. Track: Use job management tools to monitor progress

4. Results: Each file processed separately, all outputs in batch_results/
```

## Available Models

| Model Name | Size | Description | Speed | Accuracy |
|------------|------|-------------|-------|----------|
| `50M_multi_species_v2` | 50M params | Default, fastest | ⚡⚡⚡ | ⭐⭐⭐ |
| `100M_multi_species_v2` | 100M params | Balanced performance | ⚡⚡ | ⭐⭐⭐⭐ |
| `250M_multi_species_v2` | 250M params | Higher accuracy | ⚡ | ⭐⭐⭐⭐⭐ |
| `500M_multi_species_v2` | 500M params | High performance | ⚡ | ⭐⭐⭐⭐⭐ |
| `2B5_multi_species` | 2.5B params | Largest, slowest, most accurate | ⏳ | ⭐⭐⭐⭐⭐⭐ |

## Input Formats

### Sequences Parameter
```python
sequences = [
    "ATCGATCGATCG",
    "GGCCTTAAGGCC",
    "TTTTAAAAGGGG"
]
```

### Input Files
Text files with one sequence per line (comments with # ignored):
```
# Sample sequences for testing
ATTCCGAAATCGCTGACCGATCGTACGAAA
ATTTCTCTCTCTCTCTGAGATCGATCGATCGATATCTCTCGAGCTAGC
ATGAAACGCTACGGTCGCTACGGCAAACGCTAG
```

## Output Formats

### NPZ Files (Embeddings)
```python
import numpy as np
data = np.load("embeddings.npz")
embeddings = data['embeddings']  # Shape: (N, 512)
sequences = data['sequences']    # Original sequences
```

### PNG Files (Attention Maps)
Heatmap visualizations showing attention patterns between sequence positions.

### CSV Files (Predictions)
```csv
sequence_id,sequence,position,actual_token,predicted_token_1,probability_1,predicted_token_2,probability_2,predicted_token_3,probability_3
1,ATCGATCG,1,A,A,0.8234,T,0.0923,C,0.0234
```

## Error Handling

All tools return structured error responses:

```json
{
  "status": "error",
  "error": "Descriptive error message"
}
```

Common errors:
- **File not found**: Input file path doesn't exist
- **Invalid input**: Malformed DNA sequences
- **Model download**: Network issues during model download
- **Memory**: Insufficient memory for large datasets
- **Job not found**: Invalid job_id for job management

## Performance Guidelines

### Sync Tool Limits
- **Embeddings**: <1000 sequences
- **Attention**: <50 sequences (visualization intensive)
- **Prediction**: <100 sequences

### When to Use Submit API
- Datasets with >1000 sequences
- Processing multiple large files
- High-resolution visualizations
- Complex models (>250M parameters)
- Batch processing workflows

### Memory Optimization
- Use smaller models for large datasets
- Reduce `max_positions` for long sequences
- Process files in batches
- Use `submit_` tools for memory-intensive tasks