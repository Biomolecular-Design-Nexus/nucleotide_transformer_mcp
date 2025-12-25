# nucleotide-transformer MCP

> DNA sequence analysis using Nucleotide Transformer deep learning models via Model Context Protocol (MCP)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

This MCP server provides access to the Nucleotide Transformer, a state-of-the-art deep learning model for DNA sequence analysis. It offers both fast synchronous operations for small datasets and asynchronous background processing for large-scale analysis.

### Features
- **DNA Sequence Embeddings**: Extract 512-dimensional contextualized embeddings from DNA sequences
- **Attention Visualization**: Generate heatmaps showing model attention patterns across sequence positions
- **Nucleotide Prediction**: Predict nucleotide probabilities at each position for sequence analysis
- **Multiple Model Sizes**: Support for models ranging from 50M to 2.5B parameters
- **Batch Processing**: Handle multiple files and large datasets efficiently
- **Async Processing**: Background job execution for long-running tasks

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment
├── src/
│   └── server.py           # MCP server
├── scripts/
│   ├── dna_embedding.py           # DNA sequence embedding extraction
│   ├── dna_embedding_v2.py        # Enhanced embedding extraction with shared lib
│   ├── attention_visualization.py # Attention pattern visualization
│   ├── nucleotide_prediction.py   # Nucleotide probability prediction
│   └── lib/                       # Shared utilities
│       ├── io.py                  # File I/O utilities
│       ├── models.py              # Model loading utilities
│       └── visualization.py       # Visualization utilities
├── examples/
│   └── data/
│       └── sample_sequences.txt   # Demo DNA sequences
├── configs/                       # Configuration files
│   ├── dna_embedding_config.json
│   ├── attention_visualization_config.json
│   ├── nucleotide_prediction_config.json
│   └── default_config.json
└── repo/                          # Original nucleotide-transformer repository
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- At least 4GB RAM (16GB+ recommended for large models)
- Optional: CUDA-capable GPU for accelerated inference

### Create Environment

Please follow the procedure outlined in `reports/step3_environment.md` for detailed setup instructions. A quick setup workflow is shown below:

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/nucleotide_transformer_mcp

# Determine package manager (prefer mamba over conda)
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
else
    PKG_MGR="conda"
fi
echo "Using package manager: $PKG_MGR"

# Create conda environment
$PKG_MGR create -p ./env python=3.10 pip -y

# Activate environment
$PKG_MGR activate ./env
```

### Install Dependencies

```bash
# Core MCP dependencies
pip install fastmcp loguru click pandas numpy tqdm

# ML dependencies for nucleotide transformer
pip install "jax>=0.3.25" "jaxlib>=0.3.25" "dm-haiku>=0.0.9" \
    "transformers>=4.52.4" "torch>=2.7.1" "einops>=0.8.1" \
    "matplotlib>=3.5.0" "seaborn>=0.11.0" "pyfaidx>=0.7.0"

# Install nucleotide transformer package
cd repo/nucleotide-transformer
pip install -e .
cd ../..
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/dna_embedding.py` | Extract DNA sequence embeddings | See below |
| `scripts/dna_embedding_v2.py` | Enhanced embeddings with shared library | See below |
| `scripts/attention_visualization.py` | Visualize attention maps | See below |
| `scripts/nucleotide_prediction.py` | Predict nucleotide probabilities | See below |

### Script Examples

#### DNA Embedding Extraction

```bash
# Activate environment
mamba activate ./env

# Extract embeddings from sequences
python scripts/dna_embedding.py \
  --sequences "ATTCCGAAATCGCTGACCGATCGTACGAAA" \
  --model 50M_multi_species_v2 \
  --layer 12 \
  --output results/embeddings.npz

# Extract embeddings from file
python scripts/dna_embedding.py \
  --input examples/data/sample_sequences.txt \
  --output results/embeddings.npz

# Use configuration file
python scripts/dna_embedding.py \
  --config configs/dna_embedding_config.json \
  --output results/embeddings.npz
```

**Parameters:**
- `--sequences, -s`: DNA sequences as strings (required if no input file)
- `--input, -i`: Path to file with sequences (one per line) (required if no sequences)
- `--output, -o`: Output NPZ file path (default: stdout)
- `--model, -m`: Model name (default: 50M_multi_species_v2)
- `--layer, -l`: Layer to extract embeddings from (default: 12)
- `--max-positions`: Maximum sequence length (default: 32)
- `--config, -c`: JSON configuration file (optional)

#### Attention Visualization

```bash
# Create attention heatmap
python scripts/attention_visualization.py \
  --sequences "ATTCCGAAATCGCTGACCGATCGTACGAAA" \
  --layer 1 \
  --head 4 \
  --output results/attention.png

# Process file with custom settings
python scripts/attention_visualization.py \
  --input examples/data/sample_sequences.txt \
  --layer 2 \
  --head 1 \
  --dpi 300 \
  --output results/attention_heatmap.png
```

**Parameters:**
- `--sequences, -s`: DNA sequences to visualize
- `--input, -i`: Path to file with sequences
- `--output, -o`: Output PNG file path
- `--layer, -l`: Attention layer to visualize (default: 1)
- `--head`: Attention head number (default: 4)
- `--dpi`: Image resolution (default: 300)

#### Nucleotide Prediction

```bash
# Predict nucleotide probabilities
python scripts/nucleotide_prediction.py \
  --sequences "ATTCCGAAATCGCTGACCGATCGTACGAAA" \
  --top-k 5 \
  --output results/predictions.csv

# Analyze multiple sequences
python scripts/nucleotide_prediction.py \
  --input examples/data/sample_sequences.txt \
  --top-k 3 \
  --output results/detailed_predictions.csv
```

**Parameters:**
- `--sequences, -s`: DNA sequences to analyze
- `--input, -i`: Path to file with sequences
- `--output, -o`: Output CSV file path
- `--top-k`: Number of top predictions per position (default: 5)
- `--max-positions`: Maximum sequence length (default: 32)

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name nucleotide-transformer
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add nucleotide-transformer -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "nucleotide-transformer": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/nucleotide_transformer_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/nucleotide_transformer_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from nucleotide-transformer?
```

#### Basic Sequence Analysis
```
Use extract_dna_embeddings with sequences ["ATTCCGAAATCGCTGACCGATCGTACGAAA", "ATGAAACGCTACGGTCGCTACGGCAAACGCTAG"] and save to embeddings.npz
```

#### File-Based Analysis
```
Extract DNA embeddings from @examples/data/sample_sequences.txt and save results to embeddings.npz
```

#### Attention Visualization
```
Create attention visualization for sequence "ATTCCGAAATCGCTGACCGATCGTACGAAA" and save to attention.png
```

#### Nucleotide Prediction
```
Predict nucleotide probabilities for sequences in @examples/data/sample_sequences.txt with top_k 3
```

#### Long-Running Tasks (Submit API)
```
Submit DNA embedding extraction for @examples/data/sample_sequences.txt with job_name "sample_analysis"
Then check the job status and get results when completed
```

#### Batch Processing
```
Submit batch DNA analysis for files:
- @examples/data/sample_sequences.txt
with analysis_type "embeddings" and output_dir "batch_results/"
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/sample_sequences.txt` | Reference the sample DNA sequences |
| `@configs/dna_embedding_config.json` | Reference configuration file |
| `@results/` | Reference output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "nucleotide-transformer": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/nucleotide_transformer_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/nucleotide_transformer_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same syntax as Claude Code)
> What tools are available from nucleotide-transformer?
> Extract embeddings from sequence "ATGAAACGCTACGGTCGC" using model 100M_multi_species_v2
> Create attention visualization for examples/data/sample_sequences.txt
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters | Max Input |
|------|-------------|------------|-----------|
| `extract_dna_embeddings` | Extract DNA sequence embeddings | sequences, input_file, output_file, model, layer, max_positions | ~1000 sequences |
| `extract_dna_embeddings_v2` | Enhanced embeddings with shared library | Same as above | ~1000 sequences |
| `visualize_attention_patterns` | Create attention heatmaps | sequences, input_file, output_file, model, layer, head, dpi | ~50 sequences |
| `predict_nucleotide_probabilities` | Predict nucleotide probabilities | sequences, input_file, output_file, model, top_k, max_positions | ~100 sequences |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_dna_embeddings` | Async DNA embeddings extraction | Same as sync + output_dir, job_name |
| `submit_dna_embeddings_v2` | Async enhanced embeddings | Same as sync + output_dir, job_name |
| `submit_attention_visualization` | Async attention visualization | Same as sync + output_dir, job_name |
| `submit_nucleotide_prediction` | Async nucleotide prediction | Same as sync + output_dir, job_name |
| `submit_batch_dna_analysis` | Batch processing multiple files | input_files, analysis_type, model, output_dir, job_name |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress and status |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs (optionally filtered by status) |

---

## Examples

### Example 1: DNA Sequence Embedding

**Goal:** Extract 512-dimensional embeddings from DNA sequences for downstream analysis

**Using Script:**
```bash
python scripts/dna_embedding.py \
  --input examples/data/sample_sequences.txt \
  --model 50M_multi_species_v2 \
  --layer 12 \
  --output results/sample_embeddings.npz
```

**Using MCP (in Claude Code):**
```
Extract DNA embeddings from @examples/data/sample_sequences.txt using model 50M_multi_species_v2 and layer 12, save to results/sample_embeddings.npz
```

**Expected Output:**
- NPZ file containing embeddings array (N x 512)
- Metadata with model configuration and processing statistics
- Console output showing successful processing of sequences

### Example 2: Attention Pattern Visualization

**Goal:** Understand how the model focuses on different parts of DNA sequences

**Using Script:**
```bash
python scripts/attention_visualization.py \
  --input examples/data/sample_sequences.txt \
  --layer 1 \
  --head 4 \
  --output results/attention_patterns.png
```

**Using MCP (in Claude Code):**
```
Create attention visualization for sequences in @examples/data/sample_sequences.txt using layer 1 and head 4, save to results/attention_patterns.png
```

**Expected Output:**
- PNG file with attention heatmap showing inter-position attention weights
- Statistical analysis of attention patterns
- Color-coded visualization highlighting important sequence regions

### Example 3: Nucleotide Probability Prediction

**Goal:** Predict the most likely nucleotides at each position for sequence analysis

**Using Script:**
```bash
python scripts/nucleotide_prediction.py \
  --input examples/data/sample_sequences.txt \
  --top-k 5 \
  --output results/nucleotide_predictions.csv
```

**Using MCP (in Claude Code):**
```
Predict nucleotide probabilities for @examples/data/sample_sequences.txt with top_k 5 and save to results/nucleotide_predictions.csv
```

**Expected Output:**
- CSV file with position-wise predictions and probabilities
- Top-k most likely nucleotides for each position
- Accuracy metrics and perplexity scores

### Example 4: Batch Processing

**Goal:** Process multiple files at once efficiently

**Using Script:**
```bash
# Create multiple input files
for analysis in promoters genes intergenic; do
  echo "Processing $analysis sequences..."
  python scripts/dna_embedding.py \
    --input "data/${analysis}_sequences.txt" \
    --output "results/${analysis}_embeddings.npz"
done
```

**Using MCP (in Claude Code):**
```
Submit batch DNA analysis for files ["data/promoters_sequences.txt", "data/genes_sequences.txt", "data/intergenic_sequences.txt"] with analysis_type "embeddings" and output_dir "batch_results/"
```

### Example 5: Long-Running Job Workflow

**Goal:** Process large dataset using async job system

**Using MCP (in Claude Code):**
```
1. Submit DNA embeddings job:
   Submit DNA embeddings for @large_dataset.txt with job_name "genome_analysis_2025" and model 250M_multi_species_v2

2. Monitor progress:
   Check status of job "abc123de"

3. View logs:
   Get job log for job "abc123de" with tail 50

4. Retrieve results when completed:
   Get result for job "abc123de"
```

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With | Sequences |
|------|-------------|----------|-----------|
| `sample_sequences.txt` | Diverse DNA sequences covering various genomic contexts | All tools | 6 sequences |

### Sample Data Contents:
- **Short promoter-like sequence**: `ATTCCGAAATCGCTGACCGATCGTACGAAA`
- **Repetitive elements**: `ATTTCTCTCTCTCTCTGAGATCGATCGATCGATATCTCTCGAGCTAGC`
- **Gene-like coding sequence**: `ATGAAACGCTACGGTCGCTACGGCAAACGCTACGGTCGCTACGGCAAACGCTAG`
- **Regulatory sequence with TATA box**: `CCCGCGGTATATATAAAGCCGCGCTCGCGTCGCGTCGCGAAA`
- **Random sequence baseline**: `ACGTACGTACGTACGTACGTACGTACGTACGT`
- **Long complex sequence**: Mixed regulatory and coding elements

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `dna_embedding_config.json` | DNA embedding extraction settings | model, layer, max_positions, output format |
| `attention_visualization_config.json` | Attention visualization settings | model, layer, head, visualization options |
| `nucleotide_prediction_config.json` | Nucleotide prediction settings | model, top_k, analysis options |
| `default_config.json` | Base configuration with available models | Common settings and model list |

### Config Example (dna_embedding_config.json)

```json
{
  "model": {
    "name": "50M_multi_species_v2",
    "alternatives": ["100M_multi_species_v2", "250M_multi_species_v2"]
  },
  "embedding": {
    "layer": 12,
    "max_positions": 32
  },
  "sequences": {
    "default": ["ATTCCGAAATCGCTGACCGATCGTACGAAA"]
  },
  "output": {
    "format": "npz",
    "include_token_embeddings": true,
    "include_mean_embeddings": true
  }
}
```

---

## Available Models

| Model Name | Size | Description | Speed | Accuracy | Memory |
|------------|------|-------------|-------|----------|--------|
| `50M_multi_species_v2` | 50M params | Default, fastest, good for testing | ⚡⚡⚡ | ⭐⭐⭐ | ~1GB |
| `100M_multi_species_v2` | 100M params | Balanced performance and accuracy | ⚡⚡ | ⭐⭐⭐⭐ | ~2GB |
| `250M_multi_species_v2` | 250M params | Higher accuracy for research use | ⚡ | ⭐⭐⭐⭐⭐ | ~4GB |
| `500M_multi_species_v2` | 500M params | High performance for production | ⚡ | ⭐⭐⭐⭐⭐ | ~8GB |
| `2B5_multi_species` | 2.5B params | Largest, most accurate, research-grade | ⏳ | ⭐⭐⭐⭐⭐⭐ | ~16GB |

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.10 pip -y
mamba activate ./env
pip install fastmcp loguru jax dm-haiku transformers torch matplotlib
cd repo/nucleotide-transformer && pip install -e . && cd ../..
```

**Problem:** Import errors
```bash
# Verify installation
python -c "from nucleotide_transformer import get_pretrained_model; print('Success')"
python -c "from src.server import mcp; print('MCP server OK')"
```

**Problem:** CUDA/GPU issues
```bash
# Check JAX device
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# Force CPU mode if GPU unavailable
export JAX_PLATFORM_NAME=cpu
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove nucleotide-transformer
claude mcp add nucleotide-transformer -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify connection
claude mcp health
```

**Problem:** Tools not working
```bash
# Test server directly
python -c "
from src.server import mcp
print('Available tools:')
for name in mcp.list_tools():
    print(f'  - {name}')
"

# Test a specific tool
python -c "
from src.server import mcp
result = mcp.get_tool('extract_dna_embeddings')
print(f'Tool found: {result is not None}')
"
```

### Model and Memory Issues

**Problem:** Model download fails
```bash
# Check internet connection and clear cache
rm -rf ~/.cache/huggingface/
python -c "from huggingface_hub import snapshot_download; snapshot_download('InstaDeepAI/nucleotide-transformer-2.5b-multi-species')"
```

**Problem:** Out of memory
```bash
# Use smaller model
# In your prompts, specify: model 50M_multi_species_v2

# Reduce sequence length
# Add parameter: max_positions 16

# Check system memory
free -h
```

**Problem:** Slow performance
```bash
# Enable GPU if available
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Use CPU optimization
export JAX_ENABLE_X64=False
export JAX_PLATFORMS=cpu
```

### Job Issues (Submit API)

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# View job details
python -c "
from src.jobs.manager import job_manager
print(job_manager.list_jobs())
"
```

**Problem:** Job failed
```
Use get_job_log with job_id "<job_id>" and tail 100 to see error details in Claude Code
```

**Problem:** Job completion detection (known issue)**
```bash
# Check if job actually completed by examining output files
ls -la jobs/<job_id>/
cat jobs/<job_id>/result.json

# Manual verification that job finished successfully
cat jobs/<job_id>/job.log | tail -20
```

### Performance Optimization

**Memory Management:**
- Use smaller models (`50M_multi_species_v2`) for large datasets
- Reduce `max_positions` parameter for memory constraints
- Process files in smaller batches
- Use submit API for memory-intensive tasks

**Speed Optimization:**
- Enable GPU acceleration if available
- Cache models after first download (~30 second delay on first run)
- Use appropriate model size for your accuracy requirements
- Consider batch processing for multiple files

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test scripts directly
python scripts/dna_embedding.py --sequences "ATCGATCG" --output test_output.npz

# Test MCP server
python src/server.py &
# In another terminal:
curl -X POST http://localhost:8000/tools -d '{"name": "list_tools"}'
```

### Starting Dev Server

```bash
# Run MCP server in development mode
fastmcp dev src/server.py

# Monitor logs
tail -f logs/server.log
```

---

## Performance Notes

### Model Download Behavior
- **First Run**: Downloads models automatically (~100MB - 10GB depending on model)
- **Download Time**: 30 seconds to several minutes depending on model size and connection
- **Subsequent Runs**: Uses cached models for fast startup
- **Storage**: Models cached in `~/.cache/huggingface/hub/`

### Execution Times (CPU)
- **50M model**: ~15-30 seconds per sequence
- **250M model**: ~30-60 seconds per sequence
- **2B5 model**: ~1-3 minutes per sequence
- **Note**: GPU acceleration can provide 5-10x speedup

### Memory Requirements
- **50M model**: ~1-2GB RAM
- **250M model**: ~4-6GB RAM
- **2B5 model**: ~16-20GB RAM
- **Recommendations**: Ensure 2x model memory available for stable operation

---

## License

Based on the Nucleotide Transformer project by InstaDeep AI.

## Credits

Based on [nucleotide-transformer](https://github.com/instadeepai/nucleotide-transformer) by InstaDeep AI