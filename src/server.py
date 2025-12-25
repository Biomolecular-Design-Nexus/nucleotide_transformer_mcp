#!/usr/bin/env python3
"""MCP Server for nucleotide-transformer

Provides both synchronous and asynchronous (submit) APIs for DNA analysis tools.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import sys
import json

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("nucleotide-transformer")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def extract_dna_embeddings(
    sequences: Optional[List[str]] = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    model: str = "50M_multi_species_v2",
    layer: int = 12,
    max_positions: int = 32
) -> dict:
    """
    Extract DNA sequence embeddings using Nucleotide Transformer (fast operation for <1000 sequences).

    This is the main working tool from the extracted scripts. For large datasets
    or multiple files, use submit_dna_embeddings for async processing.

    Args:
        sequences: List of DNA sequences to process
        input_file: Path to file with DNA sequences (one per line)
        output_file: Optional output file path to save results
        model: Model name (default: 50M_multi_species_v2)
        layer: Layer to extract embeddings from (default: 12)
        max_positions: Maximum sequence length (default: 32)

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - embeddings_shape: Shape of extracted embeddings [N, 512]
        - num_sequences: Number of sequences processed
        - output_file: Path to saved file (if output_file provided)
        - metadata: Configuration used
        - error: Error message (if status is "error")

    Example:
        extract_dna_embeddings(sequences=["ATCGATCGATCG", "GGCCTTAA"], output_file="embeddings.npz")
    """
    # Import the script's main function
    try:
        from dna_embedding import run_dna_embedding

        result = run_dna_embedding(
            sequences=sequences,
            input_file=input_file,
            output_file=output_file,
            model=model,
            layer=layer,
            max_positions=max_positions
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"extract_dna_embeddings failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def extract_dna_embeddings_v2(
    sequences: Optional[List[str]] = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    model: str = "50M_multi_species_v2",
    layer: int = 12,
    max_positions: int = 32
) -> dict:
    """
    Extract DNA sequence embeddings using enhanced script with shared library.

    Enhanced version with better memory management and file format support.
    For large datasets, use submit_dna_embeddings_v2 for async processing.

    Args:
        sequences: List of DNA sequences to process
        input_file: Path to file with DNA sequences (one per line)
        output_file: Optional output file path to save results
        model: Model name (default: 50M_multi_species_v2)
        layer: Layer to extract embeddings from (default: 12)
        max_positions: Maximum sequence length (default: 32)

    Returns:
        Dictionary with results or error message
    """
    try:
        from dna_embedding_v2 import run_dna_embedding

        result = run_dna_embedding(
            sequences=sequences,
            input_file=input_file,
            output_file=output_file,
            model=model,
            layer=layer,
            max_positions=max_positions
        )
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"extract_dna_embeddings_v2 failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def visualize_attention_patterns(
    sequences: Optional[List[str]] = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    model: str = "50M_multi_species_v2",
    layer: int = 1,
    head: int = 4,
    dpi: int = 300
) -> dict:
    """
    Create attention pattern visualizations for DNA sequences (fast operation for <50 sequences).

    Generates heatmap visualizations showing how the model attends to different
    sequence positions. For large datasets, use submit_attention_visualization.

    Args:
        sequences: List of DNA sequences to visualize
        input_file: Path to file with DNA sequences (one per line)
        output_file: Path to save attention plot (PNG format)
        model: Model name (default: 50M_multi_species_v2)
        layer: Attention layer to visualize (default: 1)
        head: Attention head to visualize (default: 4)
        dpi: Image resolution (default: 300)

    Returns:
        Dictionary with visualization results and output file path
    """
    try:
        from attention_visualization import run_attention_visualization

        result = run_attention_visualization(
            sequences=sequences,
            input_file=input_file,
            output_file=output_file,
            model=model,
            layer=layer,
            head=head,
            dpi=dpi
        )
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"visualize_attention_patterns failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def predict_nucleotide_probabilities(
    sequences: Optional[List[str]] = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    model: str = "50M_multi_species_v2",
    top_k: int = 5,
    max_positions: int = 32
) -> dict:
    """
    Predict nucleotide probabilities at each sequence position (fast operation for <100 sequences).

    Analyzes each position in the DNA sequence and predicts the most likely
    nucleotides with their probabilities. For large datasets, use submit_nucleotide_prediction.

    Args:
        sequences: List of DNA sequences to analyze
        input_file: Path to file with DNA sequences (one per line)
        output_file: Path to save predictions as CSV
        model: Model name (default: 50M_multi_species_v2)
        top_k: Number of top predictions per position (default: 5)
        max_positions: Maximum sequence length (default: 32)

    Returns:
        Dictionary with prediction results and output file path
    """
    try:
        from nucleotide_prediction import run_nucleotide_prediction

        result = run_nucleotide_prediction(
            sequences=sequences,
            input_file=input_file,
            output_file=output_file,
            model=model,
            top_k=top_k,
            max_positions=max_positions
        )
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"predict_nucleotide_probabilities failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_dna_embeddings(
    sequences: Optional[List[str]] = None,
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    model: str = "50M_multi_species_v2",
    layer: int = 12,
    max_positions: int = 32,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit DNA embedding extraction for background processing (large datasets).

    This operation extracts embeddings for large datasets that take >10 minutes.
    Suitable for thousands of sequences or multiple large files.

    Args:
        sequences: List of DNA sequences to process
        input_file: Path to file with DNA sequences
        output_dir: Directory to save outputs
        model: Model name (default: 50M_multi_species_v2)
        layer: Layer to extract embeddings from (default: 12)
        max_positions: Maximum sequence length (default: 32)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(SCRIPTS_DIR / "dna_embedding.py")

    args = {
        "model": model,
        "layer": layer,
        "max_positions": max_positions
    }

    if sequences:
        args["sequences"] = sequences
    if input_file:
        args["input"] = input_file
    if output_dir:
        args["output"] = str(Path(output_dir) / "embeddings.npz")

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or "dna_embeddings"
    )

@mcp.tool()
def submit_dna_embeddings_v2(
    sequences: Optional[List[str]] = None,
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    model: str = "50M_multi_species_v2",
    layer: int = 12,
    max_positions: int = 32,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit enhanced DNA embedding extraction for background processing.

    Uses the enhanced version with shared library and better memory management.
    For large datasets that take >10 minutes to process.

    Args:
        sequences: List of DNA sequences to process
        input_file: Path to file with DNA sequences
        output_dir: Directory to save outputs
        model: Model name (default: 50M_multi_species_v2)
        layer: Layer to extract embeddings from (default: 12)
        max_positions: Maximum sequence length (default: 32)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the enhanced embedding extraction
    """
    script_path = str(SCRIPTS_DIR / "dna_embedding_v2.py")

    args = {
        "model": model,
        "layer": layer,
        "max_positions": max_positions
    }

    if sequences:
        args["sequences"] = sequences
    if input_file:
        args["input"] = input_file
    if output_dir:
        args["output"] = str(Path(output_dir) / "embeddings_v2.npz")

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or "dna_embeddings_v2"
    )

@mcp.tool()
def submit_attention_visualization(
    sequences: Optional[List[str]] = None,
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    model: str = "50M_multi_species_v2",
    layer: int = 1,
    head: int = 4,
    dpi: int = 300,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit attention visualization for background processing (large datasets).

    This operation creates attention visualizations for large datasets that take >10 minutes.
    Suitable for hundreds of sequences or high-resolution outputs.

    Args:
        sequences: List of DNA sequences to visualize
        input_file: Path to file with DNA sequences
        output_dir: Directory to save outputs
        model: Model name (default: 50M_multi_species_v2)
        layer: Attention layer to visualize (default: 1)
        head: Attention head to visualize (default: 4)
        dpi: Image resolution (default: 300)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the visualization job
    """
    script_path = str(SCRIPTS_DIR / "attention_visualization.py")

    args = {
        "model": model,
        "layer": layer,
        "head": head,
        "dpi": dpi
    }

    if sequences:
        args["sequences"] = sequences
    if input_file:
        args["input"] = input_file
    if output_dir:
        args["output"] = str(Path(output_dir) / "attention.png")

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or "attention_visualization"
    )

@mcp.tool()
def submit_nucleotide_prediction(
    sequences: Optional[List[str]] = None,
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    model: str = "50M_multi_species_v2",
    top_k: int = 5,
    max_positions: int = 32,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit nucleotide prediction for background processing (large datasets).

    This operation predicts nucleotides for large datasets that take >10 minutes.
    Suitable for thousands of sequences or complex models.

    Args:
        sequences: List of DNA sequences to analyze
        input_file: Path to file with DNA sequences
        output_dir: Directory to save outputs
        model: Model name (default: 50M_multi_species_v2)
        top_k: Number of top predictions per position (default: 5)
        max_positions: Maximum sequence length (default: 32)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the prediction job
    """
    script_path = str(SCRIPTS_DIR / "nucleotide_prediction.py")

    args = {
        "model": model,
        "top_k": top_k,
        "max_positions": max_positions
    }

    if sequences:
        args["sequences"] = sequences
    if input_file:
        args["input"] = input_file
    if output_dir:
        args["output"] = str(Path(output_dir) / "predictions.csv")

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or "nucleotide_prediction"
    )

# ==============================================================================
# Batch Processing Tools
# ==============================================================================

@mcp.tool()
def submit_batch_dna_analysis(
    input_files: List[str],
    analysis_type: str = "embeddings",
    model: str = "50M_multi_species_v2",
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch DNA analysis for multiple input files.

    Processes multiple sequence files with the specified analysis type.
    Suitable for processing many files at once with the same parameters.

    Args:
        input_files: List of input file paths containing DNA sequences
        analysis_type: Type of analysis - "embeddings", "attention", or "prediction"
        model: Model name to use for all files
        output_dir: Directory to save all outputs
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch analysis job
    """
    # Determine script based on analysis type
    script_map = {
        "embeddings": "dna_embedding.py",
        "attention": "attention_visualization.py",
        "prediction": "nucleotide_prediction.py"
    }

    if analysis_type not in script_map:
        return {
            "status": "error",
            "error": f"Invalid analysis_type: {analysis_type}. Must be one of: {list(script_map.keys())}"
        }

    script_path = str(SCRIPTS_DIR / script_map[analysis_type])

    # Convert list to comma-separated string for CLI
    inputs_str = ",".join(input_files)

    args = {
        "input_files": inputs_str,  # Special handling for batch mode
        "model": model
    }

    if output_dir:
        args["output_dir"] = output_dir

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"batch_{analysis_type}_{len(input_files)}_files"
    )

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()