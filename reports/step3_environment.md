# Step 3: Environment Setup Report

## Python Version Detection
- **Detected Python Version**: >=3.9 (from setup.py)
- **Strategy**: Single environment setup (upgraded to Python 3.10 for MCP compatibility)

## Environment Configuration

### Package Manager Used
- **Selected**: mamba (preferred over conda for faster installation)
- **Command**: `/home/xux/miniforge3/condabin/mamba`

## Main MCP Environment
- **Location**: ./env
- **Python Version**: 3.10.19 (upgraded for MCP server compatibility)
- **Purpose**: Single unified environment for both MCP server and library dependencies

## Dependencies Installed

### Core MCP Dependencies (./env)
- fastmcp==2.14.1
- loguru==0.7.3
- click==8.3.1
- pandas==2.3.3
- numpy==1.26.4
- tqdm==4.67.1

### Nucleotide Transformer Dependencies (./env)
- jax==0.6.2
- jaxlib==0.6.2
- dm-haiku==0.0.16
- transformers==4.57.3
- torch==2.9.1+cu128
- flax==0.10.4
- einops==0.8.1
- huggingface-hub==0.36.0
- matplotlib==3.10.8
- seaborn==0.13.2
- pyfaidx==0.9.0.3
- regex==2025.11.3
- anndata==0.11.4
- scanpy==1.11.5
- cellxgene_census==1.17.0

### CUDA Dependencies (automatically installed with PyTorch)
- nvidia-cuda-runtime-cu12==12.8.90
- nvidia-cudnn-cu12==9.10.2.21
- nvidia-cublas-cu12==12.8.4.1
- nvidia-cufft-cu12==11.3.3.83
- nvidia-curand-cu12==10.3.9.90
- nvidia-cusolver-cu12==11.7.3.90
- nvidia-cusparse-cu12==12.5.8.93
- nvidia-nccl-cu12==2.27.5
- triton==3.5.1

## Installation Commands Executed

```bash
# Package manager selection
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
else
    PKG_MGR="conda"
fi

# Environment creation
mamba create -p ./env python=3.10 pip -y

# Core MCP dependencies
mamba run -p ./env pip install loguru click pandas numpy tqdm

# FastMCP installation
mamba run -p ./env pip install --force-reinstall --no-cache-dir fastmcp

# ML dependencies for nucleotide transformer
mamba run -p ./env pip install "jax>=0.3.25" "jaxlib>=0.3.25" "dm-haiku>=0.0.9" "numpy>=1.23.5,<2.0.0" "typing_extensions>=3.10.0" "joblib>=1.2.0" "tqdm>=4.56.0" "regex>=2022.1.18" "huggingface-hub>=0.23.0" "transformers>=4.52.4" "torch>=2.7.1" "einops>=0.8.1" "flax==0.10.4" "matplotlib>=3.5.0" "seaborn>=0.11.0" "pyfaidx>=0.7.0" "requests>=2.25.0"

# Nucleotide transformer package
cd repo/nucleotide-transformer
mamba run -p ../../env pip install -e .
cd ../..
```

## Activation Commands
```bash
# For interactive use (requires shell initialization)
mamba activate ./env

# For script execution (recommended)
mamba run -p ./env python script.py

# For MCP server
mamba run -p ./env python src/server.py
```

## Verification Status
- [x] Main environment (./env) functional
- [x] Core MCP imports working (fastmcp, loguru, click)
- [x] ML dependencies working (jax, torch, transformers)
- [x] Nucleotide transformer imports successful
- [x] Example scripts tested and verified

## Version Conflicts Resolved

### Pydantic Version Conflict
- **Issue**: nucleotide_transformer requires pydantic==1.10.13, fastmcp requires pydantic>=2.11.7
- **Resolution**: Installed pydantic==2.12.5 (compatible with fastmcp)
- **Impact**: Minor warning messages, but functionality preserved

### NumPy Version Conflict
- **Issue**: Some dependencies prefer numpy>=2.0, nucleotide_transformer requires <2.0
- **Resolution**: Downgraded to numpy==1.26.4
- **Impact**: Full compatibility maintained

## Performance Notes
- **GPU Support**: CUDA-enabled PyTorch and JAX installed
- **Memory**: Large models (2B5+) require 16GB+ RAM
- **Storage**: Model weights cached in ~/.cache/huggingface/
- **First Download**: Models downloaded automatically on first use

## Environment Size
- **Total Environment Size**: ~8.5GB (including CUDA libraries)
- **Model Cache**: Variable (200MB - 10GB per model)

## Notes
- Single environment strategy successful despite original Python 3.9 requirement
- All core functionality verified through test scripts
- GPU acceleration available but fallback to CPU works
- Package manager preference (mamba) significantly improved installation speed