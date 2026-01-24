# Transformer Engine Installation SOP

## Overview

This document describes how to install NVIDIA Transformer Engine for NVFP4 training on DGX Spark (aarch64) with the `nvcr.io/nvidia/pytorch:25.08-py3` container.

## Problem Summary

Installing `transformer-engine` via uv or pip fails due to:

1. **Empty meta package**: The base `transformer-engine` package is just a meta package that requires extras like `[pytorch]`
2. **Build isolation issue**: uv builds packages in isolation, but `transformer-engine-torch` requires PyTorch at build time
3. **Missing CUDA headers**: Building from source fails because `cudnn.h` and `nccl.h` are not in the default include paths

## Environment

- **Hardware**: DGX Spark (aarch64)
- **Container**: `nvcr.io/nvidia/pytorch:25.08-py3`
- **CUDA**: 13.0
- **PyTorch**: 2.10.0+cu130
- **Python**: 3.10 (conda environment)

## Solution

### Step 1: Set CUDA Include Paths

The cudnn and nccl headers are located in the nvidia pip packages within the conda environment. Export these paths before building:

```bash
# Set the base path to nvidia packages
export NVIDIA_PKG=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia

# Add cudnn and nccl include paths for C++ compilation
export CPLUS_INCLUDE_PATH=$NVIDIA_PKG/cudnn/include:$NVIDIA_PKG/nccl/include:$CPLUS_INCLUDE_PATH

# Add library paths for linking
export LD_LIBRARY_PATH=$NVIDIA_PKG/cudnn/lib:$NVIDIA_PKG/nccl/lib:$LD_LIBRARY_PATH
```

### Step 2: Install Transformer Engine

Use `--no-build-isolation` to allow the build process to access the installed PyTorch:

```bash
pip install --no-build-isolation "transformer-engine[pytorch]==2.11.0"
```

### Step 3: Verify Installation

```bash
python -c "import transformer_engine.pytorch; print('Transformer Engine OK!')"
```

## Complete Installation Script

```bash
#!/bin/bash
# transformer_engine_install.sh

# Activate conda environment
source ~/miniconda3/bin/activate mydgx

# Set CUDA header paths
export NVIDIA_PKG=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia
export CPLUS_INCLUDE_PATH=$NVIDIA_PKG/cudnn/include:$NVIDIA_PKG/nccl/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$NVIDIA_PKG/cudnn/lib:$NVIDIA_PKG/nccl/lib:$LD_LIBRARY_PATH

# Install transformer-engine with PyTorch support
pip install --no-build-isolation "transformer-engine[pytorch]==2.11.0"

# Verify
python -c "import transformer_engine.pytorch; print('Transformer Engine installed successfully!')"
```

## Troubleshooting

### Error: `cudnn.h: No such file or directory`

The cudnn headers are not in the default include path. Solution:

```bash
# Find cudnn.h location
find $CONDA_PREFIX -name "cudnn.h"

# Add to include path
export CPLUS_INCLUDE_PATH=/path/to/cudnn/include:$CPLUS_INCLUDE_PATH
```

### Error: `nccl.h: No such file or directory`

Same issue as cudnn. Solution:

```bash
# Find nccl.h location  
find $CONDA_PREFIX -name "nccl.h"

# Add to include path
export CPLUS_INCLUDE_PATH=/path/to/nccl/include:$CPLUS_INCLUDE_PATH
```

### Error: `libcublas.so.12: cannot open shared object file`

This occurs when transformer-engine was built for CUDA 12 but you have CUDA 13. Use the `core_cu13` extra:

```bash
pip install --no-build-isolation "transformer-engine[pytorch,core_cu13]==2.11.0"
```

### Error: `RuntimeError: This package needs Torch to build`

The build is running in isolation without access to PyTorch. Solution:

```bash
pip install --no-build-isolation "transformer-engine[pytorch]==2.11.0"
```

## Why Not Use uv?

uv builds packages in isolated environments by default. Since `transformer-engine-torch` requires PyTorch at build time to:
- Detect CUDA version
- Find torch include paths
- Link against libtorch

The isolated build environment cannot access the installed PyTorch, causing the build to fail. Using pip with `--no-build-isolation` allows the build to use the system-installed PyTorch.

## Version Compatibility

| Transformer Engine | PyTorch | CUDA | Status |
|-------------------|---------|------|--------|
| 2.11.0 | 2.10.0+cu130 | 13.0 | Tested Working |
| 2.8.0 | 2.10.0+cu130 | 13.0 | Version mismatch issues |

## References

- [Transformer Engine GitHub](https://github.com/NVIDIA/TransformerEngine)
- [Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
