# AGENTS.md - Coding Agent Guidelines for nanochat

## Project Overview

nanochat is a minimal, full-stack ChatGPT clone for a single 8×H100 node covering: tokenization, pretraining, finetuning (SFT/RL), evaluation, inference, and web serving. The codebase prioritizes clarity, hackability, and minimalism.

## Build & Environment

- **Runtime**: DGX Spark with `nvcr.io/nvidia/pytorch:25.08-py3` container
- **Package manager**: `uv` (not pip directly)
- **Python**: `>=3.10` (see `.python-version`)
- **Virtual environment**: `.venv/`

```bash
uv sync --extra gpu   # CUDA/GPU support
uv sync --extra cpu   # CPU-only
source .venv/bin/activate
```

### Transformer Engine Setup (for NVFP4 training)

Transformer Engine requires building from source on aarch64. Set CUDA include paths first:

```bash
# Set paths for CUDA headers (cudnn, nccl)
export NVIDIA_PKG=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia
export CPLUS_INCLUDE_PATH=$NVIDIA_PKG/cudnn/include:$NVIDIA_PKG/nccl/include
export LD_LIBRARY_PATH=$NVIDIA_PKG/cudnn/lib:$NVIDIA_PKG/nccl/lib:$LD_LIBRARY_PATH

# Install with --no-build-isolation (needs torch at build time)
pip install --no-build-isolation "transformer-engine[pytorch]==2.11.0"

# Verify
python -c "import transformer_engine.pytorch; print('TE OK')"
```

### Running Scripts
```bash
# Single GPU / CPU
python -m scripts.base_train
python -m scripts.chat_sft

# Multi-GPU (DDP)
torchrun --nproc_per_node=8 -m scripts.base_train
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_engine.py -v -s

# Run specific test function
python -m pytest tests/test_engine.py::test_kv_cache_basic -v -s

# Skip slow tests
python -m pytest tests/ -v -m "not slow"
```

### Test Markers
- `@pytest.mark.slow` - marks tests as slow (skip with `-m "not slow"`)
- `@pytest.mark.skipif(condition, reason="...")` - conditional skip

### Test Conventions
- Files: `tests/test_*.py`, Classes: `Test*`, Functions: `test_*`
- Use `@dataclass` for mock configs (see `tests/test_engine.py`)

## Code Style

### Imports
Order: stdlib → third-party → local (no blank lines between groups)
```python
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.gpt import GPT, GPTConfig
```

### Formatting
- **Line length**: ~100-120 chars (flexible)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes `"` preferred
- **Trailing commas**: In multi-line structures
- **Type hints**: Minimal (sparingly used return types only)

### Naming Conventions
- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

**Common abbreviations**: `B`=batch, `T`=seq_len, `C`=embd_dim, `H`=heads, `D`=head_dim, `ddp`=DDP, `bpb`=bits_per_byte, `lr`=learning_rate, `idx`=index, `ctx`=context

### Docstrings
```python
"""Short module description."""

def function(arg):
    """Brief description of function."""
    pass
```

### Error Handling & Assertions
Use `assert` for invariants with clear messages:
```python
assert isinstance(tokens, list), "expecting list of ints"
assert split in ["train", "test"], "GSM8K split must be train|test"
```

### Tensor Operations
```python
# Use F. for functional ops
F.relu(x), F.softmax(x), F.cross_entropy(logits, targets)

# Use inference_mode for inference, no_grad for init
@torch.inference_mode()
def generate(self, tokens, ...): ...

@torch.no_grad()
def init_weights(self): ...

# Mixed precision autocast
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
with autocast_ctx:
    ...
```

### Distributed Training
```python
ddp, rank, local_rank, world_size = get_dist_info()
print0("Only rank 0 prints this")
compute_init()  # setup
compute_cleanup()  # teardown
```

### CLI Arguments
```python
parser = argparse.ArgumentParser(description="Script description")
parser.add_argument("--depth", type=int, default=20, help="depth of Transformer")
args = parser.parse_args()
```

### Configuration Objects
Use `@dataclass` for configs:
```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
```

## Project Structure

```
nanochat/      # Core library modules
scripts/       # Training/eval scripts (run with python -m scripts.xxx)
tasks/         # Task definitions (MMLU, GSM8K, etc.)
tests/         # pytest test files
runs/          # Shell scripts for full training runs
dev/           # Development notebooks/utilities
```

## Key Architectural Decisions

1. **No config factories** - Direct instantiation
2. **Minimal dependencies** - Core in pure PyTorch
3. **Single-file modules** - Self-contained
4. **Explicit over implicit** - No magic
5. **Flash Attention 3** on Hopper+, SDPA fallback elsewhere
6. **bfloat16** on CUDA, float32 on CPU/MPS

## Contributing

- Disclose substantial LLM contributions in PRs
- Keep code minimal - avoid "framework" patterns
- Test on single-GPU and multi-GPU configs
- Prefer simple over clever (maximally-forkable)
