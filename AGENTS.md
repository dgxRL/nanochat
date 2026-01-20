# AGENTS.md - Coding Agent Guidelines for nanochat

## Project Overview

nanochat is a minimal, full-stack ChatGPT clone designed to run on a single 8×H100 node. It covers the complete pipeline: tokenization, pretraining, finetuning (SFT/RL), evaluation, inference, and web serving. The codebase prioritizes clarity, hackability, and minimalism over exhaustive configurability.

## Build & Environment

### Package Manager
- **Use `uv`** for package management (not pip directly)
- Python version: `>=3.10` (see `.python-version`)
- Virtual environment: `.venv/`

```bash
# Install dependencies
uv sync --extra gpu   # For CUDA/GPU support
uv sync --extra cpu   # For CPU-only

# Activate venv
source .venv/bin/activate
```

### Running Scripts
Scripts are run as Python modules from the project root:
```bash
# Single GPU / CPU
python -m scripts.base_train
python -m scripts.chat_sft

# Multi-GPU (DDP)
torchrun --nproc_per_node=8 -m scripts.base_train
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
```

## Testing

### Test Commands
```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_engine.py -v -s

# Run a specific test function
python -m pytest tests/test_engine.py::test_kv_cache_basic -v -s

# Skip slow tests
python -m pytest tests/ -v -m "not slow"
```

### Test Markers
- `@pytest.mark.slow` - marks tests as slow (skip with `-m "not slow"`)
- `@pytest.mark.skipif(condition, reason="...")` - conditional skip

### Test File Conventions
- Test files: `tests/test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Use dataclasses for mock configs (see `tests/test_engine.py`)

## Code Style Guidelines

### Imports
Order imports as follows (no blank lines between groups in this codebase):
1. Standard library (`os`, `re`, `time`, `argparse`, etc.)
2. Third-party (`torch`, `wandb`, `datasets`, etc.)
3. Local modules (`from nanochat.xxx import ...`)

```python
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.gpt import GPT, GPTConfig
```

### Formatting
- **Line length**: ~100-120 characters (flexible)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes `"` preferred
- **Trailing commas**: Used in multi-line structures
- **No type hints** in most code (except return type annotations sparingly)

### Naming Conventions
- **Variables/functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `SPECIAL_TOKENS`, `SPLIT_PATTERN`)
- **Private functions**: `_leading_underscore`
- **Module-level variables**: lowercase with underscores

Common abbreviations used:
- `B` = batch size, `T` = sequence length, `C` = channels/embedding dim
- `H` = number of heads, `D` = head dimension
- `ddp` = Distributed Data Parallel
- `bpb` = bits per byte, `lr` = learning rate
- `idx` = index, `ctx` = context

### Docstrings
Use triple-quoted docstrings at module and function level:
```python
"""
Short description of module.

Longer explanation if needed.
"""

def function(arg):
    """Brief description of what this function does."""
    pass

def complex_function(arg1, arg2):
    """
    Multi-line docstring for complex functions.
    
    Note: Use 'Note:' for important caveats.
    TODO: Use 'TODO:' for future work.
    """
```

### Type Patterns
- Use `@dataclass` for configuration objects (see `GPTConfig`)
- Use `assert` statements for input validation and invariants
- Prefer explicit checks over try/except for expected conditions:
```python
assert isinstance(tokens, list), "expecting list of ints"
assert start >= 0, f"Start must be non-negative, got {start}"
```

### Error Handling
- Use `assert` for programming errors and invariants
- Use clear error messages with context:
```python
assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
assert split in ["train", "test"], "GSM8K split must be train|test"
```

### Tensor Operations
- Use `F.` for functional operations: `F.relu`, `F.softmax`, `F.cross_entropy`
- Prefer explicit device/dtype specification
- Use `torch.inference_mode()` for inference (not `torch.no_grad()`)
- Use `@torch.no_grad()` decorator for weight initialization

```python
@torch.inference_mode()
def generate(self, tokens, ...):
    ...

@torch.no_grad()
def init_weights(self):
    ...
```

### Distributed Training Patterns
- Use `get_dist_info()` to get DDP state: `ddp, rank, local_rank, world_size`
- Use `print0()` for rank-0 only printing
- Use `compute_init()` / `compute_cleanup()` for setup/teardown

### Common Patterns

**Generator/Iterator for streaming:**
```python
def generate(self, tokens, max_tokens, ...):
    for _ in range(max_tokens):
        ...
        yield token
```

**Autocast context for mixed precision:**
```python
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
with autocast_ctx:
    ...
```

**CLI argument parsing:**
```python
parser = argparse.ArgumentParser(description="Script description")
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
args = parser.parse_args()
```

## Project Structure

```
nanochat/          # Core library modules
scripts/           # Executable training/eval scripts (run with python -m scripts.xxx)
tasks/             # Task definitions (MMLU, GSM8K, etc.) for eval/training
tests/             # pytest test files
runs/              # Shell scripts for full training runs
dev/               # Development notebooks and utilities
```

## Key Architectural Decisions

1. **No config factories** - Direct instantiation, no abstract factories
2. **Minimal dependencies** - Core functionality in pure PyTorch
3. **Single-file modules** - Each module is self-contained
4. **Explicit over implicit** - Avoid magic, prefer clarity
5. **Flash Attention 3** on Hopper+, SDPA fallback elsewhere
6. **bfloat16** on CUDA, float32 on CPU/MPS

## Contributing Notes

- Disclose any substantial LLM contributions in PRs
- Keep code minimal and readable - avoid "framework" patterns
- Test on both single-GPU and multi-GPU configurations
- The codebase is "maximally-forkable" - prefer simple over clever
