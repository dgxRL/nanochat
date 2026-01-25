# Checkpoint Manager Documentation (`checkpoint_manager.py`)

Utilities for saving and loading model checkpoints.

## Functions

### `save_checkpoint(...)`
Saves:
- `model_{step}.pt`: Model weights (rank 0 only).
- `meta_{step}.json`: Metadata (config, step) (rank 0 only).
- `optim_{step}_rank{rank}.pt`: Optimizer state (sharded per rank).

### `load_checkpoint(checkpoint_dir, step, ...)`
Loads model weights, metadata, and optionally optimizer state.

### `build_model(checkpoint_dir, step, device, phase)`
High-level function to reconstruct a `GPT` instance `tokenizer` from a checkpoint path.
- Handles compilation artifacts (removes `_orig_mod.` prefix).
- Sets model to `train()` or `eval()` mode.
- Validates config against tokenizer vocab size.
