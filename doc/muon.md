# Muon Optimizer (`muon.py`)

Implements the Muon optimizer (MomentUm Orthogonalized by Newton-schulz).

## Overview

Muon represents a new class of optimizers that perform orthogonal updates.
- **Logic**: Runs SGD with momentum, then orthogonalizes the update matrix using Newton-Schulz iteration, finally scaling by the spectral radius.
- **Use Case**: Best for 2D parameters (weights of Linear layers). Not for embeddings or 1D vectors.

## Classes

### `zeropower_via_newtonschulz5`
Helper function.
- Iteratively computes the zeroth power (orthonormalization) of a matrix `G`.
- Uses a quintic polynomial for faster convergence.
- Runs in `bfloat16`.

### `Muon`
Standard single-device implementation.
- **step()**: Updates parameters by orthogonalizing gradients.

### `DistMuon`
Distributed version of Muon.
- Similar to `DistAdamW`, it shards the optimization step.
- Uses `reduce_scatter` for gradients and `all_gather` for weights.
- Handles grouping of parameters by shape to batch operations efficiently.
