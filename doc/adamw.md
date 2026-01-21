# DistAdamW Optimizer (`adamw.py`)

Implements a distributed version of the AdamW optimizer.

## Overview

This implementation is designed for distributed training (DDP), following Zeiss's ZeRO-2 style.
- **Sharded States**: Optimizer states are sharded across ranks to save memory.
- **Gradient Reduction**: Uses `reduce_scatter` to average gradients and `all_gather` to broadcast updated parameters.

## Class: `DistAdamW`

Inherits from `torch.optim.Optimizer`.

- **Parameters**: `lr`, `betas`, `eps`, `weight_decay`.
- **`step()`**:
    1.  **Reduce Scatter**: Gradients are averaged across ranks. Each rank gets a slice of the gradients.
    2.  **Update**: Each rank updates only its slice of parameters using the AdamW logic.
    3.  **All Gather**: The updated parameter slices are gathered back to full parameters on all ranks.

This effectively parallelizes the optimizer step and reduces peak memory usage per GPU.
