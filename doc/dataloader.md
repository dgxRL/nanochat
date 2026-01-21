# Dataloader Module Documentation (`dataloader.py`)

Provides efficient, distributed data loading for the training loop.

## Overview

It streams text from the parquet files, tokenizes them on-the-fly, and yields batches of token IDs (input/target pairs).

## Functions

### `tokenizing_distributed_data_loader_with_state(...)`
The core generator function.

- **Arguments**: `B` (batch size), `T` (sequence length), `split`, `tokenizer_threads`, `device`, `resume_state_dict`.
- **Logic**:
    1.  **Iterate Documents**: Streams text using `dataset.parquets_iter_batched`.
    2.  **Tokenize**: Encodes text using the tokenizer (multi-threaded).
    3.  **Buffer**: Accumulates tokens in a `deque`.
    4.  **Yield**: Pops chunks of size `B*T + 1`.
        - `inputs`: `[0, ..., T-1]`
        - `targets`: `[1, ..., T]`
        - Returns `(inputs, targets, state_dict)` where `state_dict` tracks position (file/row group index) for resumption.

### `tokenizing_distributed_data_loader(...)`
Helper wrapper that discards the state dict and only yields `(inputs, targets)`.
