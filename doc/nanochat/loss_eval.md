# Loss Evaluation Documentation (`loss_eval.py`)

Utilities for calculating loss metrics.

## Functions

### `evaluate_bpb(model, batches, steps, token_bytes)`
Calculates **Bits Per Byte (BPB)**.
- **Metric**: Standardized metric independent of vocabulary size.
- **Formula**: `total_loss_nats / (log(2) * total_bytes_of_Gold_tokens)`.
- **Logic**:
    - Iterates over validation batches.
    - Computes loss (nats).
    - Computes total bytes of target text (using `token_bytes` mapping).
    - Ignores special tokens and masked indices.
    - Aggregates across distributed ranks.
