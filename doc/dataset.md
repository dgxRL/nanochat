# Dataset Module Documentation (`dataset.py`)

Handles the downloading and management of the FineWeb-Edu dataset used for pretraining.

## Overview

- **Source**: `https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle`
- **Format**: Parquet files (`shard_{index:05d}.parquet`).
- **Location**: Stores files in `~/.cache/nanochat/base_data` (by default).

## Functions

### `download_single_file(index)`
Downloads a specific shard by index.
- Implements retry logic with exponential backoff.
- Uses a temporary file for atomic writes.

### `list_parquet_files(data_dir=None)`
Returns a sorted list of all `.parquet` files in the data directory.

### `parquets_iter_batched(split, start=0, step=1)`
Iterates through row groups in the parquet files.
- **`split`**: "train" (all except last file) or "val" (last file).
- **`start/step`**: Used for distributed striding (DDP).
- **Yields**: Lists of text strings.

## CLI

Can be run as a script to download the dataset:
```bash
python -m nanochat.dataset -n <num_shards> -w <workers>
```
