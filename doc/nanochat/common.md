# Common Utilities Documentation (`common.py`)

A collection of utility functions and setup routines used throughout the codebase.

## Features

- **Logging**: `setup_default_logging`, `ColoredFormatter` for pretty terminal output.
- **Directory Management**: `get_base_dir` resolves the base directory (default `~/.cache/nanochat`), ensuring it exists.
- **File Utilities**: `download_file_with_lock` for safe concurrent downloads in DDP.
- **Distributed Setup**:
    - `get_dist_info()`: Reads environment variables (`RANK`, `WORLD_SIZE`).
    - `compute_init(device_type)`: Initializes the process group (`dist.init_process_group`), sets seeds, and sets device (CUDA/MPS/CPU).
    - `compute_cleanup()`: Destroys process group.
- **Device Autodetection**: `autodetect_device_type()`.
