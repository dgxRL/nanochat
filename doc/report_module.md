# Report Generation Documentation (`report.py`)

Generates markdown training reports.

## Overview

Collects data from various log files (`*.md`) generated during training and compiles them into a single `report.md`.

## Classes

### `Report`
- **`log(section, data)`**: Writes a data dictionary to a section file (e.g., `base-model-loss.md`).
- **`generate()`**:
    - Reads `header.md` (environment info).
    - Aggregates all section files in `EXPECTED_FILES` order.
    - Extracts key metrics (CORE, MMLU, etc.) for a summary table.
    - Calculates total training time.
- **`reset()`**: Clears previous report files and initializes a new header.

## System Info
- Collects: Git commit, GPU info, CPU/Memory stats, Python/PyTorch versions.
- **Bloat Metrics**: Counts lines/tokens in the codebase to track complexity.

## Usage
Called by training scripts (via `get_report()`) to log progress and by the user (CLI) to generate the final report.
