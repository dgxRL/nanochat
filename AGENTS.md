# Agent Guidelines for nanochat

This document provides instructions for AI agents operating in the `nanochat` repository.

## 1. Environment & Build

- **Dependency Management:** This project uses `uv`.
  - Install dependencies: `uv sync`
  - Run commands: `uv run <command>` (e.g., `uv run python scripts/base_train.py`)
  - Add dependencies: `uv add <package>`
  - Add dev dependencies: `uv add --dev <package>`

- **Python Version:** Requires Python >= 3.10.

- **Torch:** Note that `pyproject.toml` specifies custom indices for Torch (CPU vs CUDA 12.8). Respect `uv.lock`.

## 2. Testing

- **Framework:** `pytest` is used for testing.
- **Location:** Tests are located in the `tests/` directory.

### Common Commands
- **Run all tests:**
  ```bash
  uv run pytest
  ```
- **Run a specific test file:**
  ```bash
  uv run pytest tests/test_engine.py
  ```
- **Run a specific test function:**
  ```bash
  uv run pytest tests/test_engine.py::test_kv_cache_basic
  ```
- **Run with verbose output (recommended):**
  ```bash
  uv run pytest -v -s
  ```

### Testing Patterns
- Use descriptive docstrings for tests explaining what is being tested.
- Create mock classes for testing without loading real models/data.
- Test with simple, predictable data (e.g., uniform logits).
- Use assertions for verifying expected behavior and invariants.

## 3. Code Style & Conventions

### Philosophy
- **Minimalism:** "The best ChatGPT that $100 can buy." Keep code minimal, readable, and hackable.
- **No Over-Engineering:** Avoid giant configuration objects, complex factories, or excessive abstraction layers. Prefer flat structures.
- **Hackability:** The codebase is designed to be forked and tweaked. Clarity > generic flexibility.

### Formatting & Syntax
- **Indentation:** 4 spaces.
- **Line Length:** Generally follow standard PEP 8 (88-100 chars), but don't break code just for the sake of it if it reduces readability.
- **Imports:**
  1. Standard Library (`os`, `sys`, `dataclasses`)
  2. Third-party (`torch`, `wandb`, `numpy`)
  3. Local (`nanochat.model`, `nanochat.common`)
- **Type Hints:** Use `dataclasses` for configuration. Use type hints for function signatures where helpful, but strict typing is not enforced if it hurts readability.

### Docstrings
- Use triple-quoted docstrings for modules, classes, and functions.
- Keep docstrings descriptive and concise.
- Include usage examples in script docstrings (see `scripts/base_train.py`).
- Test functions should explain what is being tested and why.

### Naming Conventions
- **Classes:** `PascalCase` (e.g., `GPTConfig`, `CausalSelfAttention`)
- **Functions/Methods:** `snake_case` (e.g., `apply_rotary_emb`, `get_dist_info`)
- **Variables:** `snake_case`
- **Constants:** `UPPER_CASE` (e.g., `HAS_FA3`)

### Specific Patterns
- **Configuration:** Use `dataclasses` for config (e.g., `GPTConfig` in `gpt.py`).
- **Torch Usage:**
  - Use `torch.nn.functional` for stateless operations where possible (e.g., `F.rms_norm`).
  - Use `.item()` for CPU-GPU sync points (e.g., `loss.item()`).
  - Use `@torch.inference_mode()` decorator for inference-only functions.
  - Flash Attention 3 is integrated; respect the `HAS_FA3` checks.
- **CLI Arguments:** Use `argparse` in scripts. Follow the pattern in `scripts/base_train.py` (explicit arguments, copying to `user_config` for logging).
- **Paths:** Always use **absolute paths** when performing file operations. Resolve relative paths against the project root.
- **Distributed Training:**
  - Use `nanochat.common.print0` for printing to ensure output only comes from the master process.
  - Check `ddp_rank == 0` or `master_process` before performing rank-0-only operations.
  - Use `synchronize()` calls before timing operations.

### Assertions & Error Handling
- Use `assert` statements for preconditions and invariants (e.g., `assert temperature >= 0.0`).
- Use standard Python `try/except` blocks for error handling.
- For fatal errors in scripts, print a clear message and exit.
- For validation errors, raise `ValueError` with descriptive messages that include context (e.g., `f"Unsupported task type: {task_type}"`).
- Use `nanochat.common.print0` for logging in distributed settings to ensure only rank 0 prints.

## 4. Project Structure

- `nanochat/`: Core source code (models, engine, utils).
- `scripts/`: Entry points for training, evaluation, and serving (e.g., `base_train.py`, `chat_web.py`).
- `runs/`: Shell scripts for orchestrating experiments (`speedrun.sh`, `run1000.sh`).
- `tests/`: Unit tests.
- `tasks/`: Evaluation tasks (ARC, GSM8K, etc.).

## 5. Development Workflow

1. **Understand:** Read relevant files in `nanochat/` and `scripts/` before modifying.
2. **Modify:** Keep changes localized. If adding a feature, consider if it belongs in the core lib or a script.
3. **Verify:**
   - Run existing tests: `uv run pytest tests/test_engine.py`
   - If modifying training logic, ensure `scripts/base_train.py` or relevant scripts still run (dry run if possible).

## 6. Error Handling

- Use standard Python `try/except` blocks.
- For fatal errors in scripts, print a clear message and exit.
- Logging: Use `nanochat.common.print0` for printing to ensure output only comes from the master process in distributed settings.
- Raise `ValueError` with descriptive context for invalid inputs or configurations.
- Use assertions for invariants and preconditions during development.
