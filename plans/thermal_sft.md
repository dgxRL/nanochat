# Thermal Pause Implementation Plan for SFT

## Goal
Add thermal protection logic to `scripts/chat_sft.py` to prevent overheating on DGX Spark, matching the implementation in `scripts/base_train.py`. This involves pausing training when CPU temperature exceeds safe limits and logging temperature to WandB/console.

## Proposed Changes

### 1. `scripts/chat_sft.py`

#### Imports
*   Import `thermal_pause_if_needed` from `nanochat.common`.
*   Import `Profiler` from `nanochat.profiler`.
*   *(Note: `base_train.py` uses `Profiler` to get temp)*

#### Initialization
*   Initialize `profiler = Profiler(device_type)` after `compute_init`.
*   Initialize `cpu_temp_history = []`.

#### Training Loop
*   **Inner Loop (Micro-steps)**:
    *   Get current temperature: `current_temp = profiler.get_cpu_temp()`.
    *   Call `thermal_pause_if_needed(cpu_temp_history, current_temp, history_size=10)`.
*   **Logging (Console)**:
    *   Capture profile metrics: `metrics = profiler.capture()`.
    *   Update `print0` to include `| temp: {metrics['cpu_temp_c']:.1f}C`.
*   **Logging (WandB)**:
    *   Add keys to `wandb_run.log`:
        *   `"train/temp"`: `metrics["cpu_temp_c"]`
        *   `"train/gpu_temp_c"`: `metrics["gpu_temp_c"]`
        *   `"train/cpu_mem"`: `metrics["cpu_mem_mb"]`
        *   `"train/gpu_mem"`: `metrics["gpu_mem_mb"]`

## Verification
*   Dry run `scripts/chat_sft.py` (using `--dry-run` or short iterations) to ensure no syntax errors and that temperature is printed.

## File to Modify
*   `scripts/chat_sft.py`
