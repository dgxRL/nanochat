# IMPL: Dynamic Load Management based on CPU Temperature

## Goal
Prevent training crashes on DGX Spark by pausing training when CPU temperature exceeds safe limits.

## Background
Training crashes occur due to CPU overheating (>95C).
Requirement:
- Check CPU temperature after each iteration.
- Calculate moving average of last 100 steps.
- If average > 92C, pause training.
- Resume when temperature < 85C.
- Reference: `dgxtop/system_monitor.py`.
-- file is @ https://github.com/GigCoder-ai/dgxtop/blob/main/dgxtop/system_monitor.py to get cpu temperature on dgx spark
## Proposed Changes

### 1. scripts/base_train.py

#### [NEW] Helper Function `get_cpu_temperature`
Add a helper function to read CPU temperature from `/sys/class/thermal`.
- Iterate through thermal zones to find "cpu" or "soc".
- Return temperature in Celsius.
- Fallback to zone 0 if specific CPU zone not found.
- Handle exceptions and return 0.0 on failure.

#### [MODIFY] Training Loop
- Import `collections.deque` for moving average.
- Initialize `cpu_temp_history = deque(maxlen=100)`.
- Inside the main `while True:` loop (after step update):
    1.  `current_temp = get_cpu_temperature()`
    2.  `cpu_temp_history.append(current_temp)`
    3.  If `len(cpu_temp_history) == 100`:
        - `avg_temp = sum(cpu_temp_history) / 100`
        - If `avg_temp > 92.0`:
            - Log warning: "CPU Overheating (Avg: {avg_temp:.1f}C). Pausing..."
            - Enter `while True:` pause loop.
            - `time.sleep(30)`
            - `current_temp = get_cpu_temperature()`
            - Log status: "Current CPU Temp: {current_temp:.1f}C. Waiting for < 85.0C..."
            - If `current_temp < 85.0`:
                - Log: "CPU Cooled down. Resuming..."
                - Break pause loop.

## Verification Plan

### Automated Tests
- None possible for hardware sensors without mocking throughout the stack.
- Will rely on manual verification and log checking.

### Manual Verification
1.  **Read Verification**: Run the script briefly and check if "current CPU temp" is logged (will add temporary logging for verification).
2.  **Logic Verification**:
    - Temporarily set `THRESHOLD = 40.0` (or below current temp) and `RESUME_THRESHOLD = 35.0` in the code.
    - Run `python -m scripts.base_train ...`
    - Observe if it enters the "Pausing..." state.
    - Observe if it resumes (might need to stop load or artificially cool, or just set resume threshold higher for the test).
    - **Revert thresholds** to 92.0/85.0 after testing.
