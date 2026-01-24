# IMPL: Profiler Class for System Metrics

## Goal
Add a `Profiler` class to `nanochat/profiler.py` to collect CPU/GPU memory and temperature stats at each micro-step, enabling detailed performance tracking on DGX Spark.

## Background
- **Requirements**: Monitor GPU Mem, GPU Temp, CPU Mem, CPU Temp.
- **Environment**: Local development on Mac, execution on DGX Spark (Linux/NVIDIA).
- **Current State**: `psutil` and `pynvml` generic libraries are missing. `torch` provides GPU memory but not temp.

## Proposed Changes

### 1. New File: `nanochat/profiler.py`
Create a `Profiler` class that handles platform-specific (Linux vs Mac) metric collection robustly.

#### Dependencies (Optional but recommended)
- `psutil`: For CPU memory.
- `pynvml` (or `nvidia-ml-py`): For fast GPU temperature reading (much faster than `nvidia-smi`).

#### Class Structure
```python
class Profiler:
    def __init__(self, device_type="cuda"):
        self.device_type = device_type
        # Initialize pynvml if available
    
    def get_gpu_memory(self):
        # Uses torch.cuda.memory_allocated() / max_memory_allocated()
        pass

    def get_gpu_temp(self):
        # 1. Try pynvml (fastest)
        # 2. Fallback to /sys/class/drm/... (Linux)
        # 3. Fallback to nvidia-smi (slow, subprocess)
        # 4. Return 0.0 if failed
        pass

    def get_cpu_memory(self):
        # 1. Try psutil
        # 2. Fallback to reading /proc/meminfo (Linux)
        # 3. Return 0.0 if failed
        pass

    def get_cpu_temp(self):
        # Reuse nanochat.common.get_cpu_temperature logic
        pass

    def capture(self):
        # Returns a dict with all 4 metrics
        pass
```

### 2. scripts/base_train.py
- Instantiate `Profiler` before the loop.
- Inside the micro-step loop, call `profiler.capture()`.
- Log the metrics:
    - **Print**: Add key metrics (e.g., temps) to the progress bar string if critical.
    - **WandB**: Log all metrics to WandB (maybe subsampled if overhead is high, but user asked for "each micro_step").

### 3. nanochat/common.py
- Move `get_cpu_temperature` to `profiler.py` (or keep as alias) to centralize monitoring logic.

## Verification Plan

### Manual Verification
1.  **Local Test (Mac)**:
    - Run `python -m scripts.base_train ... --device-type cpu`
    - Verify CPU metrics are returned (GPU will be 0/dummy).
    - Verify no crashes due to missing `pynvml`.
2.  **Mock DGX Environment**:
    - Can't fully verify `pynvml` locally without NVIDIA GPU, but can mock the library presence in a test script.

### Automated Tests
- Create `tests/test_profiler.py` (if test suite exists) to check:
    - `Profiler` instantiation.
    - `capture()` returns dictionary with correct keys.
    - Graceful degradation when libs are missing.
