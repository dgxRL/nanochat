
import os
import time
import torch
import logging

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

def get_cpu_temperature():
    """
    Get CPU temperature in Celsius from /sys/class/thermal.
    Iterates through thermal zones to find "cpu" or "soc".
    Falls back to zone 0 if specific CPU zone not found.
    Returns 0.0 on failure (e.g., on non-Linux systems).
    """
    try:
        for zone_id in range(10):
            temp_path = f"/sys/class/thermal/thermal_zone{zone_id}/temp"
            type_path = f"/sys/class/thermal/thermal_zone{zone_id}/type"

            if os.path.exists(temp_path) and os.path.exists(type_path):
                with open(type_path, "r") as f:
                    zone_type = f.read().strip().lower()

                if "cpu" in zone_type or "soc" in zone_type:
                    with open(temp_path, "r") as f:
                        # Usually in millidegrees Celsius
                        return int(f.read().strip()) / 1000

        # Fallback to zone 0
        fallback_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(fallback_path):
            with open(fallback_path, "r") as f:
                return int(f.read().strip()) / 1000
    except (IOError, ValueError):
        pass
    return 0.0

class Profiler:
    def __init__(self, device_type="cuda"):
        self.device_type = device_type
        self.pynvml_available = False
        self.device_handle = None
        self.gpu_index = 0  # Default to 0, or determine from env
        
        # Determine GPU index from environment (e.g. LOCAL_RANK)
        if self.device_type == "cuda":
            try:
                self.gpu_index = int(os.environ.get("LOCAL_RANK", 0))
            except ValueError:
                self.gpu_index = 0

            if pynvml:
                try:
                    pynvml.nvmlInit()
                    self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                    self.pynvml_available = True
                except Exception as e:
                    logging.warning(f"Failed to initialize pynvml: {e}")

    def get_gpu_memory(self):
        """Returns GPU memory usage in MB."""
        if self.device_type == "cuda":
            # Use torch.cuda.memory_allocated() / 1024**2
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    def get_gpu_temp(self):
        """Returns GPU temperature in Celsius."""
        if self.device_type == "cuda":
            # 1. Try pynvml (fastest)
            if self.pynvml_available:
                try:
                    return pynvml.nvmlDeviceGetTemperature(self.device_handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    pass
            
            # 2. Fallback to /sys/class/drm/... (Linux)
            # This is tricky because mapping cardX to device index isn't always 1:1, 
            # but usually card0 is device 0. We'll skip complex mapping for now or try simple glob.
            # Simplified fallback:
            # sys_path = f"/sys/class/drm/card{self.gpu_index}/device/hwmon/hwmon*/temp1_input"
            # But we'll skip this as it's flaky.
            
            # 3. Fallback to nvidia-smi (slow, subprocess) - skipping to avoid overhead in training loop
            
            return 0.0
        return 0.0

    def get_cpu_memory(self):
        """Returns CPU memory usage in MB."""
        if psutil:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        # Fallback to /proc/self/statm or similar if needed, but psutil is standard
        return 0.0

    def get_cpu_temp(self):
        """
        Get CPU temperature in Celsius from /sys/class/thermal.
        """
        return get_cpu_temperature()

    def capture(self):
        """Returns a dict with all 4 metrics."""
        return {
            "gpu_mem_mb": self.get_gpu_memory(),
            "gpu_temp_c": self.get_gpu_temp(),
            "cpu_mem_mb": self.get_cpu_memory(),
            "cpu_temp_c": self.get_cpu_temp(),
        }
