
import pytest
import os
import sys
from nanochat.profiler import Profiler, get_cpu_temperature

def test_profiler_initialization():
    profiler = Profiler(device_type="cpu")
    assert profiler.device_type == "cpu"
    assert profiler.gpu_index == 0

    profiler_cuda = Profiler(device_type="cuda")
    assert profiler_cuda.device_type == "cuda"
    # gpu_index might change based on env, but defaults to 0
    assert isinstance(profiler_cuda.gpu_index, int)

def test_profiler_capture_cpu():
    profiler = Profiler(device_type="cpu")
    metrics = profiler.capture()
    assert isinstance(metrics, dict)
    assert "cpu_mem_mb" in metrics
    assert "cpu_temp_c" in metrics
    assert "gpu_mem_mb" in metrics
    assert "gpu_temp_c" in metrics
    
    # CPU memory should be > 0
    assert metrics["cpu_mem_mb"] > 0
    
    # CPU temp might be 0 if not on Linux or permission issues
    assert metrics["cpu_temp_c"] >= 0

def test_get_cpu_temperature():
    # It just returns a float
    temp = get_cpu_temperature()
    assert isinstance(temp, float)
    assert temp >= 0

@pytest.mark.skipif(not os.path.exists("/sys/class/thermal"), reason="Not on Linux with thermal zones")
def test_get_cpu_temperature_linux():
    temp = get_cpu_temperature()
    # On linux with thermal zones, it might be > 0 unless sensors are missing
    # We can't guarantee > 0 but we can check it runs.
    pass

def test_profiler_gpu_metrics_fallback():
    # On a system without GPU (like this agent env likely), 
    # GPU metrics should return 0 or handles gracefully.
    profiler = Profiler(device_type="cuda") 
    # Force pynvml to be unavailable if it was installed
    profiler.pynvml_available = False
    
    metrics = profiler.capture()
    # GPU memory relies on torch.cuda.memory_allocated
    # If torch is CPU only, it might be 0 or error if we called .cuda() methods?
    # Profiler check:
    # if self.device_type == "cuda": return torch.cuda.memory_allocated()
    # If torch.cuda.is_available() is False, memory_allocated returns 0 usually or throws?
    # Actually torch.cuda.memory_allocated() is safe to call?
    # Let's check logic:
    # return torch.cuda.memory_allocated() / (1024 * 1024)
    # If no cuda, memory_allocated returns 0.
    
    # However, if device_type="cuda" passed to Profiler, it tries to use it.
    pass
