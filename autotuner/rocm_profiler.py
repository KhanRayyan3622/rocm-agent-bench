"""
ROCm Profiler Wrapper
On AMD Developer Cloud: wraps real rocm-smi / rocprof calls.
On Fireworks AI (current): simulates realistic GPU profiling data
based on published MI300X specs, so the auto-tuner logic is identical
whether run locally or via API.
"""
import subprocess
import shutil
import random
import time


class ROCmProfiler:
    def __init__(self):
        self.has_rocm = shutil.which("rocm-smi") is not None

    def get_gpu_stats(self) -> dict:
        """Return current GPU utilization, VRAM, and temperature."""
        if self.has_rocm:
            return self._real_stats()
        return self._simulated_stats()

    def _real_stats(self) -> dict:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showmemuse", "--showtemp", "--json"],
                capture_output=True, text=True, timeout=5
            )
            import json
            return json.loads(result.stdout)
        except Exception as e:
            return {"error": str(e), "source": "rocm-smi failed, using simulation"}

    def _simulated_stats(self) -> dict:
        """
        Simulated stats based on real published MI300X specs:
        192GB HBM3, 5.3TB/s bandwidth, 1300W TDP.
        Used during Fireworks AI development phase.
        """
        return {
            "gpu": "AMD Instinct MI300X (simulated profile)",
            "vram_total_gb": 192,
            "vram_used_gb": round(random.uniform(45, 85), 1),
            "gpu_utilization_pct": round(random.uniform(60, 95), 1),
            "hbm_bandwidth_gbps": round(random.uniform(4200, 5300), 0),
            "temperature_c": round(random.uniform(48, 62), 1),
            "power_draw_w": round(random.uniform(680, 950), 0),
            "source": "simulated (real values captured when on AMD Developer Cloud)",
        }
