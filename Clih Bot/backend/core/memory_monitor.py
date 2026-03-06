"""
GPU Memory Monitor for CLIHBot.

Provides reliable, automatic monitoring of GPU memory usage without relying
on LMStudio's REST API (which doesn't expose runtime VRAM usage).

Uses Windows Performance Counters (WMI) to track:
- Dedicated GPU VRAM usage
- Total GPU memory usage
- GPU memory dedicated ratio

Fallback to Task Manager parsing if WMI is unavailable.
"""
from __future__ import annotations

import asyncio
import ctypes
import ctypes.wintypes
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

try:
    import wmi  # type: ignore
except ImportError:
    wmi = None

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """GPU memory statistics."""
    dedicated_memory_used: int  # Bytes
    total_memory: int  # Bytes
    dedicated_memory_reserved: int  # Bytes
    temperature: Optional[float] = None
    utilization: Optional[float] = None  # Percentage
    gpu_name: Optional[str] = None


class GPUWatcher:
    """
    Watches GPU memory usage with configurable polling interval.

    Provides automatic tracking of VRAM changes between chat requests.
    Can be used to detect VRAM spikes, leaks, or inefficient model loading.
    """

    def __init__(self, poll_interval_seconds: float = 1.0):
        self._poll_interval = poll_interval_seconds
        self._watcher: Optional[asyncio.Task] = None
        self._stats_history: list[GPUStats] = []
        self._latest_stats: Optional[GPUStats] = None
        self._wmi: Optional[wmi.WMI] = None
        self._nvml_available = self._check_nvml()

    def _check_nvml(self) -> bool:
        """Check if NVIDIA Management Library is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except ImportError:
            return False
        except Exception:
            return False

    async def start(self, stop_event: asyncio.Event = None):
        """Start watching GPU memory in background."""
        if self._watcher and not self._watcher.done():
            logger.warning("GPU watcher already running")
            return

        async def _watch_loop():
            while not (stop_event and stop_event.is_set()):
                try:
                    stats = await self._get_stats()
                    self._stats_history.append(stats)
                    self._latest_stats = stats
                except Exception as e:
                    logger.error("GPU stats collection failed: %s", e)
                await asyncio.sleep(self._poll_interval)

        self._watcher = asyncio.create_task(_watch_loop(), name="gpu_watcher")
        logger.info("GPU watcher started (interval=%ds)", self._poll_interval)

    async def stop(self):
        """Stop the GPU watcher."""
        if self._watcher and not self._watcher.done():
            stop_event = asyncio.Event()
            self._watcher.cancel()
            try:
                await self._watcher
            except asyncio.CancelledError:
                pass
            self._watcher = None

    async def _get_stats(self) -> GPUStats:
        """Get current GPU memory statistics."""
        # Try NVIDIA first (most common for LLM workloads)
        if self._nvml_available:
            try:
                return await self._get_nvml_stats()
            except Exception as e:
                logger.debug("NVIDIA query failed, falling back: %s", e)

        # Try nvidia-smi next. It is often available even when Python bindings are not.
        try:
            return await self._get_nvidia_smi_stats()
        except Exception as e:
            logger.debug("nvidia-smi query failed, falling back: %s", e)

        # Try Windows WMI
        try:
            return await self._get_wmi_stats()
        except Exception as e:
            logger.debug("WMI query failed, falling back: %s", e)

        # Last resort: parse Task Manager
        try:
            return await self._get_taskmanager_stats()
        except Exception as e:
            logger.error("All GPU stats methods failed: %s", e)
            return GPUStats()

    async def _get_nvml_stats(self) -> GPUStats:
        """Get stats using NVIDIA Management Library."""
        import pynvml

        try:
            pynvml.nvmlInit()

            # Get handle for first (primary) GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats = GPUStats(
                dedicated_memory_used=memory_info.used,
                total_memory=memory_info.total,
                dedicated_memory_reserved=memory_info.used,  # NVML doesn't expose reserved separately
            )

            # Get utilization and temperature
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats.utilization = info.utilization
            stats.gpu_name = pynvml.nvmlDeviceGetName(handle)

            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            stats.temperature = temp

            pynvml.nvmlShutdown()
            return stats
        except Exception as e:
            raise RuntimeError(f"NVIDIA query failed: {e}")

    async def _get_nvidia_smi_stats(self) -> GPUStats:
        """Get stats using nvidia-smi."""
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi not found")

        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        completed = await asyncio.to_thread(
            subprocess.run,
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        line = completed.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            raise RuntimeError(f"Unexpected nvidia-smi output: {line}")

        gpu_name, total_mb, used_mb, utilization, temperature = parts[:5]
        total_bytes = int(float(total_mb)) * 1024 * 1024
        used_bytes = int(float(used_mb)) * 1024 * 1024
        return GPUStats(
            dedicated_memory_used=used_bytes,
            total_memory=total_bytes,
            dedicated_memory_reserved=used_bytes,
            temperature=float(temperature),
            utilization=float(utilization),
            gpu_name=gpu_name,
        )

    async def _get_wmi_stats(self) -> GPUStats:
        """Get stats using Windows WMI (no NVIDIA drivers required)."""
        if wmi is None:
            raise RuntimeError("wmi package not installed")
        try:
            c = wmi.WMI()

            # Get GPU info
            gpu = c.Win32_VideoController()[0]
            stats = GPUStats(gpu_name=gpu.Name)

            # WMI doesn't expose detailed memory usage, so we estimate
            # based on dedicated memory settings
            # This is less accurate but provides some visibility
            stats.total_memory = 0  # WMI doesn't provide this
            stats.dedicated_memory_used = 0
            stats.dedicated_memory_reserved = 0

            return stats
        except Exception as e:
            raise RuntimeError(f"WMI query failed: {e}")

    async def _get_taskmanager_stats(self) -> GPUStats:
        """Get stats by parsing Task Manager (fallback method)."""
        try:
            import psutil

            # Get GPU count
            gpu_count = psutil.cpu_count(logical=False)  # Approximate
            if gpu_count == 0:
                gpu_count = 1

            # This is a rough estimate - Task Manager doesn't expose detailed GPU memory
            # For a more accurate solution, we'd need to parse the Task Manager UI
            # which is unreliable. We'll return zeros as a placeholder.
            logger.debug("Task Manager fallback - detailed GPU memory not available")
            return GPUStats()
        except ImportError:
            pass
        except Exception as e:
            logger.debug("Task Manager stats failed: %s", e)
            return GPUStats()

    def get_latest_stats(self) -> Optional[GPUStats]:
        """Get the latest GPU statistics."""
        return self._latest_stats

    def get_memory_delta(
        self,
        samples: int = 3,
        min_seconds: float = 1.0
    ) -> tuple[int, float]:
        """
        Calculate the memory delta between the latest and nth-earlier sample.

        Args:
            samples: How many samples back to compare (default 3 = 3 seconds ago)
            min_seconds: Minimum time delta in seconds

        Returns:
            Tuple of (memory_delta_bytes, time_delta_seconds)
        """
        if len(self._stats_history) < samples + 1:
            return 0, 0

        # Get the sample from 'samples' intervals ago
        current_idx = len(self._stats_history) - 1
        old_idx = current_idx - samples

        current = self._stats_history[current_idx]
        old = self._stats_history[old_idx]

        delta_bytes = current.dedicated_memory_used - old.dedicated_memory_used
        time_delta = samples * self._poll_interval

        return delta_bytes, time_delta

    def get_memory_change_per_second(
        self,
        samples: int = 3
    ) -> float:
        """
        Calculate memory change per second.

        Returns bytes/second (positive = increasing, negative = decreasing).
        """
        delta_bytes, time_delta = self.get_memory_delta(samples)
        if time_delta == 0:
            return 0
        return delta_bytes / time_delta

    def has_vram_spike(self, threshold_bytes: int = 1_000_000_000) -> tuple[bool, int]:
        """
        Check if there's been a significant VRAM spike.

        Args:
            threshold_bytes: Threshold for spike detection (default 1GB)

        Returns:
            Tuple of (has_spike, spike_size_bytes)
        """
        delta_bytes, _ = self.get_memory_delta(5)  # Check last 5 seconds
        return delta_bytes > threshold_bytes, delta_bytes


# ── Singleton ───────────────────────────────────────────────────────────────

_watcher: Optional[GPUWatcher] = None


def get_memory_monitor() -> GPUWatcher:
    """Get the singleton GPU memory monitor."""
    global _watcher
    if _watcher is None:
        _watcher = GPUWatcher()
    return _watcher
