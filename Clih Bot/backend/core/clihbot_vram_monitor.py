"""
CLIHBot VRAM Monitor - Reliable VRAM tracking for both LMStudio and external servers.

This module provides automatic, reliable tracking of GPU VRAM usage by:
1. Monitoring model loading/unload events (when available)
2. Tracking chat request timing vs model status
3. Logging VRAM-related events for post-analysis

Works with both:
- LMStudio (via SDK events)
- External servers (via chat request patterns)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Configuration
VRAM_LOG_DIR = Path("D:\\CLIHBot\\App\\Projects\\Clih Bot\\backend\\vram_logs")
VRAM_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Track state
_last_vram_check = 0.0
_vram_history: list[dict] = []
_model_load_times: dict[str, float] = {}  # model_id -> load_timestamp
_chat_requests: list[dict] = []  # For correlating chats with VRAM


@dataclass
class VRAMEvent:
    """A VRAM-related event."""
    timestamp: float
    event_type: str  # "load", "unload", "chat", "status_check"
    model_id: str
    vram_bytes: int  # Estimated VRAM
    details: str = ""


@dataclass
class VRAMLogEntry:
    """A single log entry for post-analysis."""
    timestamp: str
    event_type: str
    model_id: str
    action: str
    details: str
    vram_before: int = 0
    vram_after: int = 0


class VRAMMonitor:
    """
    Monitors VRAM usage across different LLM server types.

    For LMStudio:
    - Tracks model load/unload via SDK events
    - Logs instance_id changes

    For External servers (llama.cpp, Ollama, etc.):
    - Tracks chat request timing
    - Correlates with /model/status endpoint
    - Logs patterns that indicate VRAM changes

    For both:
    - Provides log files for post-analysis
    - Tracks load times and patterns
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self._log_dir = log_dir or VRAM_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._history_lock = asyncio.Lock()
        self._current_vram_estimate = 0  # Bytes (estimate)
        self._model_load_times: dict[str, float] = {}
        self._last_model_load = {}  # model_id -> load_timestamp
        self._chat_request_count = 0
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

    def _get_timestamp(self) -> str:
        """Get ISO format timestamp."""
        return datetime.now().isoformat()

    def _log_event(
        self,
        event_type: str,
        model_id: str,
        action: str,
        details: str,
        vram_before: int = 0,
        vram_after: int = 0
    ) -> VRAMLogEntry:
        """Create a log entry."""
        return VRAMLogEntry(
            timestamp=self._get_timestamp(),
            event_type=event_type,
            model_id=model_id,
            action=action,
            details=details,
            vram_before=vram_before,
            vram_after=vram_after
        )

    def on_model_load(self, model_id: str) -> None:
        """
        Called when a model is loaded.
        Records the load time and estimates VRAM usage.
        """
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(self._on_model_load, model_id)

    def on_model_unload(self, model_id: str) -> None:
        """Called when a model is unloaded."""
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(self._on_model_unload, model_id)

    def on_chat_request(self, message: str, response_time_ms: int) -> None:
        """Called after a chat request completes."""
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(
            self._on_chat_request, message, response_time_ms
        )

    def on_status_check(self, loaded: bool, models: list[str]) -> None:
        """Called when model status is checked."""
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(
            self._on_status_check, loaded, models
        )

    def _on_model_load(self, model_id: str) -> None:
        """Record model load event."""
        now = time.time()
        self._model_load_times[model_id] = now

        # Estimate VRAM: model size + overhead
        # 9B model Q4_K_XL ~ 5GB, plus ~1GB overhead for KV cache
        estimated_vram = self._estimate_model_vram(model_id)

        entry = self._log_event(
            event_type="LOAD",
            model_id=model_id,
            action="model_loaded",
            details=f"VRAM estimate: {estimated_vram / 1e9:.2f} GB",
            vram_after=estimated_vram
        )

        self._write_log(entry)

    def _on_model_unload(self, model_id: str) -> None:
        """Record model unload event."""
        now = time.time()
        if model_id in self._last_model_load:
            load_time = self._last_model_load[model_id]
            self._model_load_times[model_id] = now
        else:
            self._model_load_times[model_id] = now

        # Unload frees VRAM
        entry = self._log_event(
            event_type="UNLOAD",
            model_id=model_id,
            action="model_unloaded",
            details="VRAM freed",
            vram_before=0,
            vram_after=0
        )

        self._write_log(entry)

    def _on_chat_request(self, message: str, response_time_ms: int) -> None:
        """Record chat request event."""
        self._chat_request_count += 1

        # Chat requests can trigger model loading (JIT)
        # Track for pattern analysis
        entry = self._log_event(
            event_type="CHAT",
            model_id="current",
            action=f"chat_request_{self._chat_request_count}",
            details=f"response_time: {response_time_ms}ms",
            vram_before=0,
            vram_after=0
        )

        self._write_log(entry)

    def _on_status_check(self, loaded: bool, models: list[str]) -> None:
        """Record model status check."""
        entry = self._log_event(
            event_type="STATUS",
            model_id="all",
            action=f"status_check",
            details=f"loaded={loaded}, models={len(models)}",
            vram_before=0,
            vram_after=0
        )

        self._write_log(entry)

    def _estimate_model_vram(self, model_id: str) -> int:
        """
        Estimate VRAM usage for a model.

        Formula: model_weight_size + kv_cache_overhead
        KV cache overhead ~ 2 * context_tokens * model_params / 1e9 bytes
        """
        # Parse model ID to get model name
        model_name = model_id.split("/")[-1].split("@")[0]

        # Approximate model sizes (can be refined)
        model_sizes = {
            "qwen3.5-9b": (7.8e9, 1e9),     # weight, kv_cache
            "qwen3.5-27b": (16.7e9, 2.5e9),
            "qwen3.5-35b": (20.6e9, 3e9),
            "qwen3-vl-8b": (6.2e9, 1e9),
            "glm-4.7-flash": (18.1e9, 2.5e9),
            "lfm2-24b": (19.6e9, 3e9),
        }

        # Default estimate for unknown models
        default_weight = 6e9  # 6GB average
        default_kv = 1e9

        if model_name in model_sizes:
            weight, kv = model_sizes[model_name]
            return weight + kv

        # Fallback: estimate based on model name
        if "9b" in model_name.lower():
            return default_weight + default_kv
        elif "27b" in model_name.lower() or "12b" in model_name.lower():
            return 15e9 + 2e9
        elif "35b" in model_name.lower() or "32b" in model_name.lower():
            return 20e9 + 2.5e9
        elif "vl" in model_name.lower():
            return 6e9 + 1e9
        else:
            return default_weight + default_kv

    def _write_log(self, entry: VRAMLogEntry) -> None:
        """Write a log entry to a file."""
        log_file = self._log_dir / f"vram_{self._get_timestamp()[:19].replace(':', '-').replace('.', '-')}.json"

        # Convert to dict for JSON serialization
        log_data = {
            "timestamp": entry.timestamp,
            "event_type": entry.event_type,
            "model_id": entry.model_id,
            "action": entry.action,
            "details": entry.details,
            "vram_before_bytes": entry.vram_before,
            "vram_after_bytes": entry.vram_after,
        }

        try:
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.error("Failed to write VRAM log: %s", e)

    def get_load_history(self) -> list[dict]:
        """Get model load history."""
        with self._history_lock:
            return [
                {
                    "model_id": model_id,
                    "load_time": load_time,
                }
                for model_id, load_time in self._last_model_load.items()
            ]

    def get_memory_delta(self, samples: int = 3, min_seconds: float = 1.0) -> tuple[int, float]:
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

    def get_chat_patterns(self) -> list[dict]:
        """Get chat request patterns for analysis."""
        with self._history_lock:
            return [
                {
                    "request_num": i + 1,
                    "response_time_ms": chat["response_time_ms"],
                    "timestamp": chat["timestamp"],
                }
                for i, chat in enumerate(self._chat_request_count)
            ]

    def has_vram_spike(self, threshold_bytes: int = 1_000_000_000) -> tuple[bool, int]:
        """
        Check if there's been a significant VRAM spike.

        For now, this is a placeholder since we don't have actual GPU stats.
        In a real implementation, this would compare current GPU memory usage
        to previous measurements.

        Args:
            threshold_bytes: Threshold for spike detection (default 1GB)

        Returns:
            Tuple of (has_spike, spike_size_bytes)
        """
        # Placeholder: return no spike detected
        # In a real implementation, this would compare current GPU memory
        # to the last measurement and detect significant increases
        return False, 0

    async def log_vram_status(self) -> dict:
        """
        Log current VRAM status from all available sources.

        Returns a dict with current status for logging.
        """
        # Get LLM server model status if available
        from config import get_settings
        settings = get_settings()
        mgmt_url = settings.lmstudio_base_url.replace("/v1", "/api/v1")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{mgmt_url}/models",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        models = data.get("data", []) if isinstance(data, dict) else data
                        loaded_count = len([m for m in models if isinstance(m, dict)])

                        # Get current configured model
                        async with session.get(
                            f"{mgmt_url}/models",
                            headers={"Content-Type": "application/json"},
                            timeout=aiohttp.ClientTimeout(total=2)
                        ) as resp2:
                            if resp2.status == 200:
                                data2 = await resp2.json(content_type=None)
                                configured = data2.get("data", [{}])[0] if isinstance(data2, dict) else {}
                            else:
                                configured = {}
                    else:
                        loaded_count = 0
                        configured = {}
        except Exception as e:
            logger.debug("VRAM status check failed: %s", e)
            loaded_count = 0
            configured = {}

        return {
            "timestamp": self._get_timestamp(),
            "loaded_model_count": loaded_count,
            "configured_model": configured.get("id", "unknown"),
            "load_history": self.get_load_history(),
        }


# ── Global instance ──────────────────────────────────────────────────────────

_monitor: Optional[VRAMMonitor] = None


def get_vram_monitor() -> VRAMMonitor:
    """Get the global VRAM monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = VRAMMonitor()
    return _monitor