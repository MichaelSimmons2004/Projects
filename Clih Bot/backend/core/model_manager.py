"""
LMStudio Model Lifecycle Manager with External Server Support.

Supports two modes of operation:

1. **LMStudio mode** (default):
   - Auto-starts LMStudio server if configured
   - Manages model loading/unloading via SDK or REST API
   - Supports KV cache quantization via SDK
   - TTL-based idle model management
   - **VRAM monitoring**: Tracks model loads and can detect VRAM spikes

2. **External server mode** (llama.cpp, Ollama, vLLM, etc.):
   - Detects if external LLM server is running
   - Skips model loading (model already loaded on remote server)
   - Uses pass-through for chat requests
   - Can still use model selection for choosing which model to use

Hybrid detection logic:
- If `auto_detect_external_server=true`: checks for common LLM server ports
- If external server detected AND `prefer_external_servers=true`: use external
- Falls back to LMStudio if no external server found or detection disabled

Usage:
    manager = get_model_manager()
    await manager.ensure_loaded()   # call before chat
    # ... chat happens, ttl resets on each request ...
    await manager.unload()          # optional: immediate VRAM free

    # Get available models
    models = await manager.get_available_models()
    best, reason = await manager.select_best_model(manual_preference="qwen/qwen3-vl-8b")

VRAM Monitoring:
    monitor = get_model_manager()
    # Log VRAM status after each chat request
    await monitor.log_vram_status()
    # Check for VRAM spikes
    has_spike, spike_size = monitor.check_vram_spike()
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import aiohttp

from config import get_settings

logger = logging.getLogger(__name__)


def get_llm_server_url() -> str:
    """Get the LLM server URL from settings."""
    settings = get_settings()
    # Support both old and new variable names for backward compatibility
    if hasattr(settings, 'llm_server_url'):
        return settings.llm_server_url
    return settings.lmstudio_base_url


def _server_root(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/api/v1"):
        return base[:-7]
    if base.endswith("/v1"):
        return base[:-3]
    return base


def _openai_base_url(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/api/v1"):
        return f"{base[:-7]}/v1"
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def get_llm_server_management_url() -> str:
    """Get the LLM server management URL from settings."""
    settings = get_settings()
    # Support both old and new variable names for backward compatibility
    if hasattr(settings, 'llm_server_url'):
        return f"{_server_root(settings.llm_server_url)}/api/v1"
    return f"{_server_root(settings.lmstudio_base_url)}/api/v1"

# Track the last model load time for detecting VRAM spikes
_MODEL_LOAD_HISTORY: list[tuple[str, float]] = []  # (model_id, load_timestamp)


def get_model_load_history() -> list[tuple[str, float]]:
    """Get the model load history."""
    return list(_MODEL_LOAD_HISTORY)


def clear_model_load_history() -> None:
    """Clear the model load history."""
    _MODEL_LOAD_HISTORY.clear()


def add_model_load_event(model_id: str) -> None:
    """Record a model load event."""
    _MODEL_LOAD_HISTORY.append((model_id, asyncio.get_event_loop().time()))


def get_model_load_delta_seconds(model_id: str) -> float:
    """
    Get seconds since the last time this model was loaded.
    Returns -1 if model was never loaded.
    """
    for model_id_loaded, timestamp in reversed(_MODEL_LOAD_HISTORY):
        if model_id_loaded == model_id:
            return asyncio.get_event_loop().time() - timestamp
    return -1
    """Get the model load history."""
    return list(_MODEL_LOAD_HISTORY)


def clear_model_load_history() -> None:
    """Clear the model load history."""
    _MODEL_LOAD_HISTORY.clear()


def add_model_load_event(model_id: str) -> None:
    """Record a model load event."""
    _MODEL_LOAD_HISTORY.append((model_id, asyncio.get_event_loop().time()))


def get_model_load_delta_seconds(model_id: str) -> float:
    """
    Get seconds since the last time this model was loaded.
    Returns -1 if model was never loaded.
    """
    for model_id_loaded, timestamp in reversed(_MODEL_LOAD_HISTORY):
        if model_id_loaded == model_id:
            return asyncio.get_event_loop().time() - timestamp
    return -1

# Common LLM server ports and patterns for detection
# Format: (port, description, common_server_types)
COMMON_LLM_PORTS = [
    (1234, "llama.cpp / LM Studio", ["llama.cpp", "lmstudio"]),
    (11434, "Ollama", ["ollama"]),
    (8000, "vLLM / text-generation-webui", ["vllm", "tgui"]),
    (8080, "various (llama.cpp, text-gen-webui, etc.)", ["llama.cpp", "tgui"]),
    (8081, "various (llama.cpp, text-gen-webui, etc.)", ["llama.cpp", "tgui"]),
    (5000, "text-generation-webui / Ollama", ["tgui", "ollama"]),
    (21434, "Together AI local", ["together"]),
    (22, "LLaMA-Factory", ["llama-factory"]),
    (3000, "various (vLLM, text-gen-webui)", ["vllm", "tgui"]),
    (7860, "Gradio (sometimes used for local servers)", ["gradio"]),
]

# Patterns to detect server type from URL
URL_PATTERNS = {
    r".*/v1/": "openai-compatible",
    r".*/api/v1/": "lmstudio",
    r".*/v1beta/": "openai-compatible-beta",
}

# Shared singleton — one instance per server process
_manager: "ModelManager | None" = None


def get_model_manager() -> "ModelManager":
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


class ModelManager:

    def __init__(self) -> None:
        self._settings = get_settings()
        self._instance_id: str | None = None  # last known loaded instance id
        self._load_lock = asyncio.Lock()       # prevent concurrent load races
        self._selector = None                  # ModelSelector instance
        self._external_server_detected: bool = False  # Track if external server was detected
        self._external_server_url: str | None = None  # URL of detected external server

        # Initialize model selector if ranking mode is auto
        if self._settings.model_ranking_mode == "auto":
            from core.model_selector import get_model_selector
            self._selector = get_model_selector(self._settings)

    def refresh_selector(self) -> None:
        """Rebuild or refresh the selector after live config changes."""
        if self._settings.model_ranking_mode == "auto":
            from core.model_selector import get_model_selector
            self._selector = get_model_selector(self._settings)
            self._selector.refresh_from_settings()
        else:
            self._selector = None

    # ── Public interface ──────────────────────────────────────────────────────

    async def ensure_loaded(self, manual_preference: str | None = None) -> bool:
        """
        Guarantee the configured model is in memory before a chat request.

        If MODEL_RANKING_MODE='auto' and manual_preference is provided:
        - Select the best model matching the requested family
        - Update llm_server_model config with selected model

        External server handling:
        - If `auto_detect_external_server=true`: detects external LLM servers
        - If external server detected AND `prefer_external_servers=true`: uses external
        - Falls back to the configured server if no external server is preferred

        Args:
            manual_preference: Optional user-selected model ID or family

        Returns:
            True if the model is loaded (or already was), False on failure.
            Does nothing if model_auto_load is disabled in config.
        """
        if not self._settings.model_auto_load:
            logger.debug("Model auto-load disabled, skipping")
            return True

        # Detect external servers if enabled
        if self._settings.auto_detect_external_server:
            await self._detect_external_servers()

        # Detection alone should not change routing; external preference does that.
        is_external = (
            self._external_server_detected
            and self._settings.prefer_external_servers
        )

        if is_external:
            logger.info(
                "Using external server %s (%s) — skipping LLM server model load",
                self._external_server_url,
                self._external_server_url.rsplit("/", 1)[-1].split(":")[-1] if ":" in self._external_server_url else "unknown"
            )
            return True

        # For non-LMStudio servers, skip model loading
        server_url = get_llm_server_url()
        is_lmstudio = await self._is_lmstudio_server(server_url)

        if not is_lmstudio:
            logger.info(
                "Using external OpenAI-compatible server %s — skipping model load",
                self._settings.lmstudio_base_url
            )
            return True

        async with self._load_lock:
            status = await self.status()
            if status.get("loaded"):
                return True

            # Check if we should use model selection
            if self._selector and manual_preference:
                return await self._ensure_with_selection(manual_preference)

            logger.info(
                "Model '%s' not loaded — loading now...", self._settings.lmstudio_model
            )
            return await self._load()

    async def status(self) -> dict[str, Any]:
        """
        Query the LLM server for the list of currently loaded models.
        Returns a dict describing whether our configured model is loaded.
        """
        models_url = f"{_openai_base_url(get_llm_server_url())}/models"
        headers = self._auth_headers()
        is_lmstudio = await self._is_lmstudio_server(get_llm_server_url())
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    models_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        return {"loaded": False, "error": f"HTTP {resp.status}", "models": []}
                    data = await resp.json(content_type=None)

            # LMStudio v1 API returns {"data": [...]} just like OpenAI
            models = data.get("data", data) if isinstance(data, dict) else data
            model_ids = [m.get("id", "") for m in models if isinstance(m, dict)]
            loaded = any(
                self._settings.lmstudio_model in mid or mid in self._settings.lmstudio_model
                for mid in model_ids
            ) if is_lmstudio else bool(model_ids)
            if loaded and model_ids:
                self._instance_id = model_ids[0]

            return {
                "loaded": loaded,
                "models": model_ids,
                "instance_id": self._instance_id,
                "configured_model": self._settings.lmstudio_model,
                "server_url": get_llm_server_url(),
                "is_lmstudio": is_lmstudio,
            }

        except Exception as exc:
            logger.warning("Model status check failed: %s", exc)
            return {"loaded": False, "error": str(exc), "models": []}

    async def load(self) -> dict[str, Any]:
        """
        Explicitly load the configured model via LMStudio's REST API.
        Returns the response dict or an error dict.
        """
        async with self._load_lock:
            return await self._load_and_report()

    async def unload(self) -> dict[str, Any]:
        """
        Explicitly unload the configured model to free VRAM immediately.
        Uses the instance_id from the last known status check.
        """
        mgmt = get_llm_server_management_url()
        headers = self._auth_headers()

        # Refresh instance_id if we don't have one
        if not self._instance_id:
            st = await self.status()
            if not st["loaded"]:
                return {"status": "not_loaded", "message": "Model was not loaded."}

        instance_id = self._instance_id or self._settings.lmstudio_model
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{mgmt}/models/unload",
                    headers=headers,
                    json={"instance_id": instance_id},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    body = await resp.json(content_type=None)
                    if resp.status == 200:
                        self._instance_id = None
                        logger.info("Model unloaded: %s", instance_id)
                        return {"status": "unloaded", "instance_id": instance_id}
                    return {"status": "error", "http_status": resp.status, "detail": body}

        except Exception as exc:
            logger.error("Unload failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def ttl_seconds(self) -> int | None:
        """Return the TTL to pass with each request, or None if TTL is disabled."""
        ttl = self._settings.model_idle_ttl_seconds
        return ttl if ttl > 0 else None

    # ── VRAM Monitoring Methods ───────────────────────────────────────────────

    async def log_vram_status(self) -> dict:
        """
        Log current VRAM status to file for post-analysis.

        This is called after each chat request to track model loading patterns
        and detect VRAM spikes (indicates model reloading instead of using
        already-loaded model).
        """
        from core.clihbot_vram_monitor import get_vram_monitor

        monitor = get_vram_monitor()

        # Get model status
        status = await self.status()

        # Log to VRAM monitor
        loaded = status.get("loaded", False)
        models = status.get("models", [])
        monitor.on_status_check(loaded, models)

        # Also get external server status
        is_external = self._external_server_detected
        if is_external:
            monitor.on_status_check(loaded, [self._external_server_url])
            logger.debug(
                "VRAM status: external server=%s, models_loaded=%d, model=%s",
                self._external_server_url, len(models), status.get("configured_model")
            )

        return {
            "loaded": loaded,
            "models": models,
            "external_server_detected": is_external,
        }

    async def check_vram_spike(self, threshold_mb: int = 1024) -> tuple[bool, int]:
        """
        Check if there's been a VRAM spike indicating model reloading.

        A spike is detected if the model was loaded recently (within last few
        requests) but then unloading/loading is happening.

        Args:
            threshold_mb: VRAM threshold in megabytes for spike detection

        Returns:
            Tuple of (has_spike, spike_size_bytes)
        """
        from core.clihbot_vram_monitor import get_vram_monitor

        monitor = get_vram_monitor()
        has_spike, spike_bytes = monitor.has_vram_spike(threshold_bytes=threshold_mb * 1024 * 1024)

        if has_spike:
            logger.warning(
                "VRAM spike detected: %d bytes (%.2f MB) in last 5 seconds",
                spike_bytes, spike_bytes / 1024 / 1024
            )

        return has_spike, spike_bytes

    async def log_model_load(self, model_id: str) -> None:
        """
        Log a model load event with VRAM estimate.
        """
        from core.clihbot_vram_monitor import get_vram_monitor

        monitor = get_vram_monitor()
        estimated_vram = monitor._estimate_model_vram(model_id)
        monitor.on_model_load(model_id)
        logger.info("Model '%s' loaded (estimated VRAM: %.2f GB)", model_id, estimated_vram / 1e9)

    async def log_model_unload(self, model_id: str) -> None:
        """
        Log a model unload event.
        """
        from core.clihbot_vram_monitor import get_vram_monitor

        monitor = get_vram_monitor()
        monitor.on_model_unload(model_id)
        logger.info("Model '%s' unloaded", model_id)

    # ── Model Selection Methods ───────────────────────────────────────────────

    async def get_available_models(self) -> list[dict[str, Any]]:
        """
        Get list of all available models from LMStudio.

        Returns:
            List of dicts with model info (id, size, quantization, etc.)
        """
        models: list[dict[str, Any]] = []

        if self._selector is not None:
            ranked_models = await self._selector.get_available_models()
            models = [
                {
                    "id": m.id,
                    "family": m.family_normalized,
                    "size_gb": m.size_gb,
                    "file_size_gb": m.file_size_gb,
                    "parameter_count_b": m.parameter_count_b,
                    "quantization": m.quantization,
                    "context_window": m.context_window,
                    "description": m.description,
                    "is_loaded": m.is_loaded,
                    "rank_score": m.rank_score,
                    "estimated_vram_gb": m.estimated_vram_gb,
                    "score_breakdown": m.score_breakdown,
                }
                for m in ranked_models
            ]

        if models:
            return models

        # External servers like llama.cpp do not expose LM Studio's model catalog.
        # Fall back to the live /v1/models response so the UI still reflects reality.
        status = await self.status()
        return [
            {
                "id": model_id,
                "family": "unknown",
                "size_gb": None,
                "quantization": "",
                "context_window": 0,
                "description": "Detected from live external server",
                "is_loaded": True,
                "rank_score": 0.0,
            }
            for model_id in status.get("models", [])
        ]

    async def select_best_model(
        self,
        manual_preference: str | None = None,
        override_max_vram_mb: float | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """
        Select the best model based on ranking criteria.

        Args:
            manual_preference: Optional user-selected model ID

        Returns:
            Tuple of (model info dict or None, reason string)
        """
        if self._selector is None:
            return None, "Model selection disabled (set MODEL_RANKING_MODE='auto')"

        model, reason = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._selector.select_best_model(
                manual_preference=manual_preference,
                override_max_vram_mb=override_max_vram_mb,
            )
        )

        if model is None:
            return None, reason

        # Convert to dict
        model_dict = {
            "id": model.id,
            "family": model.family_normalized,
            "size_gb": model.size_gb,
            "file_size_gb": model.file_size_gb,
            "parameter_count_b": model.parameter_count_b,
            "quantization": model.quantization,
            "context_window": model.context_window,
            "rank_score": model.rank_score,
            "estimated_vram_gb": model.estimated_vram_gb,
        }

        return model_dict, reason

    async def _ensure_with_selection(self, manual_preference: str) -> bool:
        """
        Ensure model is loaded using model selection.

        Selects the best model matching the requested family/model.
        """
        # Select the best model
        model_dict, reason = await self.select_best_model(manual_preference)

        if model_dict is None:
            logger.error("Model selection failed: %s", reason)
            return False

        selected_model = model_dict["id"]

        logger.info(
            "Model selection: %s (score: %.1f) - %s",
            selected_model,
            model_dict.get("rank_score", 0),
            reason,
        )

        # Update the configured model
        self._settings.llm_server_model = selected_model

        # Load the selected model
        return await self._load()

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _load(self) -> bool:
        result = await self._load_and_report()
        return result.get("status") == "loaded"

    async def _load_and_report(self) -> dict[str, Any]:
        """
        Load the model via the LLM server Python SDK so that KV cache quantization
        settings (model_k_cache_quant / model_v_cache_quant) are applied.

        The REST API (/api/v1/models/load) does not expose those parameters;
        the SDK uses the internal WebSocket protocol which does.

        Records load events for VRAM spike detection.
        """
        from core.lmstudio_launcher import get_launcher
        from core.model_manager import add_model_load_event

        launcher = get_launcher()
        model = self._settings.lmstudio_model  # Keep backward compatibility
        result = await asyncio.get_event_loop().run_in_executor(
            None, launcher._sdk_load_sync, model
        )
        if result.get("status") == "loaded":
            self._instance_id = model
            add_model_load_event(model)
            logger.info(
                "Model '%s' loaded successfully (instance_id=%s)",
                model,
                self._instance_id
            )
        return result

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self._settings.lmstudio_api_key
        if key and key != "lm-studio":
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def _detect_external_servers(self) -> tuple[bool, str]:
        """
        Detect if any external LLM servers are running on common ports.

        Returns:
            Tuple of (detected_server_url, server_type_description)
            If no external server found, returns (None, None)
        """
        if not self._settings.auto_detect_external_server:
            return None, None

        self._external_server_detected = False
        self._external_server_url = None

        # First, try to detect from the configured base URL
        configured_url = _openai_base_url(self._settings.lmstudio_base_url)
        external_url, server_type = await self._check_port_or_url(configured_url)
        if external_url:
            self._settings.llm_server_url = external_url
            self._external_server_detected = True
            self._external_server_url = external_url
            return external_url, server_type

        # If configured URL didn't match, scan common ports
        logger.info("Scanning for external LLM servers on common ports...")
        for port, description, server_types in COMMON_LLM_PORTS:
            url = f"http://localhost:{port}/v1"
            external_url, server_type = await self._check_port_or_url(url)
            if external_url:
                # Update settings to use the external server
                self._settings.llm_server_url = external_url
                self._external_server_detected = True
                self._external_server_url = external_url
                return external_url, f"{description} ({server_type})"

        return None, None

    async def _check_port_or_url(self, url: str) -> tuple[str | None, str | None]:
        """
        Check if the given URL or port is running an external LLM server.

        Returns:
            Tuple of (url if running, server_type) or (None, None)
        """
        try:
            port_match = re.search(r":(\d+)(?:/|$)", url)
            if not port_match:
                return None, None

            port = int(port_match.group(1))
            port_info = next((p for p in COMMON_LLM_PORTS if p[0] == port), None)
            server_type = port_info[2][0] if port_info else "openai-compatible"

            if await self._is_lmstudio_server(url):
                return None, None

            probe_url = f"{_openai_base_url(url)}/models"
            async with aiohttp.ClientSession() as session:
                async with session.get(probe_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status in (200, 401, 403):
                        logger.info(
                            "External server detected: %s at %s (port %d)",
                            server_type,
                            _openai_base_url(url),
                            port,
                        )
                        return _openai_base_url(url), server_type

            return None, None

        except Exception:
            return None, None

    async def _is_lmstudio_server(self, url: str) -> bool:
        """
        Check if the given URL is an LMStudio server.

        LMStudio has a unique /api/v1/models endpoint (note the /api/ prefix).
        OpenAI-compatible servers like llama.cpp have /v1/models.
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Check for LMStudio's unique /api/v1/models endpoint
                lmstudio_url = f"{_server_root(url)}/api/v1/models"
                async with session.get(lmstudio_url, headers=self._auth_headers(), timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        # LMStudio returns {"data": [...]} format
                        if isinstance(data, dict) and "data" in data:
                            return True
                    return False
        except Exception:
            return False
