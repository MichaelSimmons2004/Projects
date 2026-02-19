"""
LMStudio Model Lifecycle Manager.

Two complementary mechanisms:

1. TTL per-request (passive, zero-config):
   Every completion request includes a "ttl" field (seconds).
   LMStudio resets the idle countdown on each request and automatically
   unloads the model when the timer expires with no traffic.
   This works even if JIT loading is disabled in LMStudio settings.

2. Explicit load on demand (active):
   Before the first chat request, ensure_loaded() checks whether the
   configured model is already in memory.  If not, it calls the
   LMStudio REST API (POST /api/v1/models/load) to load it immediately
   rather than waiting for the slow first-token JIT load.

3. Explicit unload (user-initiated):
   unload() calls POST /api/v1/models/unload.
   Exposed via POST /model/unload on the FastAPI server so the GUI can
   add a "Free VRAM" button.

Usage:
    manager = get_model_manager()
    await manager.ensure_loaded()   # call before chat
    # ... chat happens, ttl resets on each request ...
    await manager.unload()          # optional: immediate VRAM free
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from config import get_settings

logger = logging.getLogger(__name__)

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

    # ── Public interface ──────────────────────────────────────────────────────

    async def ensure_loaded(self) -> bool:
        """
        Guarantee the configured model is in memory before a chat request.
        Returns True if the model is loaded (or already was), False on failure.
        Does nothing if model_auto_load is disabled in config.
        """
        if not self._settings.model_auto_load:
            return True

        async with self._load_lock:
            status = await self.status()
            if status["loaded"]:
                return True

            logger.info(
                "Model '%s' not loaded — loading now...", self._settings.lmstudio_model
            )
            return await self._load()

    async def status(self) -> dict[str, Any]:
        """
        Query LMStudio for the list of currently loaded models.
        Returns a dict describing whether our configured model is loaded.
        """
        mgmt = self._settings.lmstudio_management_url
        headers = self._auth_headers()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{mgmt}/models",
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
            )
            if loaded and model_ids:
                self._instance_id = model_ids[0]

            return {
                "loaded": loaded,
                "models": model_ids,
                "instance_id": self._instance_id,
                "configured_model": self._settings.lmstudio_model,
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
        mgmt = self._settings.lmstudio_management_url
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _load(self) -> bool:
        result = await self._load_and_report()
        return result.get("status") == "loaded"

    async def _load_and_report(self) -> dict[str, Any]:
        """
        Load the model via the LMStudio Python SDK so that KV cache quantization
        settings (model_k_cache_quant / model_v_cache_quant) are applied.

        The REST API (/api/v1/models/load) does not expose those parameters;
        the SDK uses the internal WebSocket protocol which does.
        """
        from core.lmstudio_launcher import get_launcher
        launcher = get_launcher()
        result = await asyncio.get_event_loop().run_in_executor(
            None, launcher._sdk_load_sync, self._settings.lmstudio_model
        )
        if result.get("status") == "loaded":
            self._instance_id = self._settings.lmstudio_model
        return result

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self._settings.lmstudio_api_key
        if key and key != "lm-studio":
            headers["Authorization"] = f"Bearer {key}"
        return headers
