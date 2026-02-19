"""
LMStudio Auto-Launcher.

Removes the need for the user to manually open LMStudio, start the server,
or pick a port.  CLIHBot calls ensure_running() at startup, which:

  1. Checks if the LMStudio HTTP server is already listening.
  2. If not, finds the `lms` CLI (ships with LMStudio) and runs:
       lms server start --port <port> --cors
  3. Polls the API until it responds (up to 60 s).
  4. Optionally pre-loads the configured model via `lms load <model>`.

The launcher never closes LMStudio on exit by default — the user may be
using it for other things.  Set LMSTUDIO_STOP_ON_EXIT=true to change that.

Requirements:
  - LMStudio must be installed and have been launched at least once
    (first launch registers the `lms` CLI and downloads runtimes).
  - The model referenced by LMSTUDIO_MODEL must already be downloaded.
    Run `lms ls` to see what's available, or download via LMStudio's UI.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import aiohttp

from config import get_settings

logger = logging.getLogger(__name__)

# ── Known install locations for the `lms` CLI ─────────────────────────────────

_LMS_CANDIDATES_WIN = [
    # Most common: installed via lmstudio.ai installer
    Path.home() / ".lmstudio" / "bin" / "lms.exe",
    Path.home() / "AppData" / "Local" / "LM-Studio" / "bin" / "lms.exe",
    Path.home() / "AppData" / "Local" / "Programs" / "LM-Studio" / "bin" / "lms.exe",
    Path.home() / "AppData" / "Roaming" / "LM-Studio" / "bin" / "lms.exe",
    Path("C:/Program Files/LM-Studio/bin/lms.exe"),
    Path("C:/Program Files (x86)/LM-Studio/bin/lms.exe"),
    # Older layout (pre-1.0)
    Path.home() / "AppData" / "Local" / "LM-Studio" / "lms.exe",
]

_LMS_CANDIDATES_MAC = [
    Path("/Applications/LM Studio.app/Contents/Resources/app/bin/lms"),
    Path.home() / ".lmstudio" / "bin" / "lms",
    Path("/usr/local/bin/lms"),
]

_LMS_CANDIDATES_LINUX = [
    Path.home() / ".lmstudio" / "bin" / "lms",
    Path("/usr/local/bin/lms"),
    Path("/opt/lmstudio/bin/lms"),
]


def _find_lms() -> Path | None:
    """Return the path to the `lms` executable, or None if not found."""
    # Fastest: check PATH first
    in_path = shutil.which("lms")
    if in_path:
        return Path(in_path)

    if sys.platform == "win32":
        candidates = _LMS_CANDIDATES_WIN
    elif sys.platform == "darwin":
        candidates = _LMS_CANDIDATES_MAC
    else:
        candidates = _LMS_CANDIDATES_LINUX

    for p in candidates:
        if p.exists():
            return p

    return None


# ── Launcher class ────────────────────────────────────────────────────────────

class LMStudioLauncher:
    """
    Manages the LMStudio server process lifetime from CLIHBot's perspective.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._lms: Path | None = None          # resolved `lms` path
        self._we_started_it: bool = False      # did we launch it this session?

    # ── Public API ────────────────────────────────────────────────────────────

    async def ensure_running(self) -> dict[str, Any]:
        """
        Make sure LMStudio's HTTP server is up and the model is loaded.
        Returns a status dict describing what happened.
        """
        # Step 1: already up?
        if await self._server_responding():
            logger.info("LMStudio server already running at %s", self._settings.lmstudio_base_url)
            return {"status": "already_running", "url": self._settings.lmstudio_base_url}

        # Step 2: find lms — respect LMSTUDIO_EXECUTABLE override from config
        override = self._settings.lmstudio_executable.strip()
        lms = Path(override) if override else _find_lms()
        if not lms or not lms.exists():
            msg = (
                "LMStudio's `lms` CLI not found. "
                "Please install LMStudio from https://lmstudio.ai and launch it at least once "
                "(first launch registers the CLI and downloads runtimes). "
                "Or set LMSTUDIO_EXECUTABLE in .env to the full path of lms.exe."
            )
            logger.error(msg)
            return {"status": "error", "error": msg}

        self._lms = lms
        logger.info("Found lms at: %s", lms)

        # Step 3: start the server
        port = self._port_from_url()
        started = await self._start_server(port)
        if not started:
            return {
                "status": "error",
                "error": f"lms server start failed. Check that LMStudio is installed correctly.",
            }

        # Step 4: wait until HTTP is ready
        ready = await self._wait_for_server(timeout=90)
        if not ready:
            return {
                "status": "error",
                "error": "LMStudio server did not respond within 90 seconds.",
            }

        self._we_started_it = True
        logger.info("LMStudio server is up at %s", self._settings.lmstudio_base_url)
        return {"status": "started", "url": self._settings.lmstudio_base_url}

    async def load_model(self) -> dict[str, Any]:
        """
        Load the configured model using the LMStudio Python SDK.

        This is the only path that supports Q8 KV cache quantization —
        the REST API and `lms` CLI do not expose those parameters yet.
        Falls back to the REST API if the SDK connection fails.
        """
        model = self._settings.lmstudio_model
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sdk_load_sync, model
        )

    def _sdk_load_sync(self, model: str) -> dict[str, Any]:
        """
        Synchronous model load via the LMStudio Python SDK (runs in a thread).
        Uses the SDK's WebSocket protocol which exposes KV cache quantization.
        """
        try:
            import lmstudio
            from lmstudio import LlmLoadModelConfig
        except ImportError:
            logger.warning("lmstudio SDK not installed; falling back to REST API load.")
            return self._rest_load_sync(model)

        s = self._settings
        quant_k = s.model_k_cache_quant or None
        quant_v = s.model_v_cache_quant or None
        ttl = s.model_idle_ttl_seconds if s.model_idle_ttl_seconds > 0 else None

        cfg = LlmLoadModelConfig(
            context_length=s.context_window_tokens,
            flash_attention=True,          # required for V-cache quantization
            llama_k_cache_quantization_type=quant_k,
            llama_v_cache_quantization_type=quant_v,
            # Keep weights pinned in system RAM after a VRAM unload so the
            # next cold start is RAM→GPU only (PCIe) instead of Disk→RAM→GPU.
            # Makes re-loads 3–5× faster at the cost of ~6 GB of system RAM.
            keep_model_in_memory=True,
            # Memory-mapped file access — faster initial load, OS handles paging.
            try_mmap=True,
        )

        log_parts = [f"context={s.context_window_tokens}"]
        if quant_k:
            log_parts.append(f"k_cache={quant_k}")
        if quant_v:
            log_parts.append(f"v_cache={quant_v}")
        logger.info("Loading '%s' via SDK (%s)", model, ", ".join(log_parts))

        try:
            with lmstudio.Client() as client:
                handle = client.llm.model(model, config=cfg, ttl=ttl)
                info = handle.get_info()
                logger.info("Model '%s' loaded (instance_id=%s).", model, getattr(info, "instance_id", "?"))
                return {"status": "loaded", "model": model, "via": "lmstudio_sdk"}
        except Exception as exc:
            logger.error("SDK load failed for '%s': %s — trying REST fallback.", model, exc)
            return self._rest_load_sync(model)

    def _rest_load_sync(self, model: str) -> dict[str, Any]:
        """Fallback: load via the REST /api/v1/models/load endpoint (no KV quant)."""
        import urllib.request, json as _json
        s = self._settings
        mgmt = s.lmstudio_management_url
        payload = _json.dumps({
            "model": model,
            "context_length": s.context_window_tokens,
            "flash_attention": True,
        }).encode()
        req = urllib.request.Request(
            f"{mgmt}/models/load",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = _json.loads(resp.read())
                logger.info("Model loaded via REST: %s", data.get("status"))
                return {"status": "loaded", "model": model, "via": "rest_api"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def list_available_models(self) -> list[str]:
        """Return models available on disk (not necessarily loaded)."""
        lms = self._lms or _find_lms()
        if not lms:
            return []
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [str(lms), "ls", "--json"],
                    capture_output=True, text=True, timeout=10,
                ),
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                return [
                    m.get("modelKey") or m.get("path") or str(m)
                    for m in (data if isinstance(data, list) else data.get("models", []))
                ]
        except Exception as exc:
            logger.debug("lms ls failed: %s", exc)
        return []

    async def stop_server(self) -> bool:
        """Stop the LMStudio server (only if LMSTUDIO_STOP_ON_EXIT is set)."""
        if not self._settings.lmstudio_stop_on_exit:
            return False
        lms = self._lms or _find_lms()
        if not lms:
            return False
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run([str(lms), "server", "stop"], timeout=10),
            )
            logger.info("LMStudio server stopped.")
            return True
        except Exception as exc:
            logger.warning("Failed to stop LMStudio: %s", exc)
            return False

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _server_responding(self) -> bool:
        """Return True if the LMStudio HTTP API answers a models request."""
        url = f"{self._settings.lmstudio_base_url}/models"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    return resp.status in (200, 401)  # 401 = up but auth required
        except Exception:
            return False

    async def _start_server(self, port: int) -> bool:
        """Run `lms server start --port <port> --cors` in a background process."""
        lms = self._lms
        cmd = [str(lms), "server", "start", "--port", str(port), "--cors"]
        logger.info("Starting LMStudio server: %s", " ".join(cmd))
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(cmd, timeout=15, capture_output=True),
            )
            # lms server start exits quickly (server runs as a background service)
            return True
        except subprocess.TimeoutExpired:
            # Some versions block briefly — a timeout still means it launched
            return True
        except Exception as exc:
            logger.error("lms server start failed: %s", exc)
            return False

    async def _wait_for_server(self, timeout: int = 90) -> bool:
        """Poll the API until it responds or timeout (seconds) is reached."""
        logger.info("Waiting for LMStudio server to come up (up to %ds)...", timeout)
        for elapsed in range(timeout):
            if await self._server_responding():
                logger.info("LMStudio ready after %ds", elapsed)
                return True
            await asyncio.sleep(1)
            if elapsed % 10 == 9:
                logger.info("  ... still waiting (%ds elapsed)", elapsed + 1)
        return False


    def _port_from_url(self) -> int:
        """Extract the port number from lmstudio_base_url."""
        from urllib.parse import urlparse
        parsed = urlparse(self._settings.lmstudio_base_url)
        return parsed.port or 1234


# ── Singleton ─────────────────────────────────────────────────────────────────

_launcher: LMStudioLauncher | None = None


def get_launcher() -> LMStudioLauncher:
    global _launcher
    if _launcher is None:
        _launcher = LMStudioLauncher()
    return _launcher
