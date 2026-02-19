"""
CLIHBot — Cybersecurity pair-programmer AI agent.
Entry point: starts the API server and background services.
"""
from __future__ import annotations

import asyncio
import logging
import sys

import uvicorn

from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("clihbot")


async def _ensure_lmstudio() -> None:
    """
    Auto-start LMStudio's server and pre-load the model if configured to do so.
    Runs before the HTTP server opens so the first chat request is never blocked
    waiting for LMStudio to come up.
    """
    settings = get_settings()
    if not settings.lmstudio_auto_start:
        return

    from core.lmstudio_launcher import get_launcher
    launcher = get_launcher()

    result = await launcher.ensure_running()
    if result.get("status") == "error":
        logger.warning(
            "LMStudio auto-start failed: %s\n"
            "  You can still start LMStudio manually — CLIHBot will connect when it's up.",
            result.get("error", "unknown error"),
        )
        return

    # If we just started the server (or it was already up) and auto_load is on,
    # kick off model loading in the background so it's warm by the first request.
    if settings.model_auto_load:
        async def _preload() -> None:
            try:
                load_result = await launcher.load_model()
                if load_result.get("status") == "error":
                    logger.warning("Model pre-load: %s", load_result.get("error"))
                else:
                    logger.info("Model pre-load complete: %s", load_result.get("status"))
            except Exception as exc:
                logger.warning("Model pre-load exception: %s", exc)

        asyncio.create_task(_preload(), name="model_preload")


async def _start_background_services() -> None:
    """Start any long-running background tasks (terminal watcher, extension WS server)."""
    settings = get_settings()

    # Terminal watcher
    try:
        from tools.terminal_watcher import TerminalWatcher
        watcher = TerminalWatcher(settings)
        asyncio.create_task(watcher.run(), name="terminal_watcher")
        logger.info("Terminal watcher started (mode=%s)", settings.terminal_mode)
    except Exception as exc:
        logger.warning("Terminal watcher could not start: %s", exc)

    # Browser extension WebSocket relay
    try:
        from tools.browser_extension import ExtensionRelay
        relay = ExtensionRelay(settings)
        asyncio.create_task(relay.serve(), name="extension_relay")
        logger.info("Extension relay started on ws://localhost:%d", settings.extension_ws_port)
    except Exception as exc:
        logger.warning("Extension relay could not start: %s", exc)


def main() -> None:
    settings = get_settings()

    logger.info("Starting CLIHBot API server on %s:%d", settings.api_host, settings.api_port)
    logger.info("LMStudio endpoint: %s  model: %s", settings.lmstudio_base_url, settings.lmstudio_model)
    logger.info("Context window: %d tokens", settings.context_window_tokens)

    # Import app here to avoid circular imports at module load time
    from api.server import create_app

    app = create_app()

    uvicorn_config = uvicorn.Config(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=60,
    )
    server = uvicorn.Server(uvicorn_config)

    async def _run() -> None:
        await _ensure_lmstudio()
        await _start_background_services()
        await server.serve()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
