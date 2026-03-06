"""
CLIHBot backend entry point.

Starts the FastAPI server plus background services. The canonical UI for this
integration lives in `Clih Bot/main.py`, so this process stays API-only.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

import uvicorn

from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("clihbot")


async def _ensure_llm_server() -> None:
    """Auto-start LM Studio when needed and leave preferred external servers alone."""
    settings = get_settings()

    from core.model_manager import get_model_manager
    manager = get_model_manager()

    if settings.auto_detect_external_server:
        external_url, server_type = await manager._detect_external_servers()
        if external_url and settings.prefer_external_servers:
            logger.info(
                "Detected external server %s (%s) - skipping LM Studio auto-start",
                external_url,
                server_type or "openai-compatible",
            )
            return

    if not settings.lmstudio_auto_start:
        logger.debug("LM Studio auto-start disabled")
        return

    from core.lmstudio_launcher import get_launcher
    launcher = get_launcher()

    result = await launcher.ensure_running()
    if result.get("status") == "error":
        logger.warning(
            "LM Studio auto-start failed: %s. You can still start LM Studio manually; "
            "CLIHBot will connect when it is up.",
            result.get("error", "unknown error"),
        )
        return

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

    try:
        from tools.terminal_watcher import TerminalWatcher
        watcher = TerminalWatcher(settings)
        asyncio.create_task(watcher.run(), name="terminal_watcher")
        logger.info("Terminal watcher started (mode=%s)", settings.terminal_mode)
    except Exception as exc:
        logger.warning("Terminal watcher could not start: %s", exc)

    try:
        from tools.browser_extension import ExtensionRelay
        relay = ExtensionRelay(settings)
        asyncio.create_task(relay.serve(), name="extension_relay")
        logger.info("Extension relay started on ws://localhost:%d", settings.extension_ws_port)
    except Exception as exc:
        logger.warning("Extension relay could not start: %s", exc)


def main() -> None:
    os.environ["CLIHBOT_EMBEDDED_BACKEND"] = "1"
    settings = get_settings()

    logger.info("Starting CLIHBot API server on %s:%d", settings.api_host, settings.api_port)
    logger.info("LLM server endpoint: %s  model: %s", settings.lmstudio_base_url, settings.lmstudio_model)
    logger.info("Context window: %d tokens", settings.context_window_tokens)

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
        await _ensure_llm_server()
        await _start_background_services()
        await server.serve()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
