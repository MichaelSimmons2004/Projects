"""
Browser Extension WebSocket Relay Server.

The companion browser extension connects to this server and pushes:
  - Current page URL
  - Full page HTML source
  - Text selections
  - Console log messages (log, warn, error)
  - Navigation events

This module:
  1. Runs a WebSocket server the extension connects to.
  2. Maintains shared in-memory state accessible by the CDP tool and agent.
  3. Exposes get_extension_state() for other tools to read.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# Shared state — last known browser state from extension
_state: dict[str, Any] = {
    "url": None,
    "html": None,
    "title": None,
    "selection": None,
    "console_logs": [],      # list of {level, message, timestamp}
    "connected": False,
}

# A deque so console logs don't grow unbounded
_console_log_buffer: deque[dict[str, Any]] = deque(maxlen=500)


def get_extension_state() -> dict[str, Any]:
    """Return a copy of the current extension state."""
    return {**_state, "console_logs": list(_console_log_buffer)}


class ExtensionRelay:
    """WebSocket server that the browser extension connects to."""

    def __init__(self, settings: Any | None = None) -> None:
        if settings is None:
            from config import get_settings
            settings = get_settings()
        self._port = settings.extension_ws_port
        self._host = "localhost"

    async def serve(self) -> None:
        import websockets

        logger.info("Extension relay listening on ws://%s:%d", self._host, self._port)
        async with websockets.serve(self._handle, self._host, self._port):
            await asyncio.Future()  # run forever

    async def _handle(self, websocket: Any) -> None:
        remote = websocket.remote_address
        logger.info("Browser extension connected from %s", remote)
        _state["connected"] = True

        try:
            async for raw in websocket:
                try:
                    msg: dict[str, Any] = json.loads(raw)
                    await self._process(msg)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON message from extension: %s", raw[:200])
        except Exception as exc:
            logger.warning("Extension connection closed: %s", exc)
        finally:
            _state["connected"] = False
            logger.info("Browser extension disconnected")

    async def _process(self, msg: dict[str, Any]) -> None:
        event = msg.get("type")

        if event == "page_load":
            _state["url"] = msg.get("url")
            _state["title"] = msg.get("title")
            _state["html"] = msg.get("html")
            logger.debug("Extension: page_load url=%s html_len=%d", _state["url"], len(_state["html"] or ""))

        elif event == "html_snapshot":
            _state["html"] = msg.get("html")
            _state["url"] = msg.get("url", _state["url"])

        elif event == "navigation":
            _state["url"] = msg.get("url")
            _state["title"] = msg.get("title")
            _state["html"] = None  # invalidate until next snapshot

        elif event == "selection":
            _state["selection"] = msg.get("text")

        elif event == "console":
            entry = {
                "level": msg.get("level", "log"),
                "message": msg.get("message", ""),
                "timestamp": msg.get("timestamp"),
                "url": msg.get("url"),
            }
            _console_log_buffer.append(entry)

        elif event == "ping":
            pass  # keepalive

        else:
            logger.debug("Unknown extension event type: %s", event)
