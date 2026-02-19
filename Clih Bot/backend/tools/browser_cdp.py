"""
Browser integration via Chrome DevTools Protocol (CDP).

Requires Chrome or Edge launched with:
    --remote-debugging-port=9222

Provides:
  - get_browser_source(url?, include_dom?) → page HTML
  - get_browser_screenshot()               → base64 PNG
  - get_browser_console(levels?, limit?)   → console log entries

Falls back to the extension relay (tools.browser_extension) when CDP is unavailable.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

import aiohttp

from config import get_settings

logger = logging.getLogger(__name__)


class CDPSession:
    """Minimal async CDP client using raw WebSocket + HTTP."""

    def __init__(self, ws_url: str) -> None:
        self._ws_url = ws_url
        self._id = 0

    async def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        import websockets
        self._id += 1
        payload = json.dumps({"id": self._id, "method": method, "params": params or {}})
        async with websockets.connect(self._ws_url, open_timeout=5) as ws:
            await ws.send(payload)
            # Collect messages until we get our response id
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                msg = json.loads(raw)
                if msg.get("id") == self._id:
                    if "error" in msg:
                        raise RuntimeError(f"CDP error: {msg['error']}")
                    return msg.get("result", {})


async def _get_active_tab() -> dict[str, Any] | None:
    """Return the first visible page target from /json."""
    settings = get_settings()
    url = f"{settings.cdp_url}/json"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                tabs = await resp.json(content_type=None)
                for tab in tabs:
                    if tab.get("type") == "page":
                        return tab
    except Exception as exc:
        logger.debug("CDP tab list failed: %s", exc)
    return None


async def _cdp_available() -> bool:
    return await _get_active_tab() is not None


# ── Public tool functions ─────────────────────────────────────────────────────

async def get_browser_source(
    url: str | None = None,
    include_dom: bool = False,
) -> dict[str, Any]:
    """Return HTML source of the current/specified browser tab."""
    # Try CDP first
    tab = await _get_active_tab()
    if tab:
        try:
            ws_url = tab["webSocketDebuggerUrl"]
            session = CDPSession(ws_url)

            if url:
                await session.send("Page.navigate", {"url": url})
                # Brief wait for navigation
                await asyncio.sleep(1.5)

            current_url = tab.get("url", "unknown")

            if include_dom:
                result = await session.send(
                    "Runtime.evaluate",
                    {"expression": "document.documentElement.outerHTML", "returnByValue": True},
                )
                html = result.get("result", {}).get("value", "")
            else:
                result = await session.send(
                    "Runtime.evaluate",
                    {
                        "expression": "(function(){ "
                        "var x = new XMLSerializer(); "
                        "return x.serializeToString(document); })()",
                        "returnByValue": True,
                    },
                )
                html = result.get("result", {}).get("value", "")
                if not html:
                    # Fallback: outerHTML
                    result = await session.send(
                        "Runtime.evaluate",
                        {"expression": "document.documentElement.outerHTML", "returnByValue": True},
                    )
                    html = result.get("result", {}).get("value", "")

            return {
                "url": current_url,
                "source": html,
                "length": len(html),
                "via": "cdp",
            }

        except Exception as exc:
            logger.warning("CDP page source failed: %s — trying extension relay", exc)

    # Fall back to extension relay
    from tools.browser_extension import get_extension_state
    ext = get_extension_state()
    if ext.get("html"):
        return {
            "url": ext.get("url", "unknown"),
            "source": ext["html"],
            "length": len(ext["html"]),
            "via": "extension",
        }

    return {"error": "No browser source available. CDP not reachable and no extension data received."}


async def get_browser_screenshot() -> dict[str, Any]:
    """Capture a screenshot of the active browser tab via CDP."""
    tab = await _get_active_tab()
    if not tab:
        return {"error": "CDP not available. Launch Chrome/Edge with --remote-debugging-port=9222"}

    try:
        ws_url = tab["webSocketDebuggerUrl"]
        session = CDPSession(ws_url)
        result = await session.send("Page.captureScreenshot", {"format": "png", "quality": 85})
        b64_data = result.get("data", "")
        png_bytes = base64.b64decode(b64_data)
        return {
            "image_b64": b64_data,
            "image_bytes": png_bytes,
            "url": tab.get("url", "unknown"),
            "description": f"Browser screenshot ({tab.get('title', '')}) — {len(png_bytes)//1024}KB",
            "via": "cdp",
        }
    except Exception as exc:
        logger.error("CDP screenshot failed: %s", exc)
        return {"error": str(exc)}


async def get_browser_console(
    levels: list[str] | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Return buffered console messages from the active tab.
    CDP does not expose historical logs without a persistent Runtime.consoleAPICalled listener;
    we use Runtime.evaluate to execute console.log interception instead.
    For real-time streaming, the extension relay is more reliable.
    """
    # Check extension relay first (richer data)
    from tools.browser_extension import get_extension_state
    ext = get_extension_state()
    console_logs = ext.get("console_logs", [])

    if console_logs:
        if levels:
            console_logs = [m for m in console_logs if m.get("level") in levels]
        console_logs = console_logs[-limit:]
        return {"messages": console_logs, "count": len(console_logs), "via": "extension"}

    tab = await _get_active_tab()
    if not tab:
        return {"error": "CDP not available and no extension data. Cannot retrieve console logs."}

    try:
        ws_url = tab["webSocketDebuggerUrl"]
        session = CDPSession(ws_url)
        # Read back any messages stored by an injected shim
        result = await session.send(
            "Runtime.evaluate",
            {
                "expression": (
                    "window.__clihbot_console_log__ ? "
                    "JSON.stringify(window.__clihbot_console_log__.slice(-200)) : '[]'"
                ),
                "returnByValue": True,
            },
        )
        raw = result.get("result", {}).get("value", "[]")
        try:
            messages = json.loads(raw)
        except json.JSONDecodeError:
            messages = []

        if levels:
            messages = [m for m in messages if m.get("level") in levels]
        messages = messages[-limit:]
        return {"messages": messages, "count": len(messages), "via": "cdp"}

    except Exception as exc:
        logger.error("CDP console retrieval failed: %s", exc)
        return {"error": str(exc)}
