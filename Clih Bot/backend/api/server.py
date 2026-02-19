"""
FastAPI server — the interface between the GUI and the CLIHBot agent.

Endpoints:
  POST /chat                 — Non-streaming chat
  WS   /stream               — Streaming chat (Server-Sent Events via WebSocket)
  GET  /context/screen       — Latest screenshot (base64 PNG)
  GET  /context/terminal     — Recent terminal buffer
  GET  /context/browser      — Current browser state (URL + source snippet)
  POST /action/{tool_name}   — User-initiated tool invocation
  POST /scan                 — Scan arbitrary code for security issues
  DELETE /history            — Clear conversation history
  GET  /health               — Liveness + LMStudio reachability

All JSON responses use snake_case keys.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    context_snapshot: dict[str, Any] | None = None

class ChatResponse(BaseModel):
    response: str
    history_tokens: int

class ScanRequest(BaseModel):
    content: str
    filename: str | None = None

class ActionRequest(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)

# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="CLIHBot API",
        description="Cybersecurity pair-programmer AI agent backend",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # GUI team can restrict this in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Single shared agent instance (one conversation per server instance)
    # For multi-session support, this would be keyed by session ID
    from core.agent import Agent
    _agent = Agent()

    # ── /health ───────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict[str, Any]:
        from core.llm_client import LLMClient
        from config import get_settings
        cfg = get_settings()
        detail = await LLMClient().health_check_with_model()
        return {
            "status": "ok",
            "model": cfg.lmstudio_model,
            "context_window_tokens": cfg.context_window_tokens,
            "idle_ttl_seconds": cfg.model_idle_ttl_seconds,
            **detail,
        }

    # ── /model/* ──────────────────────────────────────────────────────────────

    @app.get("/model/status")
    async def model_status() -> dict[str, Any]:
        """Check whether the configured model is currently loaded in LMStudio."""
        from core.model_manager import get_model_manager
        return await get_model_manager().status()

    @app.post("/model/load")
    async def model_load() -> dict[str, Any]:
        """Explicitly load the configured model into LMStudio."""
        from core.model_manager import get_model_manager
        result = await get_model_manager().load()
        if result.get("status") == "error":
            raise HTTPException(status_code=503, detail=result)
        return result

    @app.post("/model/unload")
    async def model_unload() -> dict[str, Any]:
        """Unload the model immediately to free VRAM."""
        from core.model_manager import get_model_manager
        result = await get_model_manager().unload()
        if result.get("status") == "error":
            raise HTTPException(status_code=503, detail=result)
        return result

    # ── POST /chat/stream  (Server-Sent Events) ───────────────────────────────

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        """
        Streaming chat via Server-Sent Events.

        Each event:  data: {"chunk": "..."}\n\n
        Final event: data: {"done": true, "history_tokens": N}\n\n
        Error event: data: {"error": "..."}\n\n
        """
        from fastapi.responses import StreamingResponse as _SR

        async def _generate():
            try:
                async for chunk in _agent.stream_chat(req.message, req.context_snapshot):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                yield f"data: {json.dumps({'done': True, 'history_tokens': _agent.context_manager.history_token_count()})}\n\n"
            except Exception as exc:
                logger.exception("Streaming error")
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return _SR(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable nginx buffering if proxied
            },
        )

    # ── POST /chat ────────────────────────────────────────────────────────────

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        try:
            response = await _agent.chat(req.message, req.context_snapshot)
        except Exception as exc:
            logger.exception("Chat error")
            raise HTTPException(status_code=500, detail=str(exc))
        return ChatResponse(
            response=response,
            history_tokens=_agent.context_manager.history_token_count(),
        )

    # ── WS /stream ────────────────────────────────────────────────────────────

    @app.websocket("/stream")
    async def stream(ws: WebSocket) -> None:
        """
        WebSocket streaming chat.

        Client sends:  {"message": "...", "context_snapshot": {...}}
        Server sends:  {"type": "chunk", "content": "..."}   (repeated)
                       {"type": "done", "history_tokens": N}
                       {"type": "error", "detail": "..."}
        """
        await ws.accept()
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                    message = data.get("message", "")
                    snapshot = data.get("context_snapshot")
                except (json.JSONDecodeError, KeyError) as exc:
                    await ws.send_json({"type": "error", "detail": f"Invalid request: {exc}"})
                    continue

                if not message.strip():
                    await ws.send_json({"type": "error", "detail": "Empty message"})
                    continue

                try:
                    async for chunk in _agent.stream_chat(message, snapshot):
                        await ws.send_json({"type": "chunk", "content": chunk})
                    await ws.send_json({
                        "type": "done",
                        "history_tokens": _agent.context_manager.history_token_count(),
                    })
                except Exception as exc:
                    logger.exception("Streaming error")
                    await ws.send_json({"type": "error", "detail": str(exc)})

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")

    # ── GET /context/screen ───────────────────────────────────────────────────

    @app.get("/context/screen")
    async def context_screen(
        mode: str = "active_window",
        x: int | None = None,
        y: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        from tools.screen_capture import capture_screen
        region = None
        if mode == "region":
            if None in (x, y, width, height):
                raise HTTPException(status_code=400, detail="Region mode requires x, y, width, height query params")
            region = {"x": x, "y": y, "width": width, "height": height}
        result = await capture_screen(mode=mode, region=region)
        if "error" in result:
            raise HTTPException(status_code=503, detail=result["error"])
        # Strip raw bytes before returning JSON
        result.pop("image_bytes", None)
        return result

    # ── GET /context/terminal ─────────────────────────────────────────────────

    @app.get("/context/terminal")
    async def context_terminal(lines: int = 100) -> dict[str, Any]:
        from tools.terminal_watcher import get_terminal_buffer
        return await get_terminal_buffer(lines=lines)

    # ── GET /context/browser ──────────────────────────────────────────────────

    @app.get("/context/browser")
    async def context_browser(include_dom: bool = False) -> dict[str, Any]:
        from tools.browser_cdp import get_browser_source
        result = await get_browser_source(include_dom=include_dom)
        if "error" in result:
            raise HTTPException(status_code=503, detail=result["error"])
        # Truncate large HTML for the REST endpoint; full source available via agent tool
        src = result.get("source", "")
        if len(src) > 50_000:
            result["source"] = src[:50_000] + "\n[TRUNCATED]"
            result["truncated"] = True
        return result

    # ── POST /action/{tool_name} ──────────────────────────────────────────────

    @app.post("/action/{tool_name}")
    async def action(tool_name: str, req: ActionRequest) -> dict[str, Any]:
        """
        User-initiated direct tool invocation (without going through the agent loop).
        Useful for the GUI to trigger one-off tool calls (e.g. scan a file, take a screenshot).
        """
        from tools import get_tool_registry
        from core.context_manager import ContextManager

        registry = get_tool_registry()
        handler = registry.get(tool_name)
        if not handler:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")

        ctx = ContextManager()
        try:
            result = await handler(**req.args)
            if isinstance(result, dict):
                result.pop("image_bytes", None)
            result_str = json.dumps(result, default=str)
            result_str = ctx.truncate_tool_result(result_str, tool_name)
            return json.loads(result_str)
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid arguments: {exc}")
        except Exception as exc:
            logger.exception("Action error for tool %s", tool_name)
            raise HTTPException(status_code=500, detail=str(exc))

    # ── POST /scan ────────────────────────────────────────────────────────────

    @app.post("/scan")
    async def scan(req: ScanRequest) -> dict[str, Any]:
        from tools.code_scanner import scan_code
        return await scan_code(content=req.content, filename=req.filename)

    # ── DELETE /history ───────────────────────────────────────────────────────

    @app.delete("/history")
    async def clear_history() -> dict[str, str]:
        _agent.reset()
        return {"status": "cleared"}

    return app
