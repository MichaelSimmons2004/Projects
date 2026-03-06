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
  GET  /model/list           — List all available models
  POST /model/select         — Select model manually
  POST /model/load           — Load selected model
  GET  /model/config         — Get current model config
  POST /model/set-url        — Set external LLM server URL
  GET  /model/status         — Check model status

All JSON responses use snake_case keys.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _free_vram_mb_from_stats(stats: Any) -> int:
    total_mb = int((stats.total_memory or 0) / 1024 / 1024)
    used_mb = int((stats.dedicated_memory_used or 0) / 1024 / 1024)
    return max(total_mb - used_mb, 0) if total_mb else 0

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


class ModelSelectRequest(BaseModel):
    model_id: str | None = None
    manual_preference: str | None = None


class ExternalLLMRequest(BaseModel):
    base_url: str
    api_key: str | None = None

# ── External LLM Server Configuration ─────────────────────────────────────────

# Default external LLM servers (for quick selection)
EXTERNAL_LLM_SERVERS = {
    "llama-cpp-server": {
        "default_url": "http://localhost:1234/v1",
        "description": "llama.cpp server (default port)",
        "tags": ["llama.cpp", "local"],
    },
    "ollama": {
        "default_url": "http://localhost:11434/v1",
        "description": "Ollama API",
        "tags": ["ollama", "local"],
    },
    "vllm": {
        "default_url": "http://localhost:8000/v1",
        "description": "vLLM server",
        "tags": ["vllm", "large-models"],
    },
    "lm-studio": {
        "default_url": "http://localhost:1234/v1",
        "description": "LM Studio API",
        "tags": ["lm-studio", "local"],
    },
}

# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    from core.model_manager import get_model_manager
    _manager = get_model_manager()

    # Initialize model selector if available
    _selector = None
    if _manager._selector:
        _selector = _manager._selector

    # Track the current LLM URL shown in the UI
    _external_llm_url = _manager._settings.llm_server_url

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

    # ── Discovery ────────────────────────────────────────────────────────────
    @app.get("/model/discover")
    async def discover_servers() -> dict[str, Any]:
        """Discover available LLM servers."""
        from core.llm_server_discovery import DEFAULT_SERVERS, check_server
        import asyncio

        tasks = [check_server(server.url) for server in DEFAULT_SERVERS]
        results = await asyncio.gather(*tasks)

        servers = []
        for server, (is_available, _) in zip(DEFAULT_SERVERS, results):
            servers.append({
                "name": server.name,
                "url": server.url,
                "is_available": is_available,
                "server_type": server.server_type,
                "description": server.description,
            })

        return {"servers": servers}

    # ── /model/* ──────────────────────────────────────────────────────────────

    # Note: /model/status is defined later with more comprehensive external server info

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

    # ── /model/selector/* ─────────────────────────────────────────────────────

    @app.get("/model/selector/ranking-config")
    async def ranking_config() -> dict[str, Any]:
        """Get the current model ranking configuration."""
        if _selector is None:
            return {"mode": "disabled"}
        return {
            "mode": _selector._settings.model_ranking_mode,
            "prefer_larger_context": _selector._settings.prefer_larger_context,
            "max_vram_mb": _selector._settings.max_vram_mb,
            "min_context_tokens": _selector._settings.min_context_tokens,
            "raw_weights": {
                "family": _selector._settings.model_family_weight,
                "context": _selector._settings.context_window_weight,
                "quant": _selector._settings.quantization_weight,
                "size": _selector._settings.size_weight,
                "loaded": _selector._settings.loaded_weight,
                "vram": _selector._settings.vram_weight,
                "manual": _selector._settings.manual_weight,
            },
            "weights": {
                "family": _selector._weights.get("family", 0),
                "context": _selector._weights.get("context", 0),
                "quant": _selector._weights.get("quant", 0),
                "size": _selector._weights.get("size", 0),
                "loaded": _selector._weights.get("loaded", 0),
                "vram": _selector._weights.get("vram", 0),
                "manual": _selector._weights.get("manual", 0),
            },
        }

    @app.get("/model/selector/bootstrap")
    async def selector_bootstrap() -> dict[str, Any]:
        """Return lightweight model-selection data in one request."""
        from config import get_settings
        from core.model_manager import get_model_manager
        from core.memory_monitor import get_memory_monitor

        settings = get_settings()
        manager = get_model_manager()
        watcher = get_memory_monitor()
        status, stats, models = await asyncio.gather(
            manager.status(),
            watcher._get_stats(),
            manager.get_available_models(),
        )
        free_mb = _free_vram_mb_from_stats(stats)
        effective_budget_mb = settings.max_vram_mb if settings.max_vram_mb is not None else free_mb

        ranking: dict[str, Any]
        if _selector is None:
            ranking = {"mode": "disabled"}
        else:
            ranking = {
                "mode": _selector._settings.model_ranking_mode,
                "prefer_larger_context": _selector._settings.prefer_larger_context,
                "max_vram_mb": _selector._settings.max_vram_mb,
                "effective_max_vram_mb": effective_budget_mb,
                "min_context_tokens": _selector._settings.min_context_tokens,
                "raw_weights": {
                    "family": _selector._settings.model_family_weight,
                    "context": _selector._settings.context_window_weight,
                    "quant": _selector._settings.quantization_weight,
                    "size": _selector._settings.size_weight,
                    "loaded": _selector._settings.loaded_weight,
                    "vram": _selector._settings.vram_weight,
                    "manual": _selector._settings.manual_weight,
                },
                "weights": {
                    "family": _selector._weights.get("family", 0),
                    "context": _selector._weights.get("context", 0),
                    "quant": _selector._weights.get("quant", 0),
                    "size": _selector._weights.get("size", 0),
                    "loaded": _selector._weights.get("loaded", 0),
                    "vram": _selector._weights.get("vram", 0),
                    "manual": _selector._weights.get("manual", 0),
                },
            }

        total_mb = int((stats.total_memory or 0) / 1024 / 1024)
        used_mb = int((stats.dedicated_memory_used or 0) / 1024 / 1024)
        if _selector is not None:
            reranked = _selector.rank_models(
                models=[_selector._model_cache[m["id"]] for m in models if m.get("id") in _selector._model_cache],
                override_max_vram_mb=effective_budget_mb,
            )
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
                for m in reranked
            ]
        selected_model = models[0] if models else None
        selected_model_estimate = None
        if selected_model:
            from core.lmstudio_launcher import get_launcher
            selected_context = settings.context_window_tokens or selected_model.get("context_window") or 32768
            selected_model_estimate = await get_launcher().estimate_model_memory(
                selected_model["id"],
                int(selected_context),
            )

        return {
            "config": {
                "ranking_mode": settings.model_ranking_mode,
                "prefer_larger_context": settings.prefer_larger_context,
                "max_vram_mb": settings.max_vram_mb,
                "effective_max_vram_mb": effective_budget_mb,
                "min_context_tokens": settings.min_context_tokens,
                "model_family_weight": settings.model_family_weight,
                "context_window_weight": settings.context_window_weight,
                "quantization_weight": settings.quantization_weight,
                "size_weight": settings.size_weight,
                "loaded_weight": settings.loaded_weight,
                "vram_weight": settings.vram_weight,
                "manual_weight": settings.manual_weight,
            },
            "ranking": ranking,
            "status": {
                "server_url": status.get("server_url"),
                "models": status.get("models", []),
                "loaded": status.get("loaded", False),
                "configured_model": status.get("configured_model"),
                "is_lmstudio": status.get("is_lmstudio", False),
                "error": status.get("error"),
                "external_server_detected": manager._external_server_detected,
                "prefer_external": manager._settings.prefer_external_servers,
            },
            "gpu": {
                "gpu_name": stats.gpu_name,
                "total_mb": total_mb,
                "used_mb": used_mb,
                "free_mb": free_mb,
                "temperature": stats.temperature,
                "utilization": stats.utilization,
            },
            "selected_model": selected_model,
            "selected_model_estimate": selected_model_estimate,
            "ranked_models": models[:12],
        }

    @app.post("/model/selector/available")
    async def list_models(request: dict[str, Any] | None = None) -> dict[str, Any]:
        """List all available models with their ranking scores."""
        from core.model_manager import get_model_manager
        from core.memory_monitor import get_memory_monitor
        manager = get_model_manager()
        manual_pref = request.get("manual_preference") if request else None
        models = await manager.get_available_models()
        if manager._selector is not None:
            stats = await get_memory_monitor()._get_stats()
            effective_budget_mb = manager._settings.max_vram_mb
            if effective_budget_mb is None:
                effective_budget_mb = _free_vram_mb_from_stats(stats)
            ranked_models = manager._selector.rank_models(
                models=[manager._selector._model_cache[m["id"]] for m in models if m.get("id") in manager._selector._model_cache],
                manual_preference=manual_pref,
                override_max_vram_mb=effective_budget_mb,
            )
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
        return {
            "count": len(models),
            "models": models,
        }

    @app.post("/model/selector/ranking-config")
    async def update_ranking_config(request: dict[str, Any]) -> dict[str, Any]:
        """Update ranking settings live and rebuild the selector."""
        nonlocal _selector
        from core.model_manager import get_model_manager
        manager = get_model_manager()
        settings = manager._settings

        key_map = {
            "mode": "model_ranking_mode",
            "prefer_larger_context": "prefer_larger_context",
            "max_vram_mb": "max_vram_mb",
            "min_context_tokens": "min_context_tokens",
            "family_weight": "model_family_weight",
            "context_weight": "context_window_weight",
            "quant_weight": "quantization_weight",
            "size_weight": "size_weight",
            "loaded_weight": "loaded_weight",
            "vram_weight": "vram_weight",
            "manual_weight": "manual_weight",
        }

        for request_key, settings_key in key_map.items():
            if request_key in request:
                setattr(settings, settings_key, request[request_key])

        manager.refresh_selector()
        _selector = manager._selector
        return {
            "success": True,
            "message": "Model selection settings updated.",
            "config": await ranking_config(),
        }

    @app.post("/model/selector/best")
    async def select_best(
        request: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        Select the best model based on ranking.

        Accepts optional JSON body:
          {
            "manual_preference": "qwen/qwen3-vl-8b"  # optional: force this model/family
          }
        """
        from core.model_manager import get_model_manager
        from core.memory_monitor import get_memory_monitor
        manager = get_model_manager()
        manual_pref = request.get("manual_preference") if request else None
        effective_budget_mb = manager._settings.max_vram_mb
        if effective_budget_mb is None:
            effective_budget_mb = _free_vram_mb_from_stats(await get_memory_monitor()._get_stats())
        model, reason = await manager.select_best_model(
            manual_preference=manual_pref,
            override_max_vram_mb=effective_budget_mb,
        )
        return {
            "selected_model": model,
            "reason": reason,
            "ranking_mode": _selector._settings.model_ranking_mode if _selector else "disabled",
        }

    @app.post("/model/selector/select-and-load")
    async def select_and_load(
        request: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        Select the best model and load it.

        Accepts optional JSON body:
          {
            "manual_preference": "qwen/qwen3-vl-8b"  # optional: force this model/family
          }
        """
        from core.model_manager import get_model_manager
        from core.memory_monitor import get_memory_monitor
        manager = get_model_manager()
        manual_pref = request.get("manual_preference") if request else None
        result = await manager.ensure_loaded(manual_preference=manual_pref)
        effective_budget_mb = manager._settings.max_vram_mb
        if effective_budget_mb is None:
            effective_budget_mb = _free_vram_mb_from_stats(await get_memory_monitor()._get_stats())
        model, reason = await manager.select_best_model(
            manual_preference=manual_pref,
            override_max_vram_mb=effective_budget_mb,
        )
        return {
            "success": result,
            "selected_model": model,
            "reason": reason,
            "ranking_mode": _selector._settings.model_ranking_mode if _selector else "disabled",
        }

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
        from core.clihbot_vram_monitor import get_vram_monitor

        start_time = time.time()

        try:
            response = await _agent.chat(req.message, req.context_snapshot)
        except Exception as exc:
            logger.exception("Chat error")
            raise HTTPException(status_code=500, detail=str(exc))

        response_time_ms = int((time.time() - start_time) * 1000)

        # Log VRAM status after chat
        monitor = get_vram_monitor()
        monitor.on_chat_request(req.message, response_time_ms)

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

    # ── /model/* ──────────────────────────────────────────────────────────────

    @app.get("/model/list")
    async def list_models() -> dict[str, Any]:
        """List all available models from LMStudio."""
        from core.model_manager import get_model_manager
        manager = get_model_manager()
        models = await manager.get_available_models()
        return {
            "count": len(models),
            "models": models,
        }

    @app.post("/model/select")
    async def select_model(
        request: ModelSelectRequest | None = None,
    ) -> dict[str, Any]:
        """
        Manually select a model.

        Accepts JSON body:
          {
            "model_id": "qwen/qwen3-vl-8b",      # exact model ID to use
            "manual_preference": "llama"          # or just the family name
          }
        """
        from core.model_manager import get_model_manager
        manager = get_model_manager()

        manual_pref = None
        model_id = None

        if request:
            model_id = request.model_id
            manual_pref = request.manual_preference

        if model_id:
            # Set exact model
            manager._settings.llm_server_model = model_id
        elif manual_pref:
            # Use manual preference (will be scored)
            pass

        result = await manager.ensure_loaded(manual_preference=manual_pref)
        model, reason = await manager.select_best_model(manual_preference=manual_pref)

        return {
            "success": result,
            "selected_model": model,
            "reason": reason,
        }

    @app.post("/model/load")
    async def load_model() -> dict[str, Any]:
        """Explicitly load the configured model."""
        from core.model_manager import get_model_manager
        manager = get_model_manager()
        result = await manager.load()
        if result.get("status") == "error":
            raise HTTPException(status_code=503, detail=result)
        return result

    @app.post("/model/start-lmstudio")
    async def start_lmstudio() -> dict[str, Any]:
        """Manually launch LM Studio and load the configured model if enabled."""
        from core.lmstudio_launcher import get_launcher
        from core.model_manager import get_model_manager

        launcher = get_launcher()
        manager = get_model_manager()

        result = await launcher.ensure_running()
        if result.get("status") == "error":
            raise HTTPException(status_code=503, detail=result)

        load_result = None
        if manager._settings.model_auto_load:
            load_result = await launcher.load_model()
            if load_result.get("status") == "error":
                raise HTTPException(status_code=503, detail=load_result)

        manager._settings.llm_server_url = manager._settings.lmstudio_base_url
        manager._external_server_detected = False
        manager._external_server_url = None

        return {
            "success": True,
            "server": result,
            "model": load_result,
            "message": "LM Studio launch requested.",
        }

    @app.post("/model/unload")
    async def unload_model() -> dict[str, Any]:
        """Unload the model to free VRAM."""
        from core.model_manager import get_model_manager
        manager = get_model_manager()
        result = await manager.unload()
        if result.get("status") == "error":
            raise HTTPException(status_code=503, detail=result)
        return result

    @app.get("/model/config")
    async def get_model_config() -> dict[str, Any]:
        """Get current model configuration."""
        from config import get_settings
        cfg = get_settings()
        return {
            "lmstudio_base_url": cfg.lmstudio_base_url,
            "lmstudio_model": cfg.lmstudio_model,
            "context_window_tokens": cfg.context_window_tokens,
            "lmstudio_auto_start": cfg.lmstudio_auto_start,
            "model_auto_load": cfg.model_auto_load,
            "auto_detect_external_server": cfg.auto_detect_external_server,
            "prefer_external_servers": cfg.prefer_external_servers,
            "model_idle_ttl_seconds": cfg.model_idle_ttl_seconds,
            "model_k_cache_quant": cfg.model_k_cache_quant,
            "model_v_cache_quant": cfg.model_v_cache_quant,
            "ranking_mode": cfg.model_ranking_mode,
            "prefer_larger_context": cfg.prefer_larger_context,
            "max_vram_mb": cfg.max_vram_mb,
            "min_context_tokens": cfg.min_context_tokens,
            "model_family_weight": cfg.model_family_weight,
            "context_window_weight": cfg.context_window_weight,
            "quantization_weight": cfg.quantization_weight,
            "size_weight": cfg.size_weight,
            "loaded_weight": cfg.loaded_weight,
            "vram_weight": cfg.vram_weight,
            "manual_weight": cfg.manual_weight,
        }

    @app.post("/model/set-url")
    async def set_llm_url(request: ExternalLLMRequest) -> dict[str, Any]:
        nonlocal _external_llm_url
        """
        Set the LLM server URL (for external servers like llama.cpp, Ollama, etc.).

        This is useful when you want to attach CLIHBot to a model server
        that's already running outside of LMStudio.
        """
        from config import get_settings

        settings = get_settings()
        settings.llm_server_url = request.base_url
        settings.llm_server_api_key = request.api_key or settings.llm_server_api_key

        # Update the manager's settings too
        from core.model_manager import get_model_manager
        manager = get_model_manager()
        manager._settings.llm_server_url = request.base_url
        manager._settings.llm_server_api_key = request.api_key or manager._settings.llm_server_api_key
        manager._external_server_detected = False
        manager._external_server_url = None
        _external_llm_url = request.base_url

        return {
            "success": True,
            "url": request.base_url,
            "message": f"LLM server URL set to {request.base_url}",
        }

    @app.post("/model/config/update")
    async def update_model_config(request: dict[str, Any]) -> dict[str, Any]:
        """
        Update model configuration without restart.
        Note: This updates the in-memory settings only; changes persist until restart.
        """
        nonlocal _selector
        from config import get_settings

        settings = get_settings()

        # Update based on request
        updates = {}
        for key, value in request.items():
            if hasattr(settings, key):
                updates[key] = value
                setattr(settings, key, value)

        manager = get_model_manager()
        if hasattr(manager, "_settings"):
            manager._settings = settings
            manager.refresh_selector()
            _selector = manager._selector
        return {
            "success": True,
            "message": "Configuration updated (changes persist until restart)",
            "note": "Some settings (like LMSTUDIO_AUTO_START) require a restart to take full effect.",
        }

    @app.get("/model/status")
    async def get_model_status() -> dict[str, Any]:
        """Check current model status and available models."""
        from core.model_manager import get_model_manager, get_model_load_history
        manager = get_model_manager()
        status = await manager.status()

        # Add external server detection info
        return {
            **status,
            "external_llm_url": _external_llm_url,
            "current_llm_url": manager._settings.llm_server_url,
            "external_server_detected": manager._external_server_detected if hasattr(manager, '_external_server_detected') else False,
            "model_load_history": get_model_load_history(),
            "ranking_mode": _selector._settings.model_ranking_mode if _selector else "disabled",
            "configured_model": _manager._settings.lmstudio_model,
            "auto_detect_external": manager._settings.auto_detect_external_server if hasattr(manager, '_settings') else False,
            "prefer_external": manager._settings.prefer_external_servers if hasattr(manager, '_settings') else False,
        }

    @app.post("/model/reset-load-history")
    async def reset_load_history() -> dict[str, Any]:
        """Clear the model load history."""
        from core.model_manager import clear_model_load_history
        clear_model_load_history()
        return {"status": "ok", "message": "Load history cleared"}

    @app.get("/model/vram-stats")
    async def get_vram_stats() -> dict[str, Any]:
        """
        Get VRAM statistics and history.
        Useful for debugging VRAM oscillation issues.
        """
        from core.model_manager import get_model_load_history
        from core.clihbot_vram_monitor import get_vram_monitor

        monitor = get_vram_monitor()
        load_history = get_model_load_history()

        return {
            "timestamp": datetime.now().isoformat(),
            "load_history": load_history,
            "chat_request_count": monitor._chat_request_count,
        }

    @app.get("/model/gpu-memory")
    async def get_gpu_memory() -> dict[str, Any]:
        """Get current GPU memory information for budget decisions."""
        from core.memory_monitor import get_memory_monitor

        watcher = get_memory_monitor()
        stats = await watcher._get_stats()
        total_mb = int((stats.total_memory or 0) / 1024 / 1024)
        used_mb = int((stats.dedicated_memory_used or 0) / 1024 / 1024)
        free_mb = max(total_mb - used_mb, 0) if total_mb else 0
        return {
            "gpu_name": stats.gpu_name,
            "total_mb": total_mb,
            "used_mb": used_mb,
            "free_mb": free_mb,
            "temperature": stats.temperature,
            "utilization": stats.utilization,
        }

    @app.post("/model/check-vram-spike")
    async def check_vram_spike(request: dict[str, Any] = None) -> dict[str, Any]:
        """
        Check for VRAM spike indicating model reloading.
        """
        from core.model_manager import get_model_manager

        manager = get_model_manager()
        threshold_mb = request.get("threshold_mb", 1024) if request else 1024

        has_spike, spike_bytes = await manager.check_vram_spike(threshold_mb=threshold_mb)

        return {
            "has_spike": has_spike,
            "spike_bytes": spike_bytes,
            "spike_mb": spike_bytes / 1024 / 1024 if spike_bytes else 0,
            "threshold_mb": threshold_mb,
        }

    @app.get("/model/external-servers")
    async def get_external_servers() -> dict[str, Any]:
        """
        Get list of known external LLM server options and current status.
        """
        from core.model_manager import COMMON_LLM_PORTS

        servers = []
        for port, description, server_types in COMMON_LLM_PORTS:
            servers.append({
                "port": port,
                "description": description,
                "server_types": server_types,
                "url_template": f"http://localhost:{port}/v1",
            })

        # Get current model status
        from core.model_manager import get_model_manager
        manager = get_model_manager()
        status = await manager.status()

        return {
            "servers": servers,
            "current_url": _external_llm_url,
            "external_server_detected": manager._external_server_detected if hasattr(manager, '_external_server_detected') else False,
            "external_server_url": manager._external_server_url if hasattr(manager, '_external_server_url') else None,
            "config": {
                "auto_detect_external": manager._settings.auto_detect_external_server,
                "prefer_external": manager._settings.prefer_external_servers,
            },
            "status": status,
        }

    @app.get("/model/external-server-presets")
    async def get_external_server_presets() -> dict[str, Any]:
        """Get static external LLM server presets."""
        return {
            "servers": EXTERNAL_LLM_SERVERS,
            "current_url": _external_llm_url,
        }

    # ── DELETE /history ───────────────────────────────────────────────────────

    @app.delete("/history")
    async def clear_history() -> dict[str, str]:
        _agent.reset()
        return {"status": "cleared"}

    return app
