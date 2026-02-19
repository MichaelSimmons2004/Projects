"""
Async OpenAI-compatible LLM client for LMStudio.
Handles streaming, tool calling, and vision (image) messages.
"""
from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessage

from config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin async wrapper around the OpenAI client pointed at LMStudio."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(
            base_url=settings.lmstudio_base_url,
            api_key=settings.lmstudio_api_key,
        )
        self._model = settings.lmstudio_model
        self._vision = settings.model_vision_capable

    # ── Message construction helpers ──────────────────────────────────────────

    @staticmethod
    def user_text(content: str) -> dict[str, Any]:
        return {"role": "user", "content": content}

    @staticmethod
    def assistant_text(content: str) -> dict[str, Any]:
        return {"role": "assistant", "content": content}

    @staticmethod
    def system_text(content: str) -> dict[str, Any]:
        return {"role": "system", "content": content}

    @staticmethod
    def tool_result(tool_call_id: str, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    def user_with_image(self, text: str, image_bytes: bytes, mime: str = "image/png") -> dict[str, Any]:
        """Build a multimodal user message (text + image).  Only used if MODEL_VISION_CAPABLE=true."""
        if not self._vision:
            return self.user_text(text)
        b64 = base64.b64encode(image_bytes).decode()
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }

    # ── Core API calls ────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> ChatCompletionMessage:
        """Non-streaming completion.  Returns the full assistant message."""
        settings = get_settings()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or settings.response_reserve_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Pass TTL so LMStudio resets the idle-unload countdown on each request.
        _inject_ttl(kwargs, settings.model_idle_ttl_seconds)

        response = await self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        logger.debug("LLM response: finish_reason=%s tool_calls=%s", response.choices[0].finish_reason, bool(msg.tool_calls))
        return msg

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Streaming completion.  Yields raw ChatCompletionChunk objects."""
        settings = get_settings()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or settings.response_reserve_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        _inject_ttl(kwargs, settings.model_idle_ttl_seconds)

        async with await self._client.chat.completions.create(**kwargs) as response:
            async for chunk in response:
                yield chunk

    async def health_check(self) -> bool:
        """Return True if LMStudio is reachable and lists at least one model."""
        try:
            models = await self._client.models.list()
            return len(models.data) > 0
        except Exception as exc:
            logger.warning("LMStudio health check failed: %s", exc)
            return False

    async def health_check_with_model(self) -> dict[str, Any]:
        """Extended health check that also reports model load state."""
        from core.model_manager import get_model_manager
        mgr = get_model_manager()
        status = await mgr.status()
        reachable = not bool(status.get("error"))
        return {
            "lmstudio_reachable": reachable,
            "model_loaded": status.get("loaded", False),
            "loaded_models": status.get("models", []),
            "configured_model": status.get("configured_model"),
        }

    async def summarise(self, text: str, max_words: int = 300) -> str:
        """Ask the model to summarise a block of text — used for context compression."""
        prompt = (
            f"Summarise the following conversation or content in at most {max_words} words. "
            "Preserve key facts, code snippets, security findings, and decisions made.\n\n"
            f"{text}"
        )
        msg = await self.complete(
            [self.user_text(prompt)],
            temperature=0.1,
            max_tokens=max_words * 2,
        )
        return msg.content or ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inject_ttl(kwargs: dict[str, Any], ttl_seconds: int) -> None:
    """
    Attach the LMStudio-specific 'ttl' field to a completion request.
    LMStudio resets the model's idle-unload countdown each time it receives
    a request carrying this field.  The OpenAI client passes extra kwargs
    through as additional JSON body fields, so this is transparent.
    If ttl_seconds == 0 the field is omitted (LMStudio uses its own default).
    """
    if ttl_seconds > 0:
        kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "ttl": ttl_seconds}
