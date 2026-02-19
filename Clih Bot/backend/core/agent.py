"""
Agent loop — the heart of CLIHBot.

Orchestrates:
  1. Build the message list (system prompt + context snapshot + history)
  2. Send to LMStudio with tool schemas
  3. If the model returns tool calls → execute them → re-send (repeat up to max_iterations)
  4. Return the final text response

Two public interfaces:
  - chat(user_message, context_snapshot?)  → full text response (non-streaming)
  - stream_chat(user_message, context_snapshot?)  → async generator of text chunks
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from config import get_settings
from core.context_manager import ContextManager
from core.llm_client import LLMClient
from core.tool_runner import run_all_tool_calls
from prompts.system_prompt import build_system_prompt
from tools import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

# Safety cap: prevent runaway tool call loops
_MAX_TOOL_ITERATIONS = 6


class Agent:
    """
    Stateful agent instance.  Typically one per user session.
    The ContextManager holds the conversation history.
    """

    def __init__(self) -> None:
        self._ctx = ContextManager()
        self._llm = LLMClient()
        self._settings = get_settings()

    @property
    def context_manager(self) -> ContextManager:
        return self._ctx

    def reset(self) -> None:
        """Clear conversation history (start fresh)."""
        self._ctx.clear_history()

    # ── Non-streaming ─────────────────────────────────────────────────────────

    async def chat(
        self,
        user_message: str,
        context_snapshot: dict[str, Any] | None = None,
    ) -> str:
        """
        Process a user message, run any tool calls, and return the final response text.
        Appends to the conversation history automatically.
        """
        # Ensure the model is loaded before we try to call it
        from core.model_manager import get_model_manager
        await get_model_manager().ensure_loaded()

        # Add user message to history
        self._ctx.add({"role": "user", "content": user_message})

        system_prompt = build_system_prompt(context_snapshot)
        response_text = await self._run_loop(system_prompt)

        self._ctx.add({"role": "assistant", "content": response_text})
        return response_text

    async def _run_loop(self, system_prompt: str) -> str:
        """Core tool-call loop for non-streaming mode."""
        for iteration in range(_MAX_TOOL_ITERATIONS):
            messages = await self._ctx.build_messages(system_prompt)
            msg = await self._llm.complete(messages, tools=TOOL_SCHEMAS)

            # No tool calls → we have our final text answer
            if not msg.tool_calls:
                return msg.content or ""

            logger.info(
                "Tool call iteration %d/%d: %s",
                iteration + 1, _MAX_TOOL_ITERATIONS,
                [tc.function.name for tc in msg.tool_calls],
            )

            # Append the assistant tool-call message
            self._ctx.add(_assistant_tool_call_msg(msg))

            # Execute all tool calls and append results
            results = await run_all_tool_calls(msg, self._ctx)
            self._ctx.add_many(results)

        # If we hit the iteration cap, ask the model for a final answer without tools
        logger.warning("Hit max tool iterations (%d) — requesting final answer.", _MAX_TOOL_ITERATIONS)
        messages = await self._ctx.build_messages(system_prompt)
        final = await self._llm.complete(messages, tools=None)
        return final.content or ""

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def stream_chat(
        self,
        user_message: str,
        context_snapshot: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming version.  Yields text chunks as they arrive.

        Tool calls that arrive mid-stream are buffered, executed, and then the
        model is re-invoked for a streaming final answer.
        """
        from core.model_manager import get_model_manager
        await get_model_manager().ensure_loaded()

        self._ctx.add({"role": "user", "content": user_message})
        system_prompt = build_system_prompt(context_snapshot)

        async for chunk in self._stream_loop(system_prompt):
            yield chunk

    async def _stream_loop(self, system_prompt: str) -> AsyncIterator[str]:
        """
        Streaming tool-call loop.

        - If the model streams pure text → yield each chunk, collect full text for history.
        - If the model streams tool calls → buffer them, execute, then stream the final answer.
        """
        for iteration in range(_MAX_TOOL_ITERATIONS):
            messages = await self._ctx.build_messages(system_prompt)

            # Buffer tool calls assembled across streamed chunks
            tool_calls_buf: dict[int, dict[str, Any]] = {}
            text_buf = ""

            async for chunk in self._llm.stream(messages, tools=TOOL_SCHEMAS):
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # Accumulate text
                if delta.content:
                    text_buf += delta.content
                    yield delta.content

                # Accumulate tool calls (may come as partial chunks)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_buf:
                            tool_calls_buf[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.id:
                            tool_calls_buf[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_buf[idx]["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_buf[idx]["function"]["arguments"] += tc_delta.function.arguments

            if not tool_calls_buf:
                # Pure text response — save to history and done
                self._ctx.add({"role": "assistant", "content": text_buf})
                return

            # We got tool calls — convert buffer to OpenAI format and execute
            logger.info(
                "Streaming tool call iteration %d/%d: %s",
                iteration + 1, _MAX_TOOL_ITERATIONS,
                [v["function"]["name"] for v in tool_calls_buf.values()],
            )
            tool_calls_list = [tool_calls_buf[i] for i in sorted(tool_calls_buf)]
            self._ctx.add({
                "role": "assistant",
                "content": text_buf or None,
                "tool_calls": tool_calls_list,
            })

            # Build fake tool call objects for the runner
            fake_tool_calls = [_dict_to_tool_call(tc) for tc in tool_calls_list]
            for ftc in fake_tool_calls:
                from core.tool_runner import run_tool_call
                result_msg = await run_tool_call(ftc, self._ctx)
                self._ctx.add(result_msg)

        # Final answer after max iterations
        messages = await self._ctx.build_messages(system_prompt)
        text_buf = ""
        async for chunk in self._llm.stream(messages, tools=None):
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                text_buf += delta.content
                yield delta.content
        self._ctx.add({"role": "assistant", "content": text_buf})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assistant_tool_call_msg(msg: Any) -> dict[str, Any]:
    """Convert a ChatCompletionMessage with tool calls to a serialisable dict."""
    tool_calls = []
    for tc in (msg.tool_calls or []):
        tool_calls.append({
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        })
    return {
        "role": "assistant",
        "content": msg.content,
        "tool_calls": tool_calls,
    }


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: str, function: _FakeFunction) -> None:
        self.id = id
        self.function = function


def _dict_to_tool_call(d: dict[str, Any]) -> _FakeToolCall:
    fn = _FakeFunction(
        name=d["function"]["name"],
        arguments=d["function"]["arguments"],
    )
    return _FakeToolCall(id=d["id"], function=fn)
