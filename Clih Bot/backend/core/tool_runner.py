"""
Tool runner — dispatches tool calls from the model to Python handlers.

Handles:
  - JSON argument parsing
  - Calling the correct async tool function
  - Token-capping results via ContextManager
  - Returning results in the format expected by the OpenAI messages API
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

logger = logging.getLogger(__name__)


async def run_tool_call(
    tool_call: ChatCompletionMessageToolCall,
    context_manager: Any,  # core.context_manager.ContextManager
) -> dict[str, Any]:
    """
    Execute a single tool call and return a tool-result message dict.

    The returned dict is ready to be appended to the messages list.
    """
    from tools import get_tool_registry

    name = tool_call.function.name
    raw_args = tool_call.function.arguments or "{}"

    try:
        args: dict[str, Any] = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse tool arguments for %s: %s", name, exc)
        args = {}

    registry = get_tool_registry()
    handler = registry.get(name)

    if handler is None:
        result_str = json.dumps({"error": f"Unknown tool '{name}'"})
        logger.error("Tool not found in registry: %s", name)
    else:
        logger.info("Running tool: %s  args=%s", name, _safe_repr(args))
        try:
            result = await handler(**args)
            # Remove raw bytes from the result before serialising (not JSON-safe)
            result = _strip_binary(result)
            result_str = json.dumps(result, default=str)
        except TypeError as exc:
            result_str = json.dumps({"error": f"Invalid arguments for tool '{name}': {exc}"})
        except Exception as exc:
            logger.exception("Tool %s raised an exception", name)
            result_str = json.dumps({"error": f"Tool error: {exc}"})

    # Apply token cap
    result_str = context_manager.truncate_tool_result(result_str, name)

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result_str,
    }


async def run_all_tool_calls(
    message: ChatCompletionMessage,
    context_manager: Any,
) -> list[dict[str, Any]]:
    """Execute all tool calls in a model message, return list of result messages."""
    if not message.tool_calls:
        return []

    results = []
    for tc in message.tool_calls:
        result_msg = await run_tool_call(tc, context_manager)
        results.append(result_msg)

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_binary(obj: Any) -> Any:
    """Recursively remove bytes values from a dict (e.g. image_bytes)."""
    if isinstance(obj, dict):
        return {k: _strip_binary(v) for k, v in obj.items() if not isinstance(v, (bytes, bytearray))}
    if isinstance(obj, list):
        return [_strip_binary(i) for i in obj]
    return obj


def _safe_repr(args: dict[str, Any]) -> str:
    """Repr args but truncate large values for logging."""
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:80]}{'...' if len(s) > 80 else ''}")
    return "{" + ", ".join(parts) + "}"
