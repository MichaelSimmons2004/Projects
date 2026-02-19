"""
Context window manager.

Responsibilities:
- Track token usage across the full prompt (system + context snapshot + history).
- Enforce per-tool-result token caps (truncate with notice).
- Summarise and compress old conversation history when approaching the budget.
- Build the final ordered message list to send to the model.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# tiktoken is used as a universal approximation; fall back to char-count if unavailable.
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

except ImportError:
    logger.warning("tiktoken not installed — using character-based token estimation (4 chars = 1 token)")

    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text) // 4)


def count_tokens(text: str) -> int:
    return _count_tokens(text)


def _msg_tokens(msg: dict[str, Any]) -> int:
    """Rough token count for a single message dict."""
    content = msg.get("content") or ""
    if isinstance(content, list):
        # Multimodal message — count text parts only; images are not in the text budget.
        text = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = str(content)
    # Account for role + framing overhead (~4 tokens per message)
    return count_tokens(text) + 4


class ContextManager:
    """
    Maintains the conversation history and builds prompt message lists.

    History is stored as a flat list of message dicts (role/content/...).
    When the history exceeds the soft limit, the oldest non-system messages
    are summarised and replaced with a single compressed summary message.
    """

    def __init__(self) -> None:
        from config import get_settings
        cfg = get_settings()
        self._total_budget = cfg.available_token_budget
        self._history_soft_limit = cfg.history_soft_limit
        self._tool_result_max = cfg.tool_result_max_tokens
        self._history: list[dict[str, Any]] = []
        self._summaries: list[str] = []   # compressed older turns

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, message: dict[str, Any]) -> None:
        """Append a message to the history."""
        self._history.append(message)

    def add_many(self, messages: list[dict[str, Any]]) -> None:
        for m in messages:
            self.add(m)

    def truncate_tool_result(self, content: str, tool_name: str) -> str:
        """Cap a tool result to the configured max tokens, appending a notice if trimmed."""
        tokens = count_tokens(content)
        if tokens <= self._tool_result_max:
            return content
        # Binary search would be more precise; a simple ratio cut is fast enough.
        ratio = self._tool_result_max / tokens
        cut = int(len(content) * ratio * 0.95)
        trimmed = content[:cut]
        trimmed_tokens = count_tokens(trimmed)
        notice = (
            f"\n\n[TRUNCATED: result was {tokens} tokens; "
            f"showing first {trimmed_tokens} tokens of {tokens} total]"
        )
        logger.debug("Tool result '%s' truncated: %d → %d tokens", tool_name, tokens, trimmed_tokens)
        return trimmed + notice

    async def build_messages(
        self,
        system_prompt: str,
        context_snapshot: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Assemble the full message list:
          [system] [summaries-block (if any)] [history...]

        Compresses history if needed before returning.
        """
        await self._maybe_compress()

        messages: list[dict[str, Any]] = []

        # System prompt with optional context snapshot appended
        full_system = system_prompt
        if context_snapshot:
            full_system += f"\n\n---\n## Active Context Snapshot\n{context_snapshot}"
        messages.append({"role": "system", "content": full_system})

        # Prepend any prior summaries as a user/assistant pair
        if self._summaries:
            summary_block = "\n\n---\n".join(self._summaries)
            messages.append({
                "role": "user",
                "content": f"[EARLIER CONVERSATION SUMMARY]\n{summary_block}",
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I have context from our earlier conversation.",
            })

        messages.extend(self._history)
        return messages

    def history_token_count(self) -> int:
        return sum(_msg_tokens(m) for m in self._history)

    def clear_history(self) -> None:
        self._history.clear()
        self._summaries.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _maybe_compress(self) -> None:
        """If history exceeds the soft limit, summarise the oldest half."""
        hist_tokens = self.history_token_count()
        if hist_tokens <= self._history_soft_limit:
            return

        logger.info(
            "History at %d tokens (soft limit %d) — compressing oldest messages.",
            hist_tokens,
            self._history_soft_limit,
        )

        # Take the oldest 50% of messages for summarisation
        split = max(1, len(self._history) // 2)
        to_summarise = self._history[:split]
        self._history = self._history[split:]

        # Convert to plain text for the summariser
        text_block = _messages_to_text(to_summarise)

        try:
            from core.llm_client import LLMClient
            client = LLMClient()
            summary = await client.summarise(text_block)
        except Exception as exc:
            logger.warning("Summarisation failed (%s) — keeping raw text as summary.", exc)
            summary = text_block[:2000] + "\n[truncated for summary]"

        self._summaries.append(summary)
        compressed_tokens = count_tokens(summary)
        logger.info("Compressed %d messages into ~%d-token summary.", split, compressed_tokens)


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        parts.append(f"[{role.upper()}]: {content}")
    return "\n".join(parts)
