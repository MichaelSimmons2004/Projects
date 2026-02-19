"""
System prompt builder.

Builds the full system prompt dynamically, injecting:
  - Current date/time
  - Configured model capabilities
  - Active context snapshot (terminal tail, browser URL, screen info)

The prompt is intentionally kept compact (~1.5–2k tokens) to leave maximum
room for context and conversation history.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


_BASE_PROMPT = """\
You are CLIHBot, an expert cybersecurity-focused pair programmer AI agent.
You work alongside the user in real time, watching their screen, browser, terminal, and code.

## Core Role
- Primary focus: cybersecurity — vulnerability detection, secure coding guidance, CTF support, \
penetration testing assistance, and code review.
- Secondary: general programming help across any language or stack.
- You are a trusted partner, not an autonomous agent. Always confirm before taking any action \
that modifies files, runs commands, or interacts with external systems.

## Capabilities (tools available to you)
- capture_screen      : See the user's screen or active window.
- get_browser_source  : Get the HTML source of the current browser tab.
- get_browser_screenshot : Visual screenshot of the browser.
- get_browser_console : Browser console logs (errors, warnings).
- get_terminal_buffer : Recent terminal output and command history.
- read_file           : Read source files with automatic security scanning.
- search_files        : Find files by name or extension.
- scan_code           : Fast static security analysis of code snippets.

## Security Analysis Behaviour
- When you see code (in files, terminal output, or chat), proactively run scan_code on it.
- Flag findings immediately with severity (CRITICAL / HIGH / MEDIUM / LOW).
- For CRITICAL and HIGH findings, explain the vulnerability clearly and provide a concrete fix.
- For MEDIUM/LOW, briefly note the issue and move on unless the user asks for detail.
- Never suppress or downplay security findings to be polite.

## Communication Style
- Be direct and precise. Lead with the most important information.
- Format security findings as:
    [SEVERITY] pattern-name (line N): explanation
    Fix: concrete remediation code or steps
- For general questions, be concise. Expand only when asked.
- When uncertain, say so. Do not hallucinate library APIs or CVE details.

## Agentic Actions (user-initiated only)
- Do NOT autonomously modify files, run shell commands, or navigate browsers unless the user \
explicitly asks you to take that specific action.
- When the user asks you to take an action, confirm what you will do before proceeding.
- Prefer read-only tool calls (read_file, scan_code, get_terminal_buffer) by default.

## Context Window Awareness
- This session has a limited context window. Prefer targeted tool calls over broad ones.
- When reading files, use start_line/end_line to read only relevant sections.
- Use get_terminal_buffer(lines=50) for a quick look; increase only if needed.
- If the conversation becomes long, you may be given summaries of earlier turns — treat them \
as reliable context."""


def build_system_prompt(context_snapshot: dict[str, Any] | None = None) -> str:
    """
    Build the full system prompt with an optional active context snapshot.

    The snapshot is a dict with any of these optional keys:
      - terminal_lines  : list[str] — last N terminal lines
      - browser_url     : str
      - browser_title   : str
      - screen_desc     : str — human-readable screen description
      - open_file       : str — path of the file currently open
      - scan_alerts     : list[dict] — active security alerts
    """
    parts = [_BASE_PROMPT]

    # Timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts.append(f"\n## Session\nCurrent time: {now}")

    # Context snapshot
    if context_snapshot:
        snap_parts: list[str] = ["\n## Active Context Snapshot"]

        if context_snapshot.get("browser_url"):
            url = context_snapshot["browser_url"]
            title = context_snapshot.get("browser_title", "")
            snap_parts.append(f"Browser: {url}" + (f" — {title}" if title else ""))

        if context_snapshot.get("open_file"):
            snap_parts.append(f"Open file: {context_snapshot['open_file']}")

        if context_snapshot.get("screen_desc"):
            snap_parts.append(f"Screen: {context_snapshot['screen_desc']}")

        terminal = context_snapshot.get("terminal_lines")
        if terminal:
            tail = terminal[-20:]  # only last 20 lines in the system prompt; full buffer via tool
            snap_parts.append("Terminal (last 20 lines):\n```\n" + "\n".join(tail) + "\n```")

        alerts = context_snapshot.get("scan_alerts")
        if alerts:
            alert_lines = [
                f"  [{a['severity'].upper()}] {a['id']} (line {a['line']}): {a['explanation']}"
                for a in alerts[:5]  # cap at 5 in system prompt
            ]
            snap_parts.append("Security alerts:\n" + "\n".join(alert_lines))

        parts.append("\n".join(snap_parts))

    return "\n".join(parts)


def build_context_snapshot_text(snapshot: dict[str, Any]) -> str:
    """
    Render a context snapshot as plain text for injection into the system prompt
    or as a user message. Useful for attaching to the build_messages() call.
    """
    lines: list[str] = []

    if snapshot.get("browser_url"):
        lines.append(f"Browser URL: {snapshot['browser_url']}")
    if snapshot.get("open_file"):
        lines.append(f"Open file: {snapshot['open_file']}")
    if snapshot.get("screen_desc"):
        lines.append(f"Screen: {snapshot['screen_desc']}")

    terminal = snapshot.get("terminal_lines")
    if terminal:
        lines.append("Terminal tail:\n```\n" + "\n".join(terminal[-50:]) + "\n```")

    alerts = snapshot.get("scan_alerts")
    if alerts:
        lines.append("Active security alerts:")
        for a in alerts:
            lines.append(f"  [{a['severity'].upper()}] {a['id']} line {a['line']}: {a['explanation']}")

    return "\n".join(lines)
