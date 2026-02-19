"""
Tool registry.

Each tool is represented by:
  - An OpenAI function-call schema (TOOL_SCHEMAS) — sent to the model.
  - A Python async handler in its respective module.

The TOOL_REGISTRY maps tool name → async callable.
"""
from __future__ import annotations

from typing import Any

# ── OpenAI Function Schemas ───────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "capture_screen",
            "description": (
                "Capture a screenshot of the user's screen. "
                "Use this to visually inspect what the user is working on. "
                "Prefer 'active_window' to minimise context size."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["full", "active_window", "region"],
                        "description": "Capture mode. 'full' = entire desktop, 'active_window' = focused window, 'region' = custom rectangle.",
                    },
                    "region": {
                        "type": "object",
                        "description": "Required when mode='region'. Coordinates in pixels.",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "width": {"type": "integer"},
                            "height": {"type": "integer"},
                        },
                        "required": ["x", "y", "width", "height"],
                    },
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_browser_source",
            "description": (
                "Retrieve the full HTML source of the currently open browser tab (or a specific URL). "
                "Uses Chrome DevTools Protocol if available, otherwise falls back to the browser extension."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Optional. Navigate to this URL before capturing. If omitted, captures the current tab.",
                    },
                    "include_dom": {
                        "type": "boolean",
                        "description": "If true, return the live DOM (document.documentElement.outerHTML) instead of the raw source. Default false.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_browser_screenshot",
            "description": "Take a screenshot of the current browser tab. Useful for visually inspecting web content.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_browser_console",
            "description": "Return recent browser console messages (logs, warnings, errors) from the current tab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "levels": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["log", "info", "warning", "error"]},
                        "description": "Filter by log level. Default: all levels.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to return. Default 50.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_terminal_buffer",
            "description": (
                "Return recent terminal output. "
                "Use this to see what commands the user has run and their output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lines": {
                        "type": "integer",
                        "description": "Number of recent lines to return. Default 100, max 500.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file on the user's filesystem. "
                "Automatically triggers a security scan if the file extension matches the configured list. "
                "Use start_line/end_line to read specific sections of large files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed start line (inclusive). Omit for start of file.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed end line (inclusive). Omit for end of file.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a name pattern or extension within a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '*.py', 'config.*', '**/*.env'.",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to current working directory.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Default 50.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_code",
            "description": (
                "Run a fast regex-based security scan on the provided source code. "
                "Returns a list of findings with severity, line number, pattern name, and explanation. "
                "Use this proactively when the user shares or opens code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Source code to scan.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional filename/extension hint to improve pattern matching.",
                    },
                },
                "required": ["content"],
            },
        },
    },
]

# ── Tool Registry ─────────────────────────────────────────────────────────────
# Populated lazily on first access to avoid import-time side effects.
_REGISTRY: dict[str, Any] | None = None


def get_tool_registry() -> dict[str, Any]:
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    from tools.screen_capture import capture_screen
    from tools.browser_cdp import get_browser_source, get_browser_screenshot, get_browser_console
    from tools.terminal_watcher import get_terminal_buffer
    from tools.file_reader import read_file, search_files
    from tools.code_scanner import scan_code

    _REGISTRY = {
        "capture_screen": capture_screen,
        "get_browser_source": get_browser_source,
        "get_browser_screenshot": get_browser_screenshot,
        "get_browser_console": get_browser_console,
        "get_terminal_buffer": get_terminal_buffer,
        "read_file": read_file,
        "search_files": search_files,
        "scan_code": scan_code,
    }
    return _REGISTRY
