"""
File reader tool.

Reads files from the local filesystem with optional line range limits.
Auto-triggers a security scan if the file extension matches the configured set.
"""
from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Hard cap: never read more than this many lines in a single call
_MAX_LINES_PER_READ = 1000


async def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    """Read a file, optionally bounded to a line range."""
    from config import get_settings

    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        return {"error": f"File not found: {resolved}"}
    if not resolved.is_file():
        return {"error": f"Path is not a file: {resolved}"}

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
    except PermissionError:
        return {"error": f"Permission denied: {resolved}"}
    except Exception as exc:
        return {"error": str(exc)}

    total_lines = len(all_lines)

    # Apply line range
    start = max(1, start_line or 1)
    end = min(total_lines, end_line or total_lines)

    # Enforce per-call line cap
    if (end - start + 1) > _MAX_LINES_PER_READ:
        end = start + _MAX_LINES_PER_READ - 1
        truncated = True
    else:
        truncated = False

    selected = all_lines[start - 1 : end]
    content = "".join(selected)

    # Token-cap via context manager
    settings = get_settings()
    from core.context_manager import count_tokens
    if count_tokens(content) > settings.tool_result_max_tokens:
        ratio = settings.tool_result_max_tokens / count_tokens(content)
        cut = int(len(content) * ratio * 0.95)
        content = content[:cut]
        truncated = True

    result: dict[str, Any] = {
        "path": str(resolved),
        "total_lines": total_lines,
        "returned_lines": f"{start}-{end}",
        "content": content,
    }
    if truncated:
        result["truncated"] = True
        result["note"] = "Output truncated. Use start_line/end_line to read specific sections."

    # Auto security scan
    ext = resolved.suffix.lower()
    if ext in settings.auto_scan_ext_set:
        try:
            from tools.code_scanner import scan_code_sync
            findings = scan_code_sync(content, filename=resolved.name)
            if findings:
                result["security_findings"] = findings
        except Exception as exc:
            logger.debug("Auto-scan failed for %s: %s", resolved, exc)

    return result


async def search_files(
    pattern: str,
    directory: str = ".",
    max_results: int = 50,
) -> dict[str, Any]:
    """Search for files matching a glob pattern."""
    base = Path(directory).expanduser().resolve()
    if not base.exists():
        return {"error": f"Directory not found: {base}"}
    if not base.is_dir():
        return {"error": f"Path is not a directory: {base}"}

    matches: list[str] = []
    try:
        for root, dirs, files in os.walk(base):
            # Skip hidden directories and common noise
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".git")]
            for fname in files:
                if fnmatch.fnmatch(fname, pattern) or fnmatch.fnmatch(
                    os.path.join(root, fname), pattern
                ):
                    rel = os.path.relpath(os.path.join(root, fname), base)
                    matches.append(rel)
                    if len(matches) >= max_results:
                        return {
                            "matches": matches,
                            "count": len(matches),
                            "truncated": True,
                            "note": f"Stopped at {max_results} results. Narrow your pattern or directory.",
                        }
    except PermissionError as exc:
        return {"error": str(exc)}

    return {"matches": matches, "count": len(matches), "directory": str(base)}
